import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from collections.abc import Mapping


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: int = 10000, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    # (1, seq_len, 1, head_dim//2) for broadcasting with (B, T, H, D)
    return cos[None, :, None, :], sin[None, :, None, :]


# TODO: How would we incorporate classifier-free guidance
@dataclass
class CrossAttentionConfig:
    n_head: int = 8
    cond_dim: int = 1  # will be inferred from data
    n_kv_head: int = 8
    qkv_bias: bool = True


def resolve_cross_attention_config(value: object) -> CrossAttentionConfig | None:
    """Accept plain or Hydra configuration mappings without importing Hydra."""
    if value is None or isinstance(value, CrossAttentionConfig):
        return value
    if isinstance(value, Mapping):
        return CrossAttentionConfig(**dict(value))
    raise TypeError("cross_attn must be a CrossAttentionConfig, a mapping, or None")


class CrossAttnKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # We fill the entire cross-attn kv cache once as we hit a layer for the first time
        self.populated = [False] * num_layers

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


# Cross-attention with RoPE, GQA support, QK-prenorm and KV cache
class CrossAttention(nn.Module):
    def __init__(
        self,
        config,  # untyped for now as it can be any of the models configs
        layer_idx: int,
        *,
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self.n_head = config.cross_attn.n_head
        self.n_kv_head = config.cross_attn.n_kv_head
        self.embd_dim = config.n_embd if embedding_dim is None else embedding_dim
        if self.n_head <= 0 or self.n_kv_head <= 0:
            raise ValueError("Cross-attention head counts must be positive")
        self.head_dim = self.embd_dim // self.n_head
        self.layer_idx = layer_idx

        # Ensure embedding dim is divisible by n_head
        if self.embd_dim % self.n_head or self.head_dim % 2:
            raise ValueError(
                "Cross-attention embedding dimension must be divisible by n_head with an even head dimension"
            )
        # Ensure number of kv heads is correct, this allows us to do MHA or GQA
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("Cross-attention n_kv_head must divide n_head")

        self.c_q = nn.Linear(
            self.embd_dim, self.n_head * self.head_dim, bias=config.cross_attn.qkv_bias
        )
        self.c_k = nn.Linear(
            self.embd_dim,
            self.n_kv_head * self.head_dim,
            bias=config.cross_attn.qkv_bias,
        )
        self.c_v = nn.Linear(
            self.embd_dim,
            self.n_kv_head * self.head_dim,
            bias=config.cross_attn.qkv_bias,
        )
        # TODO: Do we want bias=False?
        self.c_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        # Zero-init output projection so cross-attention starts as a no-op
        nn.init.zeros_(self.c_proj.weight)

    # TODO: Support sparse conditioning sequences
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None,
        cos_sin_x: tuple[torch.Tensor, torch.Tensor],
        cos_sin_cond: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: CrossAttnKVCache | None = None,
        current_x_pos: int | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

        # Shared RoPE so that conditioning and x tokens share
        # the same positional representation
        # e.g. a vector at timestep t=2 in the cond sequence should have
        # the same position as a vector at timestep t=2 in the x sequence
        q = apply_rotary_emb(q, *cos_sin_x)
        q = norm(q)

        if kv_cache is None or not kv_cache.populated[self.layer_idx]:
            if cond is None:
                raise ValueError(
                    "A conditioning sequence is required to populate cross-attention"
                )
            B_cond, T_cond, C_cond = cond.size()
            if B_cond != B or C_cond != self.embd_dim or T_cond == 0:
                raise ValueError(
                    "Conditioning must match the query batch and embedding dimensions and contain at least one token"
                )
            assert cos_sin_cond is not None, (
                "Need cos_sin for conditioning sequence when kv cache is none or unpopulated"
            )
            k = self.c_k(cond).view(B, T_cond, self.n_kv_head, self.head_dim)
            v = self.c_v(cond).view(B, T_cond, self.n_kv_head, self.head_dim)
            k = apply_rotary_emb(k, *cos_sin_cond)
            k = norm(k)

            if (
                kv_cache is not None
            ):  # We only need to cache once and then we can just read
                kv_cache.k_cache[self.layer_idx] = k
                kv_cache.v_cache[self.layer_idx] = v
                kv_cache.populated[self.layer_idx] = True
        else:
            assert current_x_pos is not None, (
                "Need current x pos when using cross-attention KV-cache to correctly slice."
            )
            k, v = kv_cache.get_layer_cache(self.layer_idx)

        # Positions refer to the final query in this call. A cached chunk can
        # start after zero; SDPA's built-in causal mask always starts at zero.
        # Align an explicit mask to absolute positions for these chunks.
        query_start = 0 if current_x_pos is None else current_x_pos - T + 1
        if query_start < 0:
            raise ValueError("current_x_pos must include every query token")
        attn_mask = None
        use_causal = query_start == 0
        if query_start:
            key_positions = torch.arange(k.size(1), device=x.device)
            query_positions = query_start + torch.arange(T, device=x.device)
            attn_mask = key_positions[None, :] <= query_positions[:, None]

        # (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        enable_gqa = self.n_head != self.n_kv_head

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=use_causal, enable_gqa=enable_gqa
        )

        # Same length as x
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
