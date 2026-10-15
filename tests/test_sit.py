import pytest
import torch
from metis.models.sit import SiT, SiTConfig, InterpolantTransformer, LinearInterpolant
from metis.models.utils.cross_attention import CrossAttention, CrossAttentionConfig


def make_tiny_config():
    return SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.1,
        in_and_out_biases=True,
    )


def test_forward_pass_shape():
    """SiT output shape should match (B, T, input_dim)."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    out = model(x, t, y)
    assert out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {out.shape}"
    )


def test_zero_init_output():
    """Final layer is zero-initialised, so SiT should output near-zero at init."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    model.eval()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        out = model(x, t, y)
    assert out.abs().max() < 1e-5, (
        f"Expected near-zero output at init, got max {out.abs().max()}"
    )


def test_gradient_flow():
    """Backprop through SiT should produce non-zero gradients.

    At init, adaLN zero-init means gates are zero, so gradients don't flow
    back through the blocks to input_projection. We check final_layer gets
    gradients at init, then verify input_projection gets gradients after one
    optimizer step (once gates are non-zero).
    """
    cfg = make_tiny_config()
    model = SiT(cfg)
    model.train()
    B, T = 2, cfg.sequence_len
    x = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(B)
    y = torch.randn(B, cfg.cond_dim)

    # At init: final layer should get gradients even with zero-init
    model.zero_grad()
    out = model(x, t, y)
    loss = out.sum()
    loss.backward()
    assert model.final_layer.linear.weight.grad is not None
    assert torch.any(model.final_layer.linear.weight.grad != 0), (
        "final_layer should get non-zero gradients at init"
    )

    # After one optimizer step, gates open up and input_projection should get gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.step()
    model.zero_grad()
    out = model(x, t, y)
    out.sum().backward()
    assert model.input_projection.weight.grad is not None
    assert torch.any(model.input_projection.weight.grad != 0), (
        "input_projection should get non-zero gradients after one step"
    )


def test_conditioning_dropout():
    """With force_drop_ids=1, conditioning should be the null embedding (zeros)."""
    cfg = make_tiny_config()
    model = SiT(cfg)
    B = 3
    cond = torch.randn(B, cfg.cond_dim)

    # Force all dropped
    force_drop = torch.ones(B)
    dropped = model.cond_embedder(cond, train=False, force_drop_ids=force_drop)
    assert torch.allclose(dropped, torch.zeros_like(dropped)), (
        "Dropped embeddings should be zero"
    )

    # Force none dropped
    force_keep = torch.zeros(B)
    kept = model.cond_embedder(cond, train=False, force_drop_ids=force_keep)
    assert not torch.allclose(kept, torch.zeros_like(kept)), (
        "Kept embeddings should be non-zero"
    )


def test_flow_matching_loss_finite_and_differentiable():
    """calc_loss should return a finite scalar with gradients."""
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()
    B, T = 4, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)

    model.zero_grad()

    loss = model.compute_loss(static_input=y, x=None, targets=x_1)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"


def test_sampling_shape():
    """sampling should produce output with the same shape as input."""
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.eval()
    B, T = 2, cfg.sequence_len
    x_0 = torch.randn(B, T, cfg.data_dim)
    y = torch.randn(B, cfg.cond_dim)
    with torch.no_grad():
        x_out, x_in = model.sample(x_0, y, num_steps=5, method="euler")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )

    with torch.no_grad():
        x_out, x_in = model.sample(x_0, y, num_steps=5, method="heun")
    assert x_out.shape == (B, T, cfg.data_dim), (
        f"Expected {(B, T, cfg.data_dim)}, got {x_out.shape}"
    )


def test_single_batch_overfit():
    """FlowMatchingTransformer should memorise a single batch."""
    torch.manual_seed(42)

    cfg = SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
    )
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()

    assert (
        cfg.sequence_len is not None
        and cfg.data_dim is not None
        and cfg.cond_dim is not None
    )
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    I_t = model.interpolant.calc_It(x_0, x_1, t[:, None, None])
    dot_I_t = model.interpolant.calc_It_dot(x_0, x_1, t[:, None, None])

    # We don't have control of the rng when we use .compute_loss for a SiT
    # so we lift the loss function out

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(200):
        optimizer.zero_grad()
        v_t = model.v_t(I_t, t, y)
        loss = torch.nn.functional.mse_loss(v_t, dot_I_t)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 1e-3, f"Model failed to overfit single batch, loss={final_loss}"


def test_cross_attention_training():
    torch.manual_seed(42)
    seq_cond_dim = 2
    cfg = SiTConfig(
        sequence_len=16,
        data_dim=4,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=4,
        qkv_bias=True,
        embd_dim=32,
        cond_dropout_prob=0.0,
        in_and_out_biases=True,
        cross_attn=CrossAttentionConfig(n_head=4, cond_dim=seq_cond_dim, n_kv_head=4),
    )
    model = InterpolantTransformer(cfg, LinearInterpolant())
    model.train()
    B, T = 8, cfg.sequence_len
    x_1 = torch.randn(B, T, cfg.data_dim)
    x_0 = torch.randn(B, T, cfg.data_dim)
    t = torch.rand(
        B,
    )
    y = torch.randn(B, cfg.cond_dim)
    cond_seq = torch.randn(B, cfg.sequence_len, seq_cond_dim)

    # Forward pass produces correct shape
    I_t = model.interpolant.calc_It(x_0, x_1, t[:, None, None])
    output = model.v_t(I_t, t, y, cond_seq=cond_seq)
    assert output.shape == (B, cfg.sequence_len, cfg.data_dim)

    # Loss computation works
    dot_I_t = model.interpolant.calc_It_dot(x_0, x_1, t[:, None, None])
    loss = torch.nn.functional.mse_loss(output, dot_I_t)
    assert loss.ndim == 0, "Loss not a scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    # Backward pass produces gradients and some are non-zero
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have non-zero gradients"


@pytest.mark.parametrize("cross_heads", [2, 8])
def test_cross_attention_supports_different_head_dimensions(cross_heads):
    cfg = SiTConfig(
        sequence_len=8,
        data_dim=2,
        cond_dim=3,
        n_blocks=2,
        n_head=4,
        n_kv_head=2,
        embd_dim=32,
        cond_dropout_prob=0,
        cross_attn=CrossAttentionConfig(n_head=cross_heads, n_kv_head=2, cond_dim=1),
    )
    model = InterpolantTransformer(cfg, LinearInterpolant())
    x = torch.randn(2, 8, 2)
    observed = torch.randn(2, 8, 1)
    y = torch.randn(2, 3)
    with torch.no_grad():
        torch.nn.init.normal_(model.sit.final_layer.linear.weight, std=0.1)
    out = model.v_t(x, torch.rand(2), y, observed)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert torch.any(model.sit.blocks[0].cross_attn.c_proj.weight.grad != 0)
    generated, original = model.sample(
        x, y, num_steps=2, method="heun", cond_seq=observed
    )
    assert generated.shape == x.shape
    assert torch.isfinite(generated).all()


@pytest.mark.parametrize("num_steps", [0, -1, 1.5, True])
def test_invalid_sampling_steps(num_steps):
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    with pytest.raises(ValueError, match="positive integer"):
        model.sample(
            torch.randn(1, cfg.sequence_len, cfg.data_dim),
            torch.randn(1, cfg.cond_dim),
            num_steps=num_steps,
        )


def test_unknown_solver_is_rejected():
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    with pytest.raises(ValueError, match="Solver"):
        model.sample(
            torch.randn(1, cfg.sequence_len, cfg.data_dim),
            torch.randn(1, cfg.cond_dim),
            method="typo",
        )


def test_heun_corrects_final_step():
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    # For x'=x, one unit Euler step gives 2x; Heun gives 2.5x.
    model.v_t = lambda x, t, y, cond_seq=None: x
    initial = torch.ones(1, cfg.sequence_len, cfg.data_dim)
    y = torch.zeros(1, cfg.cond_dim)
    euler, _ = model.sample(initial, y, num_steps=1, method="euler")
    heun, original = model.sample(initial, y, num_steps=1, method="heun")
    torch.testing.assert_close(euler, 2 * initial)
    torch.testing.assert_close(heun, 2.5 * initial)
    torch.testing.assert_close(original, initial)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_head": 3},
        {"embd_dim": 24, "n_head": 8},
        {"n_blocks": 0},
        {"n_kv_head": 3},
        {"sequence_len": 0},
        {"cond_dropout_prob": 1.1},
    ],
)
def test_invalid_sit_config(kwargs):
    with pytest.raises(ValueError):
        SiTConfig(**kwargs)


def test_sampling_rejects_wrong_sequence_shape():
    cfg = make_tiny_config()
    model = InterpolantTransformer(cfg, LinearInterpolant())
    with pytest.raises(ValueError, match="x_0 must have shape"):
        model.sample(
            torch.zeros(2, cfg.sequence_len - 1, cfg.data_dim),
            torch.zeros(2, cfg.cond_dim),
        )
    with pytest.raises(ValueError, match="y must have shape"):
        model.sample(
            torch.zeros(2, cfg.sequence_len, cfg.data_dim), torch.zeros(1, cfg.cond_dim)
        )
