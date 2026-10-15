"""Exercise public API, all model families, precision and checkpoint resume.

The JSON result records actual hardware checks. CUDA must be explicitly selected;
this script fails when it is unavailable instead of silently checking CPU.
"""

import argparse
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from metis import SimulationDataset, Surrogate, TrainingConfig, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("hardware-check.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA explicitly requested but unavailable; no GPU validation performed"
        )
    rng = np.random.default_rng(7)
    inputs = np.arange(24, dtype=np.float32).reshape(12, 2)
    outputs = rng.normal(size=(12, 4, 2)).astype(np.float32)
    bank = SimulationDataset.from_arrays(
        inputs,
        outputs,
        input_names=["a", "b"],
        output_names=["u", "v"],
        times=[0, 1, 2, 3],
    ).split()
    architectures = {
        "mlp": {"hidden_dim": 16, "num_blocks": 1},
        "transformer": {"n_embd": 16, "n_layer": 1},
        "flow_matching": {"embd_dim": 16, "n_blocks": 1},
        "itarflow": {"channels": 16, "num_blocks": 2, "num_layers": 1},
    }
    checks = {}
    with tempfile.TemporaryDirectory(prefix="metis-hardware-") as temporary:
        for family, options in architectures.items():
            checkpoint_dir = str(Path(temporary) / family)
            settings = dict(
                steps=3,
                batch_size=4 * args.num_devices,
                eval_every=1,
                device=args.device,
                num_devices=args.num_devices,
                precision=args.precision,
                compile_model=args.compile,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            model = Surrogate(family, options).fit(
                bank.train, bank.validation, TrainingConfig(**settings)
            )
            x = bank.train.inputs[:2]
            kwargs = dict(n_samples=3, num_steps=2) if model.is_stochastic else {}
            predictions = model.predict(x, **kwargs)
            restored = Surrogate.load(
                model.save(Path(temporary) / f"{family}.pt"), device=args.device
            )
            np.testing.assert_allclose(
                predictions, restored.predict(x, **kwargs), rtol=1e-5, atol=1e-5
            )
            resumed = Surrogate(family, options).fit(
                bank.train,
                bank.validation,
                TrainingConfig(
                    **settings, resume_from=str(Path(checkpoint_dir) / "last.pt")
                ),
            )
            np.testing.assert_allclose(
                predictions, resumed.predict(x, **kwargs), rtol=1e-5, atol=1e-5
            )
            checks[family] = {
                "steps": len(model.history),
                "save_reload": "passed",
                "completed_checkpoint_resume": "passed",
            }
    result = {
        "metis": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": args.device,
        "num_devices": args.num_devices,
        "precision": args.precision,
        "compiled": args.compile,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "checks": checks,
        "scope": "synthetic engineering acceptance; not scientific validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
