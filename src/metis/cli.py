"""Small command-line entrypoints; Hydra owns training configuration."""

import argparse
import json
from pathlib import Path
import sys


def run_demo(output, steps=100):
    """Run the complete installed-package workflow on a bundled example."""
    from metis import Surrogate, TrainingConfig, load_example

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_example()
    split = data.split(validation_fraction=.2, test_fraction=.2, seed=42)
    surrogate = Surrogate("mlp").fit(
        split.train, validation=split.validation,
        config=TrainingConfig(steps=steps, eval_every=max(1, steps//5)),
    )
    surrogate.save(output / "model.pt")
    report = surrogate.evaluate(split.test)
    report.save(output / "metrics.json")
    restored = Surrogate.load(output / "model.pt")
    import numpy as np
    prediction = restored.predict(split.test.inputs[:1])
    np.save(output / "prediction.npy", prediction)
    summary = {
        "model": surrogate.model_name,
        "training_fingerprint": split.train.fingerprint,
        "evaluation_fingerprint": split.test.fingerprint,
        "metrics": report.metrics,
        "artifact": str((output / "model.pt").resolve()),
        "prediction_shape": list(prediction.shape),
        "interpretation": "Installation example; training quality requires application-specific validation.",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        from metis.hydra_app import main as hydra_main
        # Pass the original Hydra overrides/flags through unchanged.
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return hydra_main()
    parser = argparse.ArgumentParser(
        prog="metis",
        description="Neural surrogates for simulation banks. Use 'metis train --help' for Hydra.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train using Hydra configuration and --multirun sweeps")
    demo = sub.add_parser("demo", help="Train, evaluate, save and reload a CPU example")
    demo.add_argument("--steps", type=int, default=100)
    demo.add_argument("--output", type=Path, default=Path("metis-demo"))
    inspect = sub.add_parser("inspect-data", help="Validate a long-form Parquet bank")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--inputs", nargs="+", required=True)
    inspect.add_argument("--outputs", nargs="+", required=True)
    inspect.add_argument("--groups", nargs="+")
    inspect.add_argument("--time", default="t")
    inspect.add_argument("--replicate", default="rep")
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output, args.steps)
        return None  # Console scripts pass this value to sys.exit().
    from metis import SimulationDataset
    data = SimulationDataset.from_parquet(
        args.path, inputs=args.inputs, outputs=args.outputs, groups=args.groups,
        time=args.time, replicate=args.replicate,
    )
    print(json.dumps({
        "runs": len(data), "groups": len(set(data.group_keys)),
        "inputs": list(data.input_names), "outputs": list(data.output_names),
        "timesteps": len(data.times), "fingerprint": data.fingerprint,
        "status": "valid",
    }, indent=2))
