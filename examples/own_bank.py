"""A reproducible CSV-to-surrogate handoff using the bundled SIRDS bank.

Replace its CSV with your own bank after adapting the explicit column lists.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from metis import SimulationDataset, Surrogate, TrainingConfig, load_example


INPUTS = ["I_0", "beta", "gamma", "omega", "mu"]
OUTPUTS = ["S_t", "I_t", "R_t", "D_t"]


def make_bank(path):
    """Write the same 100 SIRDS settings × five realizations × 40 days as the docs."""
    bank = load_example()
    frames = []
    for inputs, outputs, rep in zip(bank.inputs, bank.outputs, bank.replicate_ids):
        frames.append(pl.DataFrame({
            **{name: outputs[:, i] for i, name in enumerate(OUTPUTS)},
            **{name: np.repeat(value, len(bank.times))
               for name, value in zip(INPUTS, inputs)},
            "t": bank.times,
            "rep": np.repeat(rep, len(bank.times)),
        }))
    pl.concat(frames).write_csv(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("own-bank"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--make-bank-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or args.output / "bank.csv"
    if args.csv is None:
        make_bank(csv_path)
    frame = pl.read_csv(csv_path)
    bank = SimulationDataset.from_frame(frame, inputs=INPUTS, outputs=OUTPUTS)
    frame.write_parquet(args.output / "bank.parquet")
    print(f"Validated {len(bank)} runs on {len(bank.times)} timepoints")
    if args.make_bank_only:
        return
    split = bank.split(validation_fraction=0.2, test_fraction=0.2, seed=42)
    model = Surrogate("mlp").fit(
        split.train, split.validation,
        config=TrainingConfig(steps=args.steps, eval_every=max(1, args.steps // 5)),
    )
    report = model.evaluate(split.test)
    report.save(args.output / "evaluation.json")
    model.save(args.output / "model.pt")
    restored = Surrogate.load(args.output / "model.pt")
    query = dict(zip(bank.input_names, split.test.inputs[0].tolist()))
    prediction = restored.predict(query)
    assert prediction.shape == (1, len(bank.times), len(bank.output_names))
    assert np.isfinite(prediction).all()
    np.save(args.output / "prediction.npy", prediction)
    print(json.dumps(report.by_output.to_dicts(), indent=2))
    print("SIRDS installation exercise; assess accuracy before downstream use.")


if __name__ == "__main__":
    main()
