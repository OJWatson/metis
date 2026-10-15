"""A reproducible CSV-to-surrogate handoff using a small synthetic bank.

The toy recurrence is an installation exercise, not an ABM benchmark. Replace
its CSV with your own bank only after adapting the explicit column lists.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from metis import SimulationDataset, Surrogate, TrainingConfig


INPUTS = ["decay", "forcing"]
OUTPUTS = ["stock", "flux"]


def make_bank(path):
    """Write 30 scenarios, three independent realizations and 12 timepoints."""
    rng = np.random.default_rng(42)
    rows = []
    for decay in np.linspace(0.05, 0.5, 6):
        for forcing in np.linspace(0.2, 1.0, 5):
            for rep in range(3):
                stock = 10.0
                for t in range(12):
                    flux = decay * stock + rng.normal(0, 0.05)
                    rows.append({
                        "decay": float(decay), "forcing": float(forcing),
                        "rep": rep, "t": t, "stock": stock, "flux": flux,
                    })
                    stock = (1 - decay) * stock + forcing + rng.normal(0, 0.1)
    pl.DataFrame(rows).write_csv(path)


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
    print(report.by_output)
    print("Synthetic installation exercise; no scientific accuracy claim.")


if __name__ == "__main__":
    main()
