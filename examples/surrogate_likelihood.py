"""CPU smoke check of a differentiable surrogate likelihood, not inference."""

import argparse
from pathlib import Path

import numpy as np
import torch

from metis import SimulationDataset, Surrogate, TrainingConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("likelihood-example"))
    args = parser.parse_args(argv)
    rng = np.random.default_rng(7)
    inputs = np.repeat(np.linspace(0.1, 0.9, 24)[:, None], 3, axis=0)
    times = np.arange(6, dtype=float)
    outputs = inputs[:, None, :] * (1 + times[None, :, None])
    outputs = outputs + rng.normal(0, 0.1, outputs.shape)
    bank = SimulationDataset.from_arrays(
        inputs, outputs, input_names=["theta"], output_names=["response"], times=times,
    )
    split = bank.split(validation_fraction=0.2, test_fraction=0.2, seed=42)
    model = Surrogate("itarflow", model_options={
        "channels": 16, "num_layers": 1, "num_blocks": 2,
        "num_heads": 2, "num_static_tokens": 2,
    }).fit(
        split.train, split.validation,
        config=TrainingConfig(steps=args.steps, batch_size=8, eval_every=5),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.pt")
    restored = Surrogate.load(args.output / "model.pt")
    theta = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    observation = torch.tensor(split.test.outputs[:1].copy())
    log_density = restored.log_prob_tensor(theta, observation)
    gradient, = torch.autograd.grad(log_density.sum(), theta)
    assert torch.isfinite(log_density).all() and torch.isfinite(gradient).all()
    np.testing.assert_allclose(
        log_density.detach().numpy(),
        restored.log_prob(theta.detach().numpy(), observation.numpy()),
        rtol=1e-5, atol=1e-5,
    )
    print({"log_density": log_density.tolist(), "gradient": gradient.tolist()})
    print("Likelihood/gradient smoke check only; no posterior was fitted.")


if __name__ == "__main__":
    main()
