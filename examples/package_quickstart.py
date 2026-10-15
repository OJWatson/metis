"""Executable beginner example, also run against the installed package in CI."""

import argparse
from pathlib import Path

from metis.cli import run_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/package-quickstart"))
    args = parser.parse_args()
    run_demo(args.output, args.steps)
