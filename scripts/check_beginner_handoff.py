"""Run an automated beginner handoff in a fresh uv project using a built wheel.

This tests commands and artifacts; it does not substitute for a human trial.
Run with --wheel dist/<wheel-name>.whl after `uv build`.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", default="3.13")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve(strict=True)
    example = Path(__file__).resolve().parents[1] / "examples" / "own_bank.py"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    with tempfile.TemporaryDirectory(prefix="metis-handoff-") as temporary:
        project = Path(temporary) / "my-surrogate"

        def run(*command):
            subprocess.run(command, cwd=project, env=env, check=True, timeout=240)

        subprocess.run(
            ["uv", "init", str(project), "--python", args.python],
            env=env, check=True, timeout=120,
        )
        # Keep torch resolution explicit and platform-aware in this fresh project.
        with (project / "pyproject.toml").open("a") as stream:
            stream.write('\n[[tool.uv.index]]\nname = "pytorch-cpu"\n'
                         'url = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n'
                         '\n[tool.uv.sources]\ntorch = [{ index = "pytorch-cpu", '
                         'marker = "sys_platform == \'linux\' or sys_platform == \'win32\'" }]\n')
        run("uv", "add", "torch>=2.10,<3")
        run("uv", "add", str(wheel))
        run("uv", "run", "--locked", "python", "-c",
            "import metis; from pathlib import Path; "
            "assert Path(metis.__file__).resolve().is_relative_to(Path('.venv').resolve()), metis.__file__")
        run("uv", "run", "--locked", "python", "-c",
            "from metis import load_example; "
            "bank=load_example(); "
            "assert bank.inputs.shape == (5000,5) and bank.outputs.shape == (5000,200,4)")
        shutil.copy2(example, project / "own_bank.py")
        run("uv", "run", "--locked", "python", "own_bank.py", "--make-bank-only", "--small")
        run("uv", "run", "--locked", "metis", "inspect-data", "own-bank/bank.parquet",
            "--inputs", "I_0", "beta", "gamma", "omega", "mu",
            "--outputs", "S_t", "I_t", "R_t", "D_t")
        run("uv", "run", "--locked", "python", "own_bank.py", "--csv", "own-bank/bank.csv", "--steps", "5")
        # A second process reloads without reading the bank.
        run("uv", "run", "--locked", "python", "-c",
            "from metis import Surrogate; import numpy as np; "
            "m=Surrogate.load('own-bank/model.pt'); "
            "p=m.predict({'I_0':0.05,'beta':0.4,'gamma':0.15,'omega':0.02,'mu':0.02}); "
            "assert p.shape == (1,40,4) and np.isfinite(p).all()")
        # Installed Hydra exercises conditioning and strict checkpoint resume.
        hydra_arguments = [
            "dataset=parquet", "dataset.path=own-bank/bank.parquet",
            "dataset.inputs=[I_0,beta,gamma,omega,mu]", "dataset.outputs=[S_t,I_t,R_t,D_t]",
            "model=transformer", "model.observed_outputs=[D_t]",
            "model.model_options.n_embd=16", "model.model_options.n_layer=1",
            "training.steps=2", "training.batch_size=8", "training.eval_every=1",
            "training.checkpoint_every=1", f"training.checkpoint_dir={project / 'checkpoints'}",
        ]
        run("uv", "run", "--locked", "metis", "train", *hydra_arguments,
            f"hydra.run.dir={project / 'conditioned-fit'}")
        run("uv", "run", "--locked", "metis", "train", *hydra_arguments,
            "training.resume_from=checkpoints/last.pt",
            f"hydra.run.dir={project / 'conditioned-resume'}")
        run("uv", "run", "--locked", "python", "-c",
            "from metis import Surrogate, load_example; import numpy as np; "
            "a=Surrogate.load('conditioned-fit/model.pt'); "
            "b=Surrogate.load('conditioned-resume/model.pt'); "
            "data=load_example(small=True).split(seed=42).test; "
            "x=data.inputs[:1]; c=data.outputs[:1,:,-1:]; "
            "p=a.predict(x,conditioning=c); "
            "assert p.shape == (1,40,3) and np.isfinite(p).all(); "
            "np.testing.assert_allclose(p,b.predict(x,conditioning=c),rtol=1e-6,atol=1e-6)")
        for name in ("model.pt", "evaluation.json", "prediction.npy"):
            if not (project / "own-bank" / name).is_file():
                raise RuntimeError(f"Handoff missing {name}")
    print("Automated fresh-project handoff passed; no human trial is implied.")


if __name__ == "__main__":
    main()
