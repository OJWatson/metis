"""Console commands must succeed in a fresh process outside the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_demo_console_exit_status_and_artifacts(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(Path(sys.executable).with_name("metis")), "demo", "--steps", "1",
         "--output", str(tmp_path / "demo")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["prediction_shape"] == [1, 40, 4]
    assert Path(report["artifact"]).is_file()
    assert (tmp_path / "demo" / "metrics.json").is_file()
    assert (tmp_path / "demo" / "prediction.npy").is_file()
