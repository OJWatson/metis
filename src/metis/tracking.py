"""Optional live Weights & Biases logging for Python and Hydra workflows.

Only the supplied configuration and scalar training/evaluation metrics are
logged. This adapter does not upload simulation tables, predictions, or model
artifacts. Importing METIS or this module never imports W&B or starts a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import warnings


class WandbTracker:
    """Manage an optional W&B run and provide a ``Surrogate.fit`` callback.

    Use ``mode="offline"`` to keep tracking local without an account or network.
    ``enabled=False`` is a complete no-op and does not require the W&B package.
    Online runs use credentials configured with ``wandb login`` or W&B's usual
    environment variables; METIS never requests or stores credentials itself.
    The context finishes failed runs with a nonzero exit status and propagates
    the original training exception.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        project: str = "metis",
        entity: str | None = None,
        name: str | None = None,
        mode: str = "online",
        dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be True or False")
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("W&B mode must be online, offline, or disabled")
        if not isinstance(project, str) or not project:
            raise ValueError("W&B project must be a nonempty string")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("W&B config must be a mapping of experiment metadata")
        if tags is not None and (
            not isinstance(tags, (tuple, list))
            or any(not isinstance(tag, str) for tag in tags)
        ):
            raise TypeError("W&B tags must be a list of strings")
        self.enabled = enabled
        self._options: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "name": name,
            "mode": mode,
            "dir": str(Path(dir).resolve()) if dir is not None else None,
            "config": dict(config) if config is not None else None,
            "tags": list(tags) if tags is not None else None,
        }
        self._run: Any = None
        self._entered = False
        self._active = False

    def __enter__(self) -> WandbTracker:
        if self._entered:
            raise RuntimeError("Create a new WandbTracker for each experiment")
        self._entered = True
        if not self.enabled:
            return self
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging requires: uv add "
                "'metis-surrogate[wandb] @ git+https://github.com/OJWatson/metis.git'. "
                "Use enabled=False to train without W&B."
            ) from exc
        if self._options["dir"] is not None:
            Path(self._options["dir"]).mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(**self._options)
        self._active = True
        return self

    def _require_active(self) -> None:
        if self.enabled and not self._active:
            raise RuntimeError("Use WandbTracker in a 'with' block before logging")

    def log_training(self, record: dict) -> None:
        """Log one completed optimizer step; pass directly as ``on_progress``."""
        if not self.enabled:
            return
        self._require_active()
        if not isinstance(record, dict) or "step" not in record:
            raise ValueError("Training progress must be a dictionary with a step")
        step = record["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("Training progress step must be a positive integer")
        # A new dictionary prevents callbacks/backend mutation of fit history.
        self._run.log(dict(record), step=step)

    def log_evaluation(self, report: Any, *, prefix: str = "test") -> None:
        """Log report scalar metrics under a named evaluation partition."""
        if not self.enabled:
            return
        self._require_active()
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("Evaluation prefix must be a nonempty string")
        self._run.log(
            {f"{prefix}/{key}": value for key, value in report.metrics.items()}
        )

    def finish(self, *, exit_code: int = 0) -> None:
        """Finish once; a context manager normally handles this automatically."""
        if self._active:
            self._active = False
            self._run.finish(exit_code=exit_code)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.finish(exit_code=1 if exc_type is not None else 0)
        except Exception as cleanup_error:
            if exc_type is None:
                raise
            warnings.warn(
                f"W&B cleanup also failed: {cleanup_error}. The original exception is preserved.",
                RuntimeWarning,
                stacklevel=2,
            )
        return False
