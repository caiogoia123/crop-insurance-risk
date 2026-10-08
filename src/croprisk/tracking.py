"""Thin MLflow wrapper. Tracking stays local (mlruns/, git-ignored); if MLflow is
not installed (e.g. the slim production image) every call is a no-op."""

from __future__ import annotations

import contextlib
import os

from croprisk import config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
TRACKING_DIR = config.ROOT / "mlruns"


def setup(experiment: str) -> bool:
    if mlflow is None:
        return False
    TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(TRACKING_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(experiment)
    return True


@contextlib.contextmanager
def run(name: str, nested: bool = False, **tags):
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=name, nested=nested, tags=tags or None) as r:
        yield r


def log_params(d: dict) -> None:
    if mlflow is not None:
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})


def log_metrics(d: dict, step: int | None = None) -> None:
    if mlflow is not None:
        clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}
        mlflow.log_metrics(clean, step=step)


def log_artifact(path) -> None:
    if mlflow is not None:
        mlflow.log_artifact(str(path))
