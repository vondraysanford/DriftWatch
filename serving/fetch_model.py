"""Build-time pull of the registered model into serving/model/ (gitignored).

    python -m serving.fetch_model                 # latest version of DRIFTWATCH_MODEL_NAME
    python -m serving.fetch_model --version 1

Runs before ``docker build`` (locally with az login, in CI after azure/login via OIDC). The image
copies the directory, so the container never touches the registry and needs no credential to
load its model. model_info.json carries the provenance the app serves at /model.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from serving.schemas import ModelInfo

log = logging.getLogger("driftwatch.fetch_model")

# Deliberately standalone: this runs in the CI build step, so it must not drag in the training
# stack (matplotlib, xgboost, scikit-learn) to download one model. The helpers below are the
# small duplicated cousins of the ones in training/common.py.


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    for noisy in ("azure", "azureml", "urllib3", "msal"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

DEFAULT_OUT = Path("serving/model")
SERVING_REQUIREMENTS = Path("serving/requirements.txt")
# Estimator libraries the unpickled model may need; the image must pin whichever the model uses.
MODEL_LIBRARIES = ("scikit-learn", "xgboost", "lightgbm")


def latest_version(client: MlflowClient, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"no versions registered under {name!r}")
    return str(max(int(v.version) for v in versions))


def check_image_can_load(model_dir: Path) -> None:
    """Fail the build early if the model needs an estimator library the serving image does not pin."""
    wanted = {line.split("==")[0].strip().lower() for line in (model_dir / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    pinned = {line.split("==")[0].split("[")[0].strip().lower() for line in SERVING_REQUIREMENTS.read_text().splitlines()
              if "==" in line}
    missing = [lib for lib in MODEL_LIBRARIES if lib in wanted and lib not in pinned]
    if missing:
        raise SystemExit(f"model needs {missing} but {SERVING_REQUIREMENTS} does not pin it")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="registry version (default: highest version number)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(require_env("MLFLOW_TRACKING_URI"))
    name = require_env("DRIFTWATCH_MODEL_NAME")
    client = MlflowClient()
    version = args.version or latest_version(client, name)
    mv = client.get_model_version(name, version)
    run = client.get_run(mv.run_id)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}", dst_path=str(args.out / "model"))
    # download_artifacts nests the flavor under the last URI segment; flatten to <out>/model/MLmodel
    nested = args.out / "model" / "model"
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), args.out / "model" / item.name)
        nested.rmdir()

    info = ModelInfo(
        name=name,
        version=version,
        run_id=mv.run_id,
        run_name=run.data.tags.get("mlflow.runName"),
        model_kind=run.data.tags.get("model_kind"),
        threshold=float(run.data.params["operating_threshold"]),
        metrics={k: v for k, v in run.data.metrics.items() if k.startswith(("test_", "holdout_"))},
        data_version=run.data.tags.get("data_version"),
        git_commit=run.data.tags.get("git_commit"),
        fetched_at=datetime.now(timezone.utc),
    )
    (args.out / "model_info.json").write_text(info.model_dump_json(indent=2) + "\n")
    check_image_can_load(args.out / "model")
    log.info("fetched %s version %s (run %s, threshold %.4f) into %s", name, version, mv.run_id, info.threshold, args.out)


if __name__ == "__main__":
    setup_logging()
    main()
