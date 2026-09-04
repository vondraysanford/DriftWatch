"""Point serving at a registry version: tag it ``stage=champion`` and demote the previous champion.

    python -m training.promote --version 2
    python -m training.promote --show

Promotion is the one step in the retrain loop that a human approves (the deploy workflow's
``production`` environment). Registering a challenger never changes what ships; this does.
"""

from __future__ import annotations

import argparse
import logging

from mlflow.tracking import MlflowClient

from training.common import configure_tracking, require_env, setup_logging

log = logging.getLogger("driftwatch.promote")


def versions_by_stage(client: MlflowClient, model_name: str) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    for mv in client.search_model_versions(f"name='{model_name}'"):
        stages.setdefault(mv.tags.get("stage", "untagged"), []).append(str(mv.version))
    return stages


def champion_version(client: MlflowClient, model_name: str) -> str:
    champions = versions_by_stage(client, model_name).get("champion", [])
    if len(champions) != 1:
        raise SystemExit(f"expected exactly one version of {model_name!r} tagged stage=champion, found {champions}; run training.promote")
    return champions[0]


def promote(client: MlflowClient, model_name: str, version: str) -> None:
    client.get_model_version(model_name, version)  # fails loudly if the version does not exist
    for previous in versions_by_stage(client, model_name).get("champion", []):
        if previous != version:
            client.set_model_version_tag(model_name, previous, "stage", "previous")
            log.info("demoted version %s to stage=previous", previous)
    client.set_model_version_tag(model_name, version, "stage", "champion")
    log.info("version %s is now the champion of %s", version, model_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="registry version to make the champion")
    parser.add_argument("--show", action="store_true", help="print versions by stage and exit")
    args = parser.parse_args(argv)

    configure_tracking()
    model_name = require_env("DRIFTWATCH_MODEL_NAME")
    client = MlflowClient()
    if args.show or not args.version:
        for stage, versions in sorted(versions_by_stage(client, model_name).items()):
            print(f"{stage:>10}: {', '.join(sorted(versions, key=int))}")
        return
    promote(client, model_name, args.version)


if __name__ == "__main__":
    setup_logging()
    main()
