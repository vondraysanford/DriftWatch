"""Fire a GitHub repository_dispatch event when the drift verdict says so.

    python -m monitoring.retrain_trigger --verdict monitoring/out/verdict.json
    python -m monitoring.retrain_trigger --event model-registered --payload '{"version": "2"}'

The built-in GITHUB_TOKEN cannot start workflows, so this uses a fine-grained personal access
token scoped to this repository (Contents: read and write), stored as the Actions secret
RETRAIN_DISPATCH_TOKEN. That is a GitHub credential, not a cloud one, so the "no stored cloud
secrets" claim still holds (decision 12). Standard library only; nothing to install in CI.

Exit code is 0 whether or not an event was sent; "no drift, nothing to do" is a normal outcome.
A failed HTTP call exits non-zero, because a lost dispatch is a silent gap in the loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("driftwatch.monitoring.trigger")

API = "https://api.github.com"


def dispatch(repo: str, token: str, event_type: str, payload: dict) -> None:
    body = json.dumps({"event_type": event_type, "client_payload": payload}).encode("utf-8")
    request = urllib.request.Request(
        f"{API}/repos/{repo}/dispatches",
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"dispatch failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}") from exc
    if status != 204:
        raise SystemExit(f"dispatch returned HTTP {status}, expected 204")
    log.info("dispatched %r to %s with payload %s", event_type, repo, json.dumps(payload))


def payload_from_verdict(verdict: dict) -> dict:
    """The parts of a verdict a retrain workflow wants to show in its summary. Kept small: GitHub
    caps client_payload at 10 top-level properties."""
    raw = verdict.get("raw", {})
    perf = verdict.get("performance") or {}
    drifted_parts = [p["regime"] for p in verdict.get("parts", []) if p.get("drift")]
    # The model's score on the traffic that drifted, not on the whole window.
    regime_auc = next((perf.get("by_regime", {}).get(r, {}).get("roc_auc_current") for r in drifted_parts), None)
    return {
        "drift": verdict["drift"],
        "drifted_regimes": ",".join(drifted_parts),
        "raw_drifted_share": raw.get("drifted_share"),
        "raw_drifted_columns": ",".join(raw.get("drifted_columns", []))[:400],
        "settings_drifted": ",".join(verdict.get("settings_drifted", [])),
        "current_records": verdict.get("current", {}).get("records"),
        "window_hours": verdict.get("current", {}).get("window_hours"),
        "regime_roc_auc": regime_auc if regime_auc is not None else perf.get("roc_auc_current"),
        "reference_roc_auc": perf.get("roc_auc_reference"),
        "reason": verdict.get("reason", "")[:400],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verdict", type=Path, help="verdict.json from monitoring.drift; dispatches only if drift is true")
    parser.add_argument("--event", default="drift-detected", help="repository_dispatch event_type")
    parser.add_argument("--payload", help="JSON object to send instead of a verdict-derived payload")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name (default: $GITHUB_REPOSITORY)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent and exit")
    args = parser.parse_args(argv)

    if args.payload:
        payload = json.loads(args.payload)
    elif args.verdict:
        verdict = json.loads(args.verdict.read_text())
        if not verdict.get("drift"):
            log.info("verdict says no drift (%s); nothing to dispatch", verdict.get("reason", "no reason given"))
            return
        payload = payload_from_verdict(verdict)
    else:
        raise SystemExit("pass --verdict or --payload")

    if args.dry_run:
        print(json.dumps({"event_type": args.event, "repo": args.repo, "client_payload": payload}, indent=2))
        return
    if not args.repo:
        raise SystemExit("--repo or GITHUB_REPOSITORY is required")
    token = os.environ.get("RETRAIN_DISPATCH_TOKEN", "").strip()
    if not token:
        raise SystemExit("RETRAIN_DISPATCH_TOKEN is not set (fine-grained PAT, Contents: read and write, this repo only)")
    dispatch(args.repo, token, args.event, payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
