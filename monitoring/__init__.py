"""DriftWatch monitoring: read the prediction log, compare it with the training reference using
Evidently, replay the quarantined regime as production traffic, and fire the retrain trigger.

Run as modules from the repo root (``python -m monitoring.drift ...``).
"""
