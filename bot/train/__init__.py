"""Overnight / resumable model-training pipeline.

Trains a two-stage (regime + timing) gradient-boosted model on the full
available Coinbase history for multiple pairs, with honest walk-forward
validation and an untouched holdout gate. Designed to run as a chain of
time-budgeted GitHub Actions jobs that resume from a small state/
training/checkpoint.json ledger and a data/train_cache/ (Actions cache).
"""