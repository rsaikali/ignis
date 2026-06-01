"""Pre-train smoke test: data availability + window count + TF/GPU.

Does NOT fit a model. Run native (engine+training extras installed):
    .venv/bin/python scripts/smoke_train.py [--days N]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from ignis.nilm.config import settings
from ignis.training.source import load_aligned
from ignis.training.windows import make_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)

    ds = load_aligned(start, end)
    hours = len(ds) * ds.step / 3600.0
    print(f"aligned ticks: {len(ds)} | hours: {hours:.1f} | step: {ds.step}s")

    for app, c in ds.coverage(settings.nilm_min_power_threshold).items():
        print(f"  {app:16s} active={c.active_hours:7.1f}h  frac={c.active_fraction:.3f}  ticks={c.total_ticks}")

    seq_len = settings.effective_sequence_length
    w = make_windows(ds.aggregate, ds.appliances, seq_len, stride=10)
    window_hours = seq_len * ds.step / 3600.0
    print(f"windows: {len(w)} | seq_len: {w.sequence_length} (= {window_hours:.1f}h input window)")

    if len(w) == 0:
        print("\n>>> 0 windows. Need >= seq_len aligned ticks. Likely backfill required")
        print(f"    (have {len(ds)} ticks, need >= {seq_len}).")

    try:
        import tensorflow as tf

        print(f"\nTF: {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("\nTF not installed (run: make train-deps)")


if __name__ == "__main__":
    main()
