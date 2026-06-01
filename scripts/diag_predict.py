"""Diagnose why predictions are ~0: inspect raw model output vs truth.

Run native:
    .venv/bin/python scripts/diag_predict.py models/champion.keras --days 3
"""

from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ignis.nilm.config import settings
from ignis.nilm.nilm.models import Seq2PointMultiOutputModel
from ignis.training.source import load_aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--days", type=float, default=3)
    args = ap.parse_args()

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    ds = load_aligned(start, end, step=settings.ingest_grid_seconds)
    agg = np.asarray(ds.aggregate, dtype=np.float32)

    model = Seq2PointMultiOutputModel(
        appliance_ids=[], appliance_names=[], sequence_length=settings.effective_sequence_length
    )
    model.load(str(args.model))
    sc = pickle.load(args.model.with_suffix(".scalers.pkl").open("rb"))
    model.preprocessor.input_scaler = sc["input_scaler"]
    model.preprocessor.target_scaler = sc["target_scaler"]
    model.preprocessor.fitted = True

    print(f"aggregate: n={len(agg)} min={agg.min():.0f} max={agg.max():.0f} mean={agg.mean():.0f}")
    print(f"target_scaler: data_min={sc['target_scaler'].data_min_} data_max={sc['target_scaler'].data_max_}")
    print(f"input_scaler: mean={sc['input_scaler'].mean_} scale={sc['input_scaler'].scale_}")

    preds = model.predict(agg, stride=1)
    print("\nappliance        pred[min/max/mean]      truth[min/max/mean]   truth_active%")
    for idx, app in enumerate(model.appliance_names):
        p = np.asarray(preds[model.appliance_ids[idx]], dtype=np.float32)
        t = np.asarray(ds.appliances.get(app, []), dtype=np.float32)
        ta = (t > settings.nilm_min_power_threshold).mean() * 100 if len(t) else 0
        print(
            f"{app:16s} {p.min():6.1f}/{p.max():7.1f}/{p.mean():6.2f}   "
            f"{t.min():6.1f}/{t.max():7.1f}/{t.mean():6.2f}   {ta:.1f}%"
        )


if __name__ == "__main__":
    main()
