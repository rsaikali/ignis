"""Native (Metal) training driver: aligned truth -> .keras + report.

Runs on macOS with the ``training`` extra (tensorflow-metal). Imports numpy /
the engine model lazily so the rest of ``training`` stays importable in the
dev venv. Bypasses the engine's signature-based ``train()`` and feeds the
multi-output model real concurrent targets via ``build_model`` + ``fit``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from ignis.nilm.config import settings

from .dataset import AlignedDataset
from .report import TrainReport, attach_fit_metrics, build_report
from .source import load_aligned
from .windows import WindowedData, activation_stats, balance_windows, make_windows


def _model_name(model_type: str, end: datetime) -> str:
    return f"ignis_{model_type}_{end.strftime('%Y%m%dT%H%M%S')}"


def train(
    start: datetime,
    end: datetime,
    model_type: str | None = None,
    epochs: int | None = None,
    stride: int = 10,
    val_split: float = 0.15,
    balance: bool = True,
    dead_ratio: float = 1.0,
) -> TrainReport:
    """Train a multi-output Seq2Point on self-supervised truth and export it.

    Returns the train report (also written next to the .keras artifact).
    """
    import numpy as np

    from ignis.nilm.nilm.models import Seq2PointMultiOutputModel

    model_type = model_type or settings.nilm_model_type
    epochs = epochs or settings.nilm_epochs
    seq_len = settings.effective_sequence_length
    threshold = settings.nilm_min_power_threshold

    dataset: AlignedDataset = load_aligned(start, end, step=settings.ingest_grid_seconds)
    if len(dataset) < seq_len:
        raise SystemExit(f"Not enough aligned data: {len(dataset)} ticks < sequence_length {seq_len}")

    apps = list(dataset.appliances)
    windows: WindowedData = make_windows(dataset.aggregate, dataset.appliances, seq_len, stride=stride)
    if len(windows) == 0:
        raise SystemExit("No windows produced (series shorter than sequence_length).")

    # Counter the ~2-3% activation imbalance: keep all active windows + a
    # matched number of dead ones, so the model can't win by predicting zero.
    raw_count = len(windows)
    if balance:
        windows = balance_windows(windows, threshold, dead_ratio=dead_ratio)
        logger.info(
            "Balanced windows: {} -> {} (dead_ratio={}); activation now {}",
            raw_count,
            len(windows),
            dead_ratio,
            {a: round(f, 3) for a, f in activation_stats(windows, threshold).items()},
        )
    if len(windows) == 0:
        raise SystemExit("No windows left after balancing (no activity in window).")

    model_name = _model_name(model_type, end)
    report = build_report(model_name, model_type, start, end, dataset, seq_len, len(windows), threshold)
    logger.info("Train report (pre-fit): {}", json.dumps(report.to_dict(), indent=2))

    # --- numpy / TF boundary -------------------------------------------------
    x_raw = np.asarray(windows.x, dtype=np.float32)  # (N, seq_len)
    appliance_ids = list(range(len(apps)))
    model = Seq2PointMultiOutputModel(
        appliance_ids=appliance_ids,
        appliance_names=apps,
        sequence_length=seq_len,
        model_type=model_type,
    )

    # Normalise via the model's own preprocessor so predict() is consistent.
    # input_scaler (StandardScaler) on aggregate; target_scaler (MinMax) on the
    # pooled per-appliance targets -- matches the engine's predict()/inverse.
    pre = model.preprocessor
    targets_raw = {app: np.asarray(windows.targets[app], dtype=np.float32) for app in apps}
    all_targets = np.concatenate([targets_raw[app] for app in apps]) if apps else np.zeros(1, dtype=np.float32)
    pre.input_scaler.fit(x_raw.reshape(-1, 1))
    pre.target_scaler.fit(all_targets.reshape(-1, 1))
    pre.fitted = True

    x_scaled, _ = pre.transform(x_raw)
    x = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
    # Outputs ordered as build_model names them: output_appliance_{id}.
    y = [pre.target_scaler.transform(targets_raw[app].reshape(-1, 1)).flatten() for app in apps]

    keras_model = model.build_model()
    # The engine's asymmetric_loss penalises FALSE POSITIVES, which with 97% OFF
    # collapses predictions to ~0. Use under-prediction penalty instead.
    _recompile_under_penalty(keras_model, settings.nilm_learning_rate)

    # Per-appliance sample weights: window balancing can't fix per-appliance
    # imbalance (the high-duty TV dominates which windows survive, so the rare
    # four/pc stay ~2% ON even after balancing). Weight each appliance's ON and
    # OFF samples to equal mass, so a 2%-duty appliance still drives its output.
    sample_weight = [_balanced_sample_weight(targets_raw[app], threshold) for app in apps]

    history = keras_model.fit(
        x,
        y,
        sample_weight=sample_weight,
        validation_split=val_split,
        epochs=epochs,
        batch_size=settings.nilm_batch_size,
        verbose=2,
    )

    metrics = _per_appliance_mae(history, appliance_ids, apps)
    report = attach_fit_metrics(report, len(history.history["loss"]), metrics)

    # Recompile with a stock loss before saving: the custom loss is only needed
    # for training, and keeping it in the .keras forces every loader to register
    # it (eval crashed with "Could not locate function 'under_penalty_loss'").
    # Weights are already trained; this only swaps the (unused-at-inference) loss.
    keras_model.compile(optimizer="adam", loss="mae")

    model.model = keras_model
    out = Path(settings.nilm_model_path) / f"{model_name}.keras"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out), metadata=report.to_dict())
    _save_scalers(out, pre)
    _write_report(out, report)
    logger.info("Saved model {} and report", out)
    return report


def _balanced_sample_weight(target_raw, threshold: float):
    """Per-sample weights so an appliance's ON and OFF masses are equal.

    With p = fraction ON, weight ON samples by 0.5/p and OFF by 0.5/(1-p). A
    2%-duty appliance gets its rare ON samples up-weighted ~25x, so the model
    can't ignore it. Degenerate (all ON / all OFF) -> uniform weights.
    """
    import numpy as np

    on = target_raw > threshold
    p = float(on.mean())
    if p <= 0.0 or p >= 1.0:
        return np.ones_like(target_raw, dtype=np.float32)
    w = np.where(on, 0.5 / p, 0.5 / (1.0 - p)).astype(np.float32)
    return w


def _recompile_under_penalty(keras_model, learning_rate: float, fn_penalty: float = 3.0) -> None:
    """Recompile every output with an under-prediction-penalising MAE.

    Penalises misses (pred < truth) ``fn_penalty`` x more than over-prediction,
    the opposite of the engine's false-positive penalty. Counters the sparse-OFF
    bias that collapses predictions to zero.
    """
    import tensorflow as tf
    from tensorflow import keras

    def under_penalty_loss(y_true, y_pred):
        err = y_true - y_pred
        under = tf.cast(err > 0.0, tf.float32)  # truth above pred = a miss
        weight = 1.0 + (fn_penalty - 1.0) * under
        return tf.reduce_mean(weight * tf.abs(err))

    # The loss callable applies to every output; metrics must be per-output.
    # Use a plain list aligned to the model's outputs (Keras matches by order).
    n_out = len(keras_model.outputs)
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=under_penalty_loss,
        metrics=[["mae"] for _ in range(n_out)],
    )


def _save_scalers(model_path: Path, preprocessor) -> None:
    """Persist fitted scalers (engine save() does not) so predict() works on load."""
    import pickle

    sidecar = model_path.with_suffix(".scalers.pkl")
    with sidecar.open("wb") as f:
        pickle.dump(
            {"input_scaler": preprocessor.input_scaler, "target_scaler": preprocessor.target_scaler},
            f,
        )


def _per_appliance_mae(history, appliance_ids, apps) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for app_id, app in zip(appliance_ids, apps, strict=True):
        key = f"output_appliance_{app_id}_mae"
        m: dict[str, float] = {}
        if key in history.history:
            m["train_mae"] = float(history.history[key][-1])
        if f"val_{key}" in history.history:
            m["val_mae"] = float(history.history[f"val_{key}"][-1])
        out[app] = m
    return out


def _write_report(model_path: Path, report: TrainReport) -> None:
    report_path = model_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
