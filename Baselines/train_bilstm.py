import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers

# ---------------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent                   
DATA_DIR  = BASE_DIR.parent / "Data"                 
TRAIN_CSV = DATA_DIR / "combined_synthetic_train_data.csv"
VAL_CSV   = DATA_DIR / "Validation.csv"
OUT_DIR   = BASE_DIR / "outputs_bilstm"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
LABEL_COL    = "label"
TRIP_COL     = "trip_id"
SEQ_LEN      = 45    # Sliding window length (time-steps)
STEP         = 5     # Stride between consecutive windows

# Model hyperparameters
HIDDEN_UNITS = 128
DROPOUT      = 0.2
LR           = 1e-3
BATCH_SIZE   = 64
EPOCHS       = 60
PATIENCE     = 8     # ✅ Increased from 5 — more time to converge

# Output artefact paths  (str() for keras / joblib compatibility)
MODEL_PATH     = str(OUT_DIR / "bilstm_model.keras")
SCALER_PATH    = str(OUT_DIR / "scaler.joblib")
ENCODER_PATH   = str(OUT_DIR / "label_encoder.joblib")
HISTORY_PATH   = str(OUT_DIR / "history.json")
CM_WINDOW_PATH = str(OUT_DIR / "confusion_matrix_window.png")
CM_TRIP_PATH   = str(OUT_DIR / "confusion_matrix_trip.png")

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Feature inference
# ---------------------------------------------------------------------------

def infer_feature_columns(df: pd.DataFrame) -> list:
    
    excluded = {LABEL_COL, TRIP_COL}
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    seq_len: int = 45,
    step: int = 5,
    trip_col: str = None,
) -> tuple:
    
    data = df.reset_index(drop=True)
    X_list, y_list, trip_list = [], [], []
    kept = 0

    def _process(group, trip_id=None):
        nonlocal kept
        n        = len(group)
        features = group[feature_cols].to_numpy(dtype=np.float32, copy=False)
        labels   = group[label_col].to_numpy(dtype=np.int64, copy=False)

        for start in range(0, n - seq_len + 1, step):
            end     = start + seq_len
            win_lbl = labels[start:end]
            vals, cnts = np.unique(win_lbl, return_counts=True)
            maj_label  = int(vals[np.argmax(cnts)])   # majority label

            X_list.append(features[start:end])
            y_list.append(maj_label)
            trip_list.append(trip_id)
            kept += 1

    if trip_col is not None and trip_col in data.columns:
        for trip_id, group in data.groupby(trip_col, sort=False):
            group = group.reset_index(drop=True)
            if len(group) >= seq_len:
                _process(group, trip_id)
    else:
        if len(data) >= seq_len:
            _process(data)

    if not X_list:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.array([], dtype=np.int64),
            None,
        )

    print(f"  Windows created: {kept:,}")
    X     = np.stack(X_list).astype(np.float32)
    y     = np.asarray(y_list, dtype=np.int64)
    trips = np.asarray(trip_list) if trip_list[0] is not None else None
    return X, y, trips


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: dict) -> None:
    """Save accuracy and loss training curves to OUT_DIR."""
    for metric, ylabel in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        plt.figure(figsize=(7, 5))
        plt.plot(history[metric],          color="steelblue", label=f"Train {ylabel}", linewidth=2)
        plt.plot(history[f"val_{metric}"], color="tomato",    label=f"Val {ylabel}",   linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel,  fontsize=12)
        plt.title(f"Model {ylabel}", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = str(OUT_DIR / f"{metric}_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a labelled confusion matrix figure."""
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=11)
    plt.yticks(ticks, class_names, fontsize=11)
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.ylabel("True",      fontsize=12, fontweight="bold")

    threshold = cm.max() * 0.6 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > threshold else "black",
                fontweight="bold" if i == j else "normal",
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trip-level majority vote
# ---------------------------------------------------------------------------

def majority_vote_by_trip(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trip_ids: np.ndarray,
) -> tuple:
    """Compute trip-level labels via majority voting."""
    df_votes = pd.DataFrame({
        "trip_id": trip_ids,
        "y_true":  y_true,
        "y_pred":  y_pred,
    })

    trip_true, trip_pred = [], []
    for _, group in df_votes.groupby("trip_id"):
        trip_true.append(group["y_true"].value_counts().idxmax())
        trip_pred.append(group["y_pred"].value_counts().idxmax())

    return np.array(trip_true), np.array(trip_pred)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_bilstm_model(n_features: int, seq_len: int, n_classes: int) -> tf.keras.Model:
    """Build a 3-layer stacked Bidirectional LSTM model."""
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.Bidirectional(layers.LSTM(HIDDEN_UNITS, return_sequences=True)),
        layers.Dropout(DROPOUT),
        layers.Bidirectional(layers.LSTM(HIDDEN_UNITS, return_sequences=True)),
        layers.Dropout(DROPOUT),
        layers.Bidirectional(layers.LSTM(HIDDEN_UNITS, return_sequences=False)),
        layers.Dropout(DROPOUT),
        layers.Dense(128, activation="relu"),
        layers.Dropout(DROPOUT),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, windowing, training, evaluation, and saving."""

    print("[LOAD] Reading CSVs...")
    print(f"  Train : {TRAIN_CSV}")
    print(f"  Val   : {VAL_CSV}")
    print(f"  OutDir: {OUT_DIR}")

    # .exists() works because TRAIN_CSV / VAL_CSV are Path objects
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV.resolve()}\n"
            f"Place your file at:     {TRAIN_CSV.resolve()}"
        )
    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {VAL_CSV.resolve()}\n"
            f"Place your file at:       {VAL_CSV.resolve()}"
        )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    print(f"  Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    for col in [LABEL_COL]:
        if col not in train_df.columns or col not in val_df.columns:
            raise ValueError(
                f"Missing '{col}' column.\n"
                f"Train columns: {train_df.columns.tolist()}\n"
                f"Val columns  : {val_df.columns.tolist()}"
            )

    # --- Feature columns (inferred from train, enforced on val) ---
    feature_cols = infer_feature_columns(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found. Check your CSV format.")

    missing_in_val = [c for c in feature_cols if c not in val_df.columns]
    if missing_in_val:
        raise ValueError(f"Validation CSV is missing features: {missing_in_val}")

    # Keep only train features in val (drops any extra val columns silently)
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")

    # --- Clean: Inf → NaN → fill ---
    print("[CLEAN] Removing Inf / NaN values...")
    for df in (train_df, val_df):
        df[feature_cols] = (
            df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )
    nan_left = train_df[feature_cols].isnull().sum().sum()
    print(f"  NaN remaining after clean: {nan_left}  {'✅' if nan_left == 0 else '⚠️'}")

    # --- Label encoding ---
    print("[ENCODE] Fitting LabelEncoder...")
    le = LabelEncoder()
    le.fit(
        np.concatenate([
            train_df[LABEL_COL].astype(str).to_numpy(),
            val_df[LABEL_COL].astype(str).to_numpy(),
        ])
    )
    train_df[LABEL_COL] = le.transform(train_df[LABEL_COL].astype(str))
    val_df[LABEL_COL]   = le.transform(val_df[LABEL_COL].astype(str))

    class_names = le.classes_.tolist()
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train dist: {dict(pd.Series(train_df[LABEL_COL]).value_counts().sort_index())}")
    print(f"  Val dist  : {dict(pd.Series(val_df[LABEL_COL]).value_counts().sort_index())}")
    joblib.dump(le, ENCODER_PATH)

    # --- Scale features (fit on TRAIN only to prevent leakage) ---
    print("[SCALE] MinMaxScaler fit on TRAIN only...")
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(
        train_df[feature_cols].to_numpy()
    ).astype(np.float32)
    val_df[feature_cols] = scaler.transform(
        val_df[feature_cols].to_numpy()
    ).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)

    # --- Build sliding windows ---
    trip_col = TRIP_COL if (
        TRIP_COL in train_df.columns and TRIP_COL in val_df.columns
    ) else None
    print(f"[WINDOW] Building windows (seq_len={SEQ_LEN}, step={STEP})...")

    X_train, y_train, trip_train = build_windows(
        train_df, feature_cols, LABEL_COL, SEQ_LEN, STEP, trip_col
    )
    X_val, y_val, trip_val = build_windows(
        val_df, feature_cols, LABEL_COL, SEQ_LEN, STEP, trip_col
    )

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError(
            f"No windows created. SEQ_LEN ({SEQ_LEN}) may be too large.\n"
            f"Train rows={len(train_df)}, Val rows={len(val_df)}"
        )

    print(f"[INFO] X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"[INFO] RAM — Train: {X_train.nbytes/1e6:.1f} MB | Val: {X_val.nbytes/1e6:.1f} MB")

    # --- Class weights ---
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train.astype(int),
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"[INFO] Class weights: { {class_names[k]: round(v, 3) for k, v in class_weight_dict.items()} }")

    # --- Build model ---
    model = build_bilstm_model(X_train.shape[-1], SEQ_LEN, num_classes)
    print("\n==== MODEL SUMMARY ====")
    model.summary()

    # --- Callbacks ---
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Train ---
    print("\n[TRAIN] Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=1,
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history.history, fh, indent=2)

    # --- Training curves ---
    print("\n[PLOT] Saving training curves...")
    plot_history(history.history)

    # --- Window-level evaluation ---
    print("\n[EVAL] Predicting on validation windows...")
    val_probs  = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_pred   = np.argmax(val_probs, axis=1)
    window_acc = accuracy_score(y_val, val_pred)

    print(f"\n{'='*55}")
    print(f"[WINDOW-LEVEL] Accuracy : {window_acc:.4f}")
    print("[WINDOW-LEVEL] Classification report:")
    print(classification_report(y_val, val_pred, target_names=class_names, digits=4))

    cm_window = confusion_matrix(y_val, val_pred)
    print("[WINDOW-LEVEL] Confusion matrix:")
    print(cm_window)
    plot_confusion_matrix(cm_window, class_names, CM_WINDOW_PATH,
                          "Confusion Matrix (Window-Level)")
    print(f"  Saved: {CM_WINDOW_PATH}")

    # --- Trip-level evaluation ---
    if trip_val is not None:
        trip_true, trip_pred = majority_vote_by_trip(y_val, val_pred, trip_val)
        trip_acc = accuracy_score(trip_true, trip_pred)

        print(f"\n{'='*55}")
        print(f"[TRIP-LEVEL] Accuracy (majority vote): {trip_acc:.4f}")
        print("[TRIP-LEVEL] Classification report:")
        print(classification_report(trip_true, trip_pred, target_names=class_names, digits=4))

        cm_trip = confusion_matrix(trip_true, trip_pred)
        print("[TRIP-LEVEL] Confusion matrix:")
        print(cm_trip)
        plot_confusion_matrix(cm_trip, class_names, CM_TRIP_PATH,
                              "Confusion Matrix (Trip-Level)")
        print(f"  Saved: {CM_TRIP_PATH}")

    print(f"\n{'='*55}")
    print(f"[DONE] Model   : {MODEL_PATH}")
    print(f"[DONE] Scaler  : {SCALER_PATH}")
    print(f"[DONE] Encoder : {ENCODER_PATH}")
    print(f"[DONE] History : {HISTORY_PATH}")
    print(f"[DONE] Plots   : {OUT_DIR}")


if __name__ == "__main__":
    main()