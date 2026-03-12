

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.optimizers import AdamW

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent                   
DATA_DIR  = BASE_DIR.parent / "Data"                 
TRAIN_CSV = DATA_DIR / "Train.csv"
VAL_CSV   = DATA_DIR / "Validation.csv"
OUT_DIR   = BASE_DIR / "outputs_cnn_bigru"

OUT_DIR.mkdir(parents=True, exist_ok=True)


SEED         = 42
SEQ_LEN      = 45    # Sliding window length (time-steps)
STEP         = 5     # Stride between consecutive windows
BATCH_SIZE   = 256
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.35

LABEL_COL = "label"
TRIP_COL  = "trip_id"

# Column names that are not predictive features
NON_FEATURE_COLS = {
    LABEL_COL, TRIP_COL,
    "behaviour", "Behavior", "Behaviour",
    "timestamp", "time", "date",
}

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Feature inference
# ---------------------------------------------------------------------------

def infer_feature_columns(df: pd.DataFrame) -> list:
    """
    Return numeric columns that are valid model features.

    Excludes all columns listed in NON_FEATURE_COLS.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list of str
        Ordered list of numeric feature column names.

    Raises
    ------
    ValueError
        If no valid numeric feature columns exist.
    """
    feat_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feat_cols:
        raise ValueError("No numeric feature columns found. Check your CSV columns.")
    return feat_cols


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _windows_for_trip(
    features: np.ndarray,
    seq_len: int,
    step: int,
) -> np.ndarray:
    """
    Slide a window over a single trip's feature array.

    Parameters
    ----------
    features : np.ndarray of shape (n_rows, n_features)
    seq_len  : int
    step     : int

    Returns
    -------
    np.ndarray of shape (n_windows, seq_len, n_features), or None if
    the trip is shorter than seq_len.
    """
    n = len(features)
    if n < seq_len:
        return None

    windows = [features[start:start + seq_len] for start in range(0, n - seq_len + 1, step)]
    return np.stack(windows, axis=0)


def build_trip_windows(
    df: pd.DataFrame,
    feat_cols: list,
    seq_len: int = 45,
    step: int = 5,
) -> tuple:
    """
    Build sliding windows grouped by trip_id.

    Each window receives the label of the first row of its trip (assumes
    a trip is labelled homogeneously — consistent with trip-level splitting).

    Parameters
    ----------
    df        : pd.DataFrame — scaled data with LABEL_COL and TRIP_COL.
    feat_cols : list of str  — feature column names.
    seq_len   : int          — window length.
    step      : int          — stride.

    Returns
    -------
    tuple (np.ndarray, np.ndarray, np.ndarray)
        X_all     : shape (N, seq_len, n_feats), float32.
        y_all     : shape (N,), int64.
        trip_all  : shape (N,), object — trip ID per window.

    Raises
    ------
    ValueError
        If no windows were created (all trips shorter than seq_len).
    """
    X_list, y_list, trip_list = [], [], []

    for trip_id, group in df.groupby(TRIP_COL, sort=False):
        group  = group.sort_index()
        label  = group[LABEL_COL].iloc[0]
        windows = _windows_for_trip(
            group[feat_cols].to_numpy(dtype=np.float32),
            seq_len, step,
        )
        if windows is None:
            continue

        X_list.append(windows)
        y_list.append(np.full((windows.shape[0],), label, dtype=np.int64))
        trip_list.append(np.full((windows.shape[0],), trip_id, dtype=object))

    if not X_list:
        raise ValueError("No windows created. Trips may be shorter than SEQ_LEN.")

    return (
        np.concatenate(X_list,    axis=0),
        np.concatenate(y_list,    axis=0),
        np.concatenate(trip_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    out_path: str,
) -> None:
    """
    Save a labelled confusion matrix figure.

    Parameters
    ----------
    cm          : np.ndarray  — confusion matrix.
    class_names : list of str — axis tick labels.
    title       : str         — figure title.
    out_path    : str         — output PNG path.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() * 0.6 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Trip-level voting
# ---------------------------------------------------------------------------

def average_prob_vote_by_trip(
    pred_probs: np.ndarray,
    trip_ids: np.ndarray,
) -> tuple:
    """
    Predict trip-level classes by averaging window probabilities per trip.

    Averaging softmax probabilities is generally more robust than hard
    majority voting because it incorporates prediction confidence.

    Parameters
    ----------
    pred_probs : np.ndarray (N_windows, n_classes) — softmax probabilities.
    trip_ids   : np.ndarray (N_windows,)            — trip ID per window.

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        Unique trip IDs and their predicted class indices.
    """
    prob_df          = pd.DataFrame(pred_probs)
    prob_df["trip_id"] = trip_ids

    mean_probs  = prob_df.groupby("trip_id").mean()
    y_pred_trip = mean_probs.to_numpy().argmax(axis=1)
    trips       = mean_probs.index.to_numpy()

    return trips, y_pred_trip


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_cnn_bigru_model(
    seq_len: int,
    n_feats: int,
    n_classes: int,
    dropout: float = 0.35,
) -> tf.keras.Model:
    """
    Build the CNN + Bidirectional GRU classification model.

    Architecture:
        Conv1D(128) → BN → ReLU → MaxPool1D
        Conv1D(256) → BN → ReLU → MaxPool1D
        BiGRU(128, return_sequences=True)
        Dropout → BiGRU(64) → Dense(256) → Dropout → Softmax

    Parameters
    ----------
    seq_len   : int   — window length (time-steps).
    n_feats   : int   — number of sensor features per time-step.
    n_classes : int   — number of output behaviour classes.
    dropout   : float — dropout rate applied after GRU layers and dense.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    inp = layers.Input(shape=(seq_len, n_feats))

    # Local pattern extraction via Conv1D blocks
    x = layers.Conv1D(128, kernel_size=5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    # Temporal modelling via Bidirectional GRU
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)

    # Classification head
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inp, out)


# ---------------------------------------------------------------------------
# tf.data pipeline helper
# ---------------------------------------------------------------------------

def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool = False,
) -> tf.data.Dataset:
    """
    Wrap numpy arrays in a tf.data.Dataset with optional shuffling.

    Parameters
    ----------
    X         : np.ndarray — feature windows.
    y         : np.ndarray — labels.
    batch_size: int        — batch size.
    training  : bool       — if True, shuffle with re-shuffle each epoch.

    Returns
    -------
    tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 20_000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, windowing, training, evaluation, and saving."""
    print(f"[LOAD] Train: {TRAIN_CSV}")
    print(f"[LOAD] Val  : {VAL_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # Validate required columns
    for required in [LABEL_COL, TRIP_COL]:
        if required not in train_df.columns or required not in val_df.columns:
            raise ValueError(f"Missing required column '{required}' in CSV files.")

    # Ensure label column is integer-encoded
    for df in (train_df, val_df):
        if not pd.api.types.is_integer_dtype(df[LABEL_COL]):
            df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)

    feat_cols = infer_feature_columns(train_df)
    print(f"[INFO] #features = {len(feat_cols)}")
    print(f"[INFO] Example cols: {feat_cols[:10]}")

    # --- Scale features (fit on TRAIN only to prevent leakage) ---
    scaler = MinMaxScaler()
    scaler.fit(train_df[feat_cols].values)

    train_scaled      = train_df.copy()
    val_scaled        = val_df.copy()
    train_scaled[feat_cols] = scaler.transform(train_scaled[feat_cols].values)
    val_scaled[feat_cols]   = scaler.transform(val_scaled[feat_cols].values)

    # Save feature column list for inference reproducibility
    with open(os.path.join(OUT_DIR, "scaler_minmax.json"), "w") as fh:
        json.dump({"feature_cols": feat_cols}, fh, indent=2)

    # --- Build windows ---
    print("[WIN] Building train windows...")
    X_train, y_train, trip_train = build_trip_windows(train_scaled, feat_cols, SEQ_LEN, STEP)
    print("[WIN] Building val windows...")
    X_val, y_val, trip_val       = build_trip_windows(val_scaled,   feat_cols, SEQ_LEN, STEP)

    n_classes = int(max(y_train.max(), y_val.max()) + 1)
    n_feats   = X_train.shape[-1]
    print(f"[SHAPE] X_train: {X_train.shape} | y_train: {y_train.shape} | classes: {n_classes}")
    print(f"[SHAPE] X_val  : {X_val.shape}   | y_val  : {y_val.shape}")

    # --- Class weights ---
    classes      = np.unique(y_train)
    weights      = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[INFO] class_weight:", class_weight)

    # --- tf.data pipelines ---
    train_ds = make_tf_dataset(X_train, y_train, BATCH_SIZE, training=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   BATCH_SIZE, training=False)

    # --- Build and compile model ---
    model = build_cnn_bigru_model(SEQ_LEN, n_feats, n_classes, DROPOUT)
    model.compile(
        optimizer=AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n================= MODEL SUMMARY =================")
    model.summary()
    print("=================================================\n")

    # --- Callbacks ---
    ckpt_path = os.path.join(OUT_DIR, "best_model.keras")
    cbs = [
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=8, restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "training_log.csv")),
    ]

    # --- Train ---
    print("[TRAIN] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )
    model.save(os.path.join(OUT_DIR, "final_model.keras"))

    # --- Window-level evaluation ---
    print("\n[EVAL] Window-level evaluation...")
    val_probs  = model.predict(val_ds, verbose=0)
    val_pred   = val_probs.argmax(axis=1)
    window_acc = accuracy_score(y_val, val_pred)
    print(f"[RESULT] Window-level Val Accuracy: {window_acc:.4f}\n")
    print("[REPORT] Window-level Classification Report:")
    print(classification_report(y_val, val_pred, digits=4))

    cm_window     = confusion_matrix(y_val, val_pred)
    class_names   = [f"class_{i}" for i in range(n_classes)]
    cm_window_path = os.path.join(OUT_DIR, "confusion_matrix_window.png")
    plot_confusion_matrix(
        cm_window, class_names,
        title=f"Confusion Matrix (Window-level) | Acc={window_acc:.4f}",
        out_path=cm_window_path,
    )
    print(f"[SAVE] Window confusion matrix: {cm_window_path}")

    # --- Trip-level evaluation ---
    print("\n[EVAL] Trip-level evaluation (avg prob per trip)...")
    trip_ids, y_pred_trip = average_prob_vote_by_trip(val_probs, trip_val)

    # Map trips back to their true labels from the validation dataframe
    true_trip_label_map = val_df.groupby(TRIP_COL)[LABEL_COL].first()
    y_true_trip         = true_trip_label_map.loc[trip_ids].to_numpy()
    trip_acc            = accuracy_score(y_true_trip, y_pred_trip)

    print(f"[RESULT] Trip-level Val Accuracy: {trip_acc:.4f}\n")
    print("[REPORT] Trip-level Classification Report:")
    print(classification_report(y_true_trip, y_pred_trip, digits=4))

    cm_trip      = confusion_matrix(y_true_trip, y_pred_trip)
    cm_trip_path = os.path.join(OUT_DIR, "confusion_matrix_trip.png")
    plot_confusion_matrix(
        cm_trip, class_names,
        title=f"Confusion Matrix (Trip-level Voting) | Acc={trip_acc:.4f}",
        out_path=cm_trip_path,
    )
    print(f"[SAVE] Trip confusion matrix: {cm_trip_path}")

    # --- Save run summary ---
    summary = {
        "window_val_accuracy": float(window_acc),
        "trip_val_accuracy":   float(trip_acc),
        "seq_len":             SEQ_LEN,
        "step":                STEP,
        "batch_size":          BATCH_SIZE,
        "epochs_ran":          int(len(history.history["loss"])),
        "n_features":          int(n_feats),
        "n_classes":           int(n_classes),
        "feature_cols":        feat_cols,
    }
    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[DONE] Outputs saved to:", OUT_DIR)
    for fname in [
        "best_model.keras", "final_model.keras",
        "confusion_matrix_window.png", "confusion_matrix_trip.png",
        "training_log.csv", "run_summary.json",
    ]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()