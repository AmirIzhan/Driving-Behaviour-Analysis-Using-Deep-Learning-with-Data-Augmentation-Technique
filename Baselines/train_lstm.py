
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
from tensorflow.keras import callbacks, layers, models, optimizers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent          
DATA_DIR  = BASE_DIR.parent / "Data"       
TRAIN_CSV = DATA_DIR / "Train.csv"
VAL_CSV   = DATA_DIR / "Validation.csv"
OUT_DIR   = BASE_DIR / "outputs_lstm"  
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL  = "label"                       
DROP_COLS  = ["behaviour", "trip_id"]       
SEED       = 42
EPOCHS     = 200    
BATCH_SIZE = 512
LR         = 3e-4

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
   
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def split_features_labels(df: pd.DataFrame) -> tuple:
    
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Missing '{LABEL_COL}' column. Found: {list(df.columns)}"
        )

    y = df[LABEL_COL].astype(int).values

    cols_to_drop = [LABEL_COL] + [c for c in DROP_COLS if c in df.columns]
    X_df = df.drop(columns=cols_to_drop, errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])

    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after removing non-feature columns.")

    return X_df.values.astype(np.float32), y, X_df.columns.tolist()


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_lstm_model(n_features: int, n_classes: int) -> tf.keras.Model:
    
    inp = layers.Input(shape=(1, n_features))

    # Layer 1
    x = layers.LSTM(128, return_sequences=True,  activation="tanh")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    # Layer 2
    x = layers.LSTM(64,  return_sequences=True,  activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    # Layer 3
    x = layers.LSTM(32,  return_sequences=False, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Classification head
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(history, save_path: str) -> None:
    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy",     []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_scaler(scaler: MinMaxScaler, save_path: str) -> None:
    
    np.savez(
        save_path,
        min_        = scaler.min_,
        scale_      = scaler.scale_,
        data_min_   = scaler.data_min_,
        data_max_   = scaler.data_max_,
        data_range_ = scaler.data_range_,
        feature_range = np.array(scaler.feature_range, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, model training, evaluation, and saving."""
    print("[INFO] Loading data...")
    train_df = load_dataframe(TRAIN_CSV)
    val_df   = load_dataframe(VAL_CSV)

    X_train_raw, y_train, feat_cols     = split_features_labels(train_df)
    X_val_raw,   y_val,   feat_cols_val = split_features_labels(val_df)

    # Align to common feature columns when train/val differ
    if feat_cols != feat_cols_val:
        print("[WARN] Train/Val feature columns differ. Using intersection.")
        common = [c for c in feat_cols if c in feat_cols_val]
        if not common:
            raise ValueError("No common numeric features between train and val.")

        train_df2    = train_df[common + [LABEL_COL]]
        val_df2      = val_df[common + [LABEL_COL]]
        X_train_raw, y_train, feat_cols = split_features_labels(train_df2)
        X_val_raw,   y_val,   _         = split_features_labels(val_df2)

    n_classes = len(np.unique(y_train))
    print(f"[INFO] Train : X={X_train_raw.shape}, y={y_train.shape}, classes={np.unique(y_train)}")
    print(f"[INFO] Val   : X={X_val_raw.shape},   y={y_val.shape}")

    # --- Scale features (fit on train only to prevent leakage) ---
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    # LSTM input shape: (n_samples, 1, n_features) — single timestep per row
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_val   = X_val.reshape(-1,   1, X_val.shape[1])

    # --- Compute class weights to mitigate class imbalance ---
    classes      = np.unique(y_train)
    weights      = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[INFO] Class weights:", class_weight)

    # --- Build model ---
    model = build_lstm_model(n_features=X_train.shape[2], n_classes=n_classes)
    model.summary()

    # --- Callbacks ---
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "training_log.csv")),
    ]

    # --- Train ---
    print("[INFO] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )

    # --- Evaluate ---
    print("[INFO] Evaluating...")
    y_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc    = accuracy_score(y_val, y_pred)
    print(f"[FINAL] Validation Accuracy: {acc:.4f}")

    # Resolve readable class names from the behaviour column if available
    if "behaviour" in val_df.columns:
        label_to_name = (
            val_df[[LABEL_COL, "behaviour"]]
            .groupby(LABEL_COL)["behaviour"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        class_names = [str(label_to_name.get(i, i)) for i in sorted(np.unique(y_train))]
    else:
        class_names = [str(i) for i in sorted(np.unique(y_train))]
    print("[INFO] Class names:", class_names)

    cm = confusion_matrix(y_val, y_pred, labels=sorted(np.unique(y_train)))
    print("[FINAL] Confusion matrix:\n", cm)
    print("\n[FINAL] Classification report:\n",
          classification_report(y_val, y_pred, target_names=class_names, digits=4))

    # --- Save outputs ---
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"[SAVE] Confusion matrix : {cm_path}")

    final_model_path = os.path.join(OUT_DIR, "final_model.keras")
    model.save(final_model_path)
    print(f"[SAVE] Model            : {final_model_path}")

    scaler_path = os.path.join(OUT_DIR, "scaler_minmax.npz")
    save_scaler(scaler, scaler_path)
    print(f"[SAVE] Scaler           : {scaler_path}")

    feats_path = os.path.join(OUT_DIR, "feature_columns.json")
    with open(feats_path, "w", encoding="utf-8") as fh:
        json.dump(feat_cols, fh, indent=2)
    print(f"[SAVE] Feature columns  : {feats_path}")

    curves_path = os.path.join(OUT_DIR, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"[SAVE] Training curves  : {curves_path}")


if __name__ == "__main__":
    main()