import time
from collections import Counter
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, regularizers
from tensorflow.keras.optimizers import AdamW

# ===========================================================================
# Configuration
# ===========================================================================

TRAIN_CSV  = "data/combined_synthetic_train_data.csv"
VAL_CSV    = "data/Validation.csv"

LABEL_COL  = "label"
TRIP_COL   = "trip_id"
DROP_COLS  = ["behaviour"]

SEQ_LEN    = 45     
STEP       = 15     
THRESHOLD  = 0.80   

N_SPLITS   = 5
EPOCHS     = 50     
BATCH_SIZE = 256
LR         = 3e-4
SEED       = 42

CLASS_NAMES = ["Aggressive", "Drowsy", "Normal"]

tf.keras.utils.set_random_seed(SEED)


# ===========================================================================
# Window statistics  (computed on RAW data — scaled separately per fold)
# ===========================================================================

def compute_window_statistics(window: np.ndarray) -> np.ndarray:
    mean_  = window.mean(axis=0)
    std_   = window.std(axis=0)
    min_   = window.min(axis=0)
    max_   = window.max(axis=0)
    dx     = np.diff(window, axis=0)
    j_mean = dx.mean(axis=0)
    j_std  = dx.std(axis=0)
    return np.concatenate([mean_, std_, min_, max_, j_mean, j_std]).astype(np.float32)


# ===========================================================================
# Windowing
# ===========================================================================

def build_windows(df: pd.DataFrame, seq_len: int, step: int, threshold: float):
    if LABEL_COL not in df.columns or TRIP_COL not in df.columns:
        raise ValueError(f"Dataset must contain '{LABEL_COL}' and '{TRIP_COL}'.")

    exclude = {LABEL_COL, TRIP_COL} | {c for c in DROP_COLS if c in df.columns}
    feat_df   = df.drop(columns=list(exclude), errors="ignore")
    feat_df   = feat_df.select_dtypes(include=[np.number])
    feat_cols = feat_df.columns.tolist()

    seq_list, stat_list, label_list, trip_list = [], [], [], []
    kept = dropped = 0

    for trip_id, grp in df.groupby(TRIP_COL, sort=False):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < seq_len:
            continue

        labels_arr   = grp[LABEL_COL].to_numpy(dtype=int)
        features_arr = grp[feat_cols].to_numpy(dtype=np.float32)

        for start in range(0, n - seq_len + 1, step):
            end      = start + seq_len
            win_lbl  = labels_arr[start:end]
            vals, cnts = np.unique(win_lbl, return_counts=True)
            maj_label  = int(vals[np.argmax(cnts)])
            ratio      = float(np.max(cnts)) / seq_len

            if ratio >= threshold:
                win_feat = features_arr[start:end]
                seq_list.append(win_feat)
                stat_list.append(compute_window_statistics(win_feat))
                label_list.append(maj_label)
                trip_list.append(trip_id)
                kept += 1
            else:
                dropped += 1

    X_seq  = np.array(seq_list,  dtype=np.float32)
    X_stat = np.array(stat_list, dtype=np.float32)
    y      = np.array(label_list, dtype=int)
    trips  = np.array(trip_list)

    print(f"\nWindowing: kept={kept:,}, dropped={dropped:,}")
    print(f"X_seq={X_seq.shape}, X_stat={X_stat.shape}, y={y.shape}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    return X_seq, X_stat, y, trips, feat_cols


# ===========================================================================
# Model  — reduced dropout, proven loss
# ===========================================================================

def build_model(timesteps: int, n_features: int, n_stats: int, n_classes: int) -> tf.keras.Model:
    # --- Sequence branch ---
    seq_in = layers.Input(shape=(timesteps, n_features), name="seq_in")

    x = layers.LayerNormalization()(seq_in)
    x = layers.SpatialDropout1D(0.10)(x)                               

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)                                        

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,  dropout=0.15))(x)  
    x = layers.Dropout(0.15)(x)                                        
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.10))(x)  
    x = layers.Dropout(0.10)(x)                                        

    # --- Statistics branch ---
    stat_in = layers.Input(shape=(n_stats,), name="stat_in")
    s = layers.LayerNormalization()(stat_in)
    s = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(s)
    s = layers.Dropout(0.10)(s)                                        

    # --- Merge + head ---
    h = layers.Concatenate()([x, s])
    h = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(h)
    h = layers.Dropout(0.15)(h)                                       
    h = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(5e-5))(h)
    h = layers.Dropout(0.10)(h)                                        

    out = layers.Dense(n_classes, activation="softmax")(h)

    model = models.Model(inputs=[seq_in, stat_in], outputs=out)
    model.compile(
        optimizer=AdamW(learning_rate=LR, weight_decay=1e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",  
        metrics=["accuracy"],
    )
    return model


# ===========================================================================
# Callbacks
# ===========================================================================

def get_callbacks(fold: int) -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,                  
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,                 
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=f"best_model_fold_{fold}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]


# ===========================================================================
# Plotting
# ===========================================================================

def plot_fold(history, fold: int) -> None:
    for metric, ylabel, fname in [
        ("accuracy", "Accuracy", f"accuracy_fold_{fold}.png"),
        ("loss",     "Loss",     f"loss_fold_{fold}.png"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.plot(history.history[metric],           color="steelblue", label=f"Train {ylabel}",      linewidth=2)
        plt.plot(history.history[f"val_{metric}"],  color="tomato",    label=f"Validation {ylabel}",  linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel,  fontsize=12)
        plt.title(f"Model {ylabel} — Fold {fold}", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved: {fname}")


def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right", fontsize=11)
    plt.yticks(ticks, CLASS_NAMES, fontsize=11)
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.ylabel("True",      fontsize=12, fontweight="bold")
    thresh = cm.max() * 0.5 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", fontsize=12,
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ===========================================================================
# Trip-level probability-sum voting
# ===========================================================================

def trip_vote(probs: np.ndarray, trip_ids: np.ndarray,
              true_labels: np.ndarray, n_classes: int):
    df = pd.DataFrame({"trip_id": trip_ids, "true": true_labels})
    for c in range(n_classes):
        df[f"p{c}"] = probs[:, c]

    def _vote(g):
        sums = [g[f"p{c}"].sum() for c in range(n_classes)]
        return pd.Series({"true": int(g["true"].mode()[0]),
                          "pred": int(np.argmax(sums))})

    res = df.groupby("trip_id").apply(_vote)
    return res["true"].to_numpy(), res["pred"].to_numpy()


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("  BiLSTM Driving Behaviour — Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)

    print(f"\nTrain CSV : {len(df_train):,} rows")
    print(f"Val CSV   : {len(df_val):,}   rows  (held out — NOT used in CV folds)")

    # ── Build windows ────────────────────────────────────────────────────────
    print("\n[1/4] Building windows from TRAIN data only...")
    X_seq, X_stat, y_raw, trips, feat_cols = build_windows(
        df_train, SEQ_LEN, STEP, THRESHOLD
    )

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    timesteps  = X_seq.shape[1]
    n_features = X_seq.shape[2]
    n_stats    = X_stat.shape[1]
    n_classes  = len(le.classes_)

    print(f"\nDataset summary")
    print(f"  Windows    : {len(y):,}")
    print(f"  Trips      : {len(np.unique(trips)):,}")
    print(f"  Classes    : {n_classes}  →  {dict(Counter(y.tolist()))}")
    print(f"  Features   : {n_features}")
    print(f"  Stat vector: {n_stats}")

    # ── GroupKFold cross-validation ──────────────────────────────────────────
    print(f"\n[2/4] GroupKFold CV  ({N_SPLITS} folds, grouped by trip_id)")
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_win_acc, fold_trip_acc       = [], []
    all_win_true,  all_win_pred       = [], []
    all_trip_true, all_trip_pred      = [], []
    fold_times                        = []
    overall_start                     = time.time()

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_seq, y, groups=trips), start=1
    ):
        fold_start = time.time()
        print(f"\n{'─'*55}\nFold {fold}/{N_SPLITS}\n{'─'*55}")

        Xs_tr, Xs_va = X_seq[tr_idx],  X_seq[va_idx]
        Xt_tr, Xt_va = X_stat[tr_idx], X_stat[va_idx]
        y_tr,  y_va  = y[tr_idx],      y[va_idx]
        va_trips      = trips[va_idx]

        # Sanity-check: zero trip overlap
        assert len(set(trips[tr_idx]) & set(trips[va_idx])) == 0, "Data leakage!"
        print(f"  Train trips: {len(set(trips[tr_idx])):,} | Val trips: {len(set(trips[va_idx])):,}")
        print(f"  Train dist : {Counter(y_tr.tolist())}")
        print(f"  Val dist   : {Counter(y_va.tolist())}")

        # ── Scale (fit on train ONLY) ────────────────────────────────────────
        seq_scaler  = MinMaxScaler()
        Xs_tr = seq_scaler.fit_transform(
            Xs_tr.reshape(-1, n_features)
        ).reshape(-1, timesteps, n_features).astype(np.float32)
        Xs_va = seq_scaler.transform(
            Xs_va.reshape(-1, n_features)
        ).reshape(-1, timesteps, n_features).astype(np.float32)

        stat_scaler = MinMaxScaler()
        Xt_tr = stat_scaler.fit_transform(Xt_tr).astype(np.float32)
        Xt_va = stat_scaler.transform(Xt_va).astype(np.float32)

        cw      = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw_dict = dict(enumerate(cw))
        print(f"  Class weights: { {CLASS_NAMES[k]: round(v, 3) for k, v in cw_dict.items()} }")

        # ── Build & train ────────────────────────────────────────────────────
        model = build_model(timesteps, n_features, n_stats, n_classes)
        if fold == 1:
            model.summary()

        history = model.fit(
            {"seq_in": Xs_tr, "stat_in": Xt_tr},
            y_tr,
            validation_data=({"seq_in": Xs_va, "stat_in": Xt_va}, y_va),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=cw_dict,  
            callbacks=get_callbacks(fold),
            verbose=2,
        )

        # ── Plots ────────────────────────────────────────────────────────────
        plot_fold(history, fold)

        # ── Window-level eval ────────────────────────────────────────────────
        probs    = model.predict({"seq_in": Xs_va, "stat_in": Xt_va}, verbose=0)
        win_pred = probs.argmax(axis=1)
        win_acc  = accuracy_score(y_va, win_pred)

        fold_win_acc.append(win_acc)
        all_win_true.append(y_va)
        all_win_pred.append(win_pred)
        print(f"\n   Window accuracy  : {win_acc:.4f}")

        # ── Trip-level voting ────────────────────────────────────────────────
        t_true, t_pred = trip_vote(probs, va_trips, y_va, n_classes)
        trip_acc       = (t_true == t_pred).mean()

        fold_trip_acc.append(trip_acc)
        all_trip_true.append(t_true)
        all_trip_pred.append(t_pred)
        print(f"   Trip accuracy    : {trip_acc:.4f}")

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        print(f" Fold time: {timedelta(seconds=int(fold_time))}")
        if fold < N_SPLITS:
            eta = np.mean(fold_times) * (N_SPLITS - fold)
            print(f" ETA remaining: {timedelta(seconds=int(eta))}")

    # ── Final hold-out evaluation on VAL CSV ─────────────────────────────────
    print(f"\n[3/4] Hold-out evaluation on {VAL_CSV}...")

    # Build windows from val CSV using same parameters
    X_seq_v, X_stat_v, y_raw_v, trips_v, _ = build_windows(
        df_val, SEQ_LEN, STEP, THRESHOLD
    )
    y_v = le.transform(y_raw_v)   

    # Scale using a scaler fitted on ALL training windows
    final_seq_scaler  = MinMaxScaler()
    final_stat_scaler = MinMaxScaler()

    X_seq_tr_2d = X_seq.reshape(-1, n_features)
    Xs_v = final_seq_scaler.fit_transform(X_seq_tr_2d)
    Xs_v = final_seq_scaler.transform(
        X_seq_v.reshape(-1, n_features)
    ).reshape(-1, timesteps, n_features).astype(np.float32)

    final_stat_scaler.fit(X_stat)
    Xt_v = final_stat_scaler.transform(X_stat_v).astype(np.float32)

    # Use last fold's model for hold-out (or reload best checkpoint)
    probs_v    = model.predict({"seq_in": Xs_v, "stat_in": Xt_v}, verbose=0)
    win_pred_v = probs_v.argmax(axis=1)
    heldout_win_acc = accuracy_score(y_v, win_pred_v)

    t_true_v, t_pred_v = trip_vote(probs_v, trips_v, y_v, n_classes)
    heldout_trip_acc = (t_true_v == t_pred_v).mean()

    print(f"  Hold-out window accuracy : {heldout_win_acc:.4f}")
    print(f"  Hold-out trip accuracy   : {heldout_trip_acc:.4f}")

    # ── Aggregate CV results ──────────────────────────────────────────────────
    print(f"\n[4/4] Aggregating results...")

    total_time = time.time() - overall_start
    all_win_true  = np.concatenate(all_win_true)
    all_win_pred  = np.concatenate(all_win_pred)
    all_trip_true = np.concatenate(all_trip_true)
    all_trip_pred = np.concatenate(all_trip_pred)

    print(f"\n{'='*60}")
    print("  GroupKFold CV Results (Trip-level, No Data Leakage)")
    print(f"{'='*60}")
    print(f"  Total time        : {timedelta(seconds=int(total_time))}")
    print(f"  Window accs       : {[round(float(a), 4) for a in fold_win_acc]}")
    print(f"  Window mean ± std : {np.mean(fold_win_acc):.4f} ± {np.std(fold_win_acc):.4f}")
    print(f"  Trip accs         : {[round(float(a), 4) for a in fold_trip_acc]}")
    print(f"  Trip mean ± std   : {np.mean(fold_trip_acc):.4f} ± {np.std(fold_trip_acc):.4f}")

    print(f"\n  Hold-out Window acc : {heldout_win_acc:.4f}")
    print(f"  Hold-out Trip acc   : {heldout_trip_acc:.4f}")

    print("\nWINDOW-LEVEL confusion matrix (all CV folds):")
    cm_win = confusion_matrix(all_win_true, all_win_pred)
    print(cm_win)
    print(classification_report(all_win_true, all_win_pred,
                                 target_names=CLASS_NAMES, digits=3))

    print("\nTRIP-LEVEL confusion matrix (all CV folds):")
    cm_trip = confusion_matrix(all_trip_true, all_trip_pred)
    print(cm_trip)
    print(classification_report(all_trip_true, all_trip_pred,
                                  target_names=CLASS_NAMES, digits=3))

    plot_confusion_matrix(cm_win,  "Window-level Confusion Matrix (All Folds)", "cm_window.png")
    plot_confusion_matrix(cm_trip, "Trip-level Confusion Matrix (All Folds)",   "cm_trip.png")

    print("\n Done! Files saved:")
    for i in range(1, N_SPLITS + 1):
        print(f"   accuracy_fold_{i}.png  |  loss_fold_{i}.png  |  best_model_fold_{i}.keras")
    print("   cm_window.png  |  cm_trip.png")


if __name__ == "__main__":
    main()