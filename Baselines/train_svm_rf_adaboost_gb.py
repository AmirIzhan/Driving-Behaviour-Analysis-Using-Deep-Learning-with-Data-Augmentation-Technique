import json
import os
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration  
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent          
DATA_DIR  = BASE_DIR.parent / "Data"       
TRAIN_CSV = DATA_DIR / "Train.csv"
VAL_CSV   = DATA_DIR / "Validation.csv"
OUT_DIR   = BASE_DIR / "outputs_svm_rf_adaboost_gb"   
CM_DIR    = OUT_DIR / "confusion_matrices"             

TARGET_COL = "label"
DROP_COLS  = ["behaviour"]
META_COLS  = ["trip_id"]

OUT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features_and_labels(path: Path) -> tuple:
  
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}\n"
            f"Expected at  : {path.resolve()}"
        )

    print(f"[LOAD] Reading: {path}", flush=True)
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")

    y = df[TARGET_COL].astype(int).values

    cols_to_drop = (
        [TARGET_COL]
        + [c for c in DROP_COLS if c in df.columns]
        + [c for c in META_COLS if c in df.columns]
    )
    X = df.drop(columns=cols_to_drop, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Drop zero-variance columns
    non_zero_var = X.var() > 1e-6
    X = X.loc[:, non_zero_var]

    print(f"[LOAD] Done. X={X.shape}, y={y.shape}", flush=True)
    return X.values, y, list(X.columns)


def align_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    train_cols: list,
    val_cols: list,
) -> tuple:
    
    common = [c for c in train_cols if c in set(val_cols)]   
    print(f"[INFO] Aligned to {len(common)} common features", flush=True)

    train_idx = [train_cols.index(f) for f in common]
    val_idx   = [val_cols.index(f)   for f in common]

    return X_train[:, train_idx], X_val[:, val_idx], common


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    name: str,
    model_num: int,
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    labels: np.ndarray,
) -> tuple:
    
    print(f"[EVAL] Predicting for {name}...", flush=True)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="weighted")
    cm  = confusion_matrix(y_val, y_pred, labels=labels)

    print("\n" + "=" * 60)
    print(f"{name} (Model {model_num})")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Weighted F1-Score   : {f1:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    # Confusion matrix figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(
        f"{name} (Model {model_num}) – Confusion Matrix",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)

    plt.text(
        0.02, 0.98,
        f"Accuracy: {acc:.4f}\nF1-Score: {f1:.4f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()

    fig_path = str(CM_DIR / f"model_{model_num}_cm.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] Confusion matrix saved: {fig_path}", flush=True)

    return acc, f1


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def define_models(n_features: int) -> dict:
    
    model_1 = Pipeline([
        ("scaler", RobustScaler()),
        ("feature_selection", SelectKBest(f_classif, k=min(50, n_features))),
        ("clf", LinearSVC(
            random_state=1,
            max_iter=20_000,
            C=0.1,
            class_weight="balanced",
            dual=False,
        )),
    ])

    model_2 = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=1,
    )

    model_3 = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=1,
        n_jobs=-1,
    )

    model_4 = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=1,
    )

    return {
        "SVM_Linear":       (1, model_1),
        "AdaBoost":         (2, model_2),
        "RandomForest":     (3, model_3),
        "GradientBoosting": (4, model_4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate loading, training, evaluation, and reporting."""
    print("=" * 60)
    print("  ENSEMBLE MODEL TRAINING")
    print("=" * 60)
    print(f"  TRAIN_CSV : {TRAIN_CSV}")
    print(f"  VAL_CSV   : {VAL_CSV}")
    print(f"  OUT_DIR   : {OUT_DIR}")
    print(f"  CM_DIR    : {CM_DIR}", flush=True)

    # --- Load data ---
    t0 = time.time()
    X_train, y_train, train_cols = load_features_and_labels(TRAIN_CSV)
    X_val,   y_val,   val_cols   = load_features_and_labels(VAL_CSV)
    print(f"[INFO] Data loaded in {time.time() - t0:.1f}s", flush=True)

    # Align feature sets if they differ
    if set(train_cols) != set(val_cols):
        print("[WARNING] Feature mismatch — aligning features...", flush=True)
        X_train, X_val, train_cols = align_features(
            X_train, X_val, train_cols, val_cols
        )

    labels = np.unique(np.concatenate([y_train, y_val]))
    print(f"[INFO] Classes found: {labels}", flush=True)

    class_weights_arr = compute_class_weight("balanced", classes=labels, y=y_train)
    class_weight_dict = dict(zip(labels.tolist(), class_weights_arr.tolist()))
    print(f"[INFO] Class weights: {class_weight_dict}", flush=True)

    # --- Train & evaluate each model ---
    clf_models = define_models(n_features=X_train.shape[1])
    results    = {}

    for name, (model_num, model) in clf_models.items():
        print(f"\n{'='*60}\n[TRAIN] Model {model_num}: {name}\n{'='*60}", flush=True)

        t1 = time.time()
        model.fit(X_train, y_train)
        print(f"[TRAIN] Fit completed in {time.time() - t1:.1f}s", flush=True)

        acc, f1 = evaluate_and_plot(
            name, model_num, model, X_val, y_val, labels=labels
        )
        results[name] = {
            "model_number": model_num,
            "accuracy":     float(acc),
            "f1_score":     float(f1),
        }

        # Save fitted model with metadata
        save_path = str(OUT_DIR / f"model_{model_num}_{name}.joblib")
        joblib.dump(
            {
                "model":         model,
                "feature_names": train_cols,
                "target_col":    TARGET_COL,
                "drop_cols":     DROP_COLS,
                "meta_cols":     META_COLS,
                "model_number":  model_num,
                "accuracy":      acc,
                "f1_score":      f1,
            },
            save_path,
        )
        print(f"[SAVE] Model saved: {save_path}", flush=True)

    # --- Summary ---
    best_name = max(results, key=lambda x: results[x]["f1_score"])
    best      = results[best_name]

    summary = {
        "results":           results,
        "best_model":        best_name,
        "best_model_number": best["model_number"],
        "best_accuracy":     best["accuracy"],
        "best_f1_score":     best["f1_score"],
    }

    summary_path = str(OUT_DIR / "results_summary.json")   
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED ")
    print("=" * 60)
    print("\nResults Summary:")
    for name, metrics in results.items():
        print(f"  Model {metrics['model_number']} ({name}):")
        print(f"    Accuracy : {metrics['accuracy']:.4f}")
        print(f"    F1-Score : {metrics['f1_score']:.4f}")

    print(f"\nBest Model : Model {best['model_number']} ({best_name})")
    print(f"  Accuracy : {best['accuracy']:.4f}")
    print(f"  F1-Score : {best['f1_score']:.4f}")
    print(f"\nModels saved to       : {OUT_DIR}")
    print(f"Confusion matrices to : {CM_DIR}")
    print(f"Summary saved to      : {summary_path}", flush=True)


if __name__ == "__main__":
    main()