import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent                   
DATA_DIR  = BASE_DIR.parent / "Data"                 
TRAIN_CSV = DATA_DIR / "combined_synthetic_train_data.csv"
VAL_CSV   = DATA_DIR / "Validation.csv"
OUT_DIR   = BASE_DIR / "outputs_knn_logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"      # Integer-encoded class column
NAME_COL  = "behaviour"  # Human-readable string label
TRIP_COL  = "trip_id"    # Trip identifier for aggregation
DROP_COLS = [LABEL_COL, NAME_COL, TRIP_COL]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    save_path: str,
) -> None:
   
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Annotate each cell with its count
    threshold = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Trip-level evaluation
# ---------------------------------------------------------------------------

def trip_majority_vote_accuracy(
    df_val: pd.DataFrame,
    y_pred: np.ndarray,
    label_col: str,
    trip_col: str,
) -> float:
    
    trip_df          = df_val[[trip_col, label_col]].copy()
    trip_df["pred"]  = y_pred

    def _majority(series: pd.Series):
        values, counts = np.unique(series, return_counts=True)
        return values[np.argmax(counts)]

    trip_true = trip_df.groupby(trip_col)[label_col].apply(_majority)
    trip_pred = trip_df.groupby(trip_col)["pred"].apply(_majority)

    return accuracy_score(trip_true.values, trip_pred.values)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def define_models() -> dict:
   
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.5,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",     # Closer neighbours have higher influence
            n_jobs=-1,
        ),
        "NaiveBayes": GaussianNB(),
        "LogReg": LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate loading, training, evaluation, and reporting."""
    # --- Load data ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # Validate required columns
    for required_col in [LABEL_COL, NAME_COL, TRIP_COL]:
        if required_col not in train_df.columns or required_col not in val_df.columns:
            raise ValueError(f"Missing required column '{required_col}' in train/val CSV.")

    # Build feature matrices by dropping label and metadata columns
    X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    y_train = train_df[LABEL_COL].astype(int)
    X_val   = val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns])
    y_val   = val_df[LABEL_COL].astype(int)

    # Derive readable class names from the behaviour string column
    if NAME_COL in val_df.columns:
        class_names = (
            val_df.groupby(LABEL_COL)[NAME_COL]
            .agg(lambda x: x.value_counts().index[0])
            .sort_index()
            .tolist()
        )
    else:
        class_names = [str(i) for i in sorted(y_val.unique())]

    # --- Train and evaluate each model ---
    results = []
    classifiers = define_models()

    for name, clf in classifiers.items():
        print("\n" + "=" * 60)
        print(f"[TRAIN] {name}")

        # Wrap every classifier in a scaling pipeline for fair comparison
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  clf),
        ])
        pipe.fit(X_train, y_train)

        y_pred    = pipe.predict(X_val)
        window_acc = accuracy_score(y_val, y_pred)
        trip_acc   = trip_majority_vote_accuracy(val_df, y_pred, LABEL_COL, TRIP_COL)

        print(f"[EVAL] Window-level accuracy          : {window_acc:.4f}")
        print(f"[EVAL] Trip-level (majority vote) acc : {trip_acc:.4f}")

        report = classification_report(y_val, y_pred, target_names=class_names, digits=4)
        print(report)

        # Save confusion matrix figure
        cm       = confusion_matrix(y_val, y_pred)
        cm_path  = os.path.join(OUT_DIR, f"confusion_{name}.png")
        plot_confusion_matrix(cm, class_names, f"Confusion Matrix — {name}", cm_path)
        print(f"[SAVE] Confusion matrix : {cm_path}")

        # Serialise the fitted pipeline
        model_path = os.path.join(OUT_DIR, f"{name}.joblib")
        joblib.dump(pipe, model_path)
        print(f"[SAVE] Model            : {model_path}")

        # Save text classification report
        report_path = os.path.join(OUT_DIR, f"report_{name}.txt")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"{name}\n")
            fh.write(f"Window Accuracy              : {window_acc:.6f}\n")
            fh.write(f"Trip Accuracy (majority vote): {trip_acc:.6f}\n\n")
            fh.write(report)
        print(f"[SAVE] Report           : {report_path}")

        results.append({
            "model":                      name,
            "window_accuracy":            float(window_acc),
            "trip_accuracy_majority_vote": float(trip_acc),
        })

    # --- Save ranked summary table ---
    summary_df  = (
        pd.DataFrame(results)
        .sort_values(by="window_accuracy", ascending=False)
    )
    summary_csv = os.path.join(OUT_DIR, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print(f"[DONE] Summary saved: {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()