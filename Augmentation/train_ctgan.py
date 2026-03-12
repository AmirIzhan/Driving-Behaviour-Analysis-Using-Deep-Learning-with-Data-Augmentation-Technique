import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)



SCRIPT_DIR     = Path(__file__).parent                        
ROOT_DIR       = SCRIPT_DIR.parent                          
DATA_DIR       = ROOT_DIR / "Data"                           
SAVED_MODELS   = ROOT_DIR / "saved_models"
SAVED_MODELS.mkdir(exist_ok=True)                            

# ---------------------------------------------------------------------------
# Behaviours to train — add or remove entries here as needed
# ---------------------------------------------------------------------------

BEHAVIOURS = [
    {
        "file":       DATA_DIR / "Drowsy Behaviour.xlsx",
        "label":      "drowsy",
        "n_generate": 35_937,
        "out_model":  SAVED_MODELS / "ctgan_drowsy.pkl",
        "out_synth":  DATA_DIR    / "synthetic_drowsy.csv",
        "out_clean":  DATA_DIR    / "drowsy_clean_used_for_ctgan.csv",
    },
    {
        "file":       DATA_DIR / "Aggressive Behaviour.xlsx",
        "label":      "aggressive",
        "n_generate": 57_064,
        "out_model":  SAVED_MODELS / "ctgan_aggressive.pkl",
        "out_synth":  DATA_DIR    / "synthetic_aggressive.csv",
        "out_clean":  DATA_DIR    / "aggressive_clean_used_for_ctgan.csv",
    },
]

# ---------------------------------------------------------------------------
# CTGAN settings
# ---------------------------------------------------------------------------

LABEL_COL  = "label"
DROP_COLS  = ["behaviour"]   # columns to remove before training
EPOCHS     = 300
BATCH_SIZE = 256             # will be adjusted to satisfy PAC constraint

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: '{ext}'. Expected .xlsx, .xls, or .csv.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Remove fully empty columns
    df = df.dropna(axis=1, how="all")

    # Remove explicitly unwanted columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # SDV handles mixed types better when object columns are explicit strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# PAC-safe batch size
# ---------------------------------------------------------------------------

def compute_pac_safe_batch_size(
    requested_batch_size: int,
    n_rows: int,
    pac: int = 10,
) -> int:
    if n_rows < pac:
        raise ValueError(
            f"Dataset too small for CTGAN (pac={pac}). "
            f"Need at least {pac} rows, got {n_rows}."
        )
    capped = min(requested_batch_size, n_rows)
    safe   = max(pac, (capped // pac) * pac)
    return safe


# ---------------------------------------------------------------------------
# Train and generate for one behaviour
# ---------------------------------------------------------------------------

def run_ctgan(behaviour: dict) -> None:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    label     = behaviour["label"]
    file_path = behaviour["file"]

    print(f"\n{'='*60}")
    print(f"  Behaviour : {label.upper()}")
    print(f"  Input     : {file_path}")
    print(f"{'='*60}")

    # --- Step 1: Load and clean ---
    if not file_path.exists():
        print(f"❌  File not found: {file_path}")
        print(f"    Skipping {label}. Add the file and re-run.")
        return

    df = read_table(file_path)
    df = clean_dataframe(df)
    df[LABEL_COL] = label

    # Save the exact data used for training (audit trail)
    df.to_csv(behaviour["out_clean"], index=False)
    print(f"✅ Loaded and cleaned — shape: {df.shape}")
    print(f"   Saved clean copy : {behaviour['out_clean']}")

    # --- Step 2: Configure metadata ---
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(LABEL_COL, sdtype="categorical")

    # --- Step 3: PAC-safe batch size ---
    PAC             = 10
    safe_batch_size = compute_pac_safe_batch_size(BATCH_SIZE, n_rows=len(df), pac=PAC)
    print(
        f"  Requested batch_size={BATCH_SIZE} → "
        f"Using batch_size={safe_batch_size} (PAC={PAC}, rows={len(df)})"
    )

    # --- Step 4: Train ---
    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=EPOCHS,
        batch_size=safe_batch_size,
        verbose=True,
    )
    print(f"\n Training CTGAN on {label} behaviour...")
    ctgan.fit(df)
    print(f"✅ Training complete!")

    # --- Step 5: Save model ---
    ctgan.save(str(behaviour["out_model"]))
    print(f"✅ Model saved   : {behaviour['out_model']}")

    # --- Step 6: Generate synthetic samples ---
    n = behaviour["n_generate"]
    print(f"\n Generating {n:,} synthetic {label} rows...")
    df_synthetic = ctgan.sample(n)
    df_synthetic[LABEL_COL] = label          # safety guard

    df_synthetic.to_csv(behaviour["out_synth"], index=False)
    print(f"✅ Synthetic CSV : {behaviour['out_synth']}")
    print(f"   Shape         : {df_synthetic.shape}")
    print(df_synthetic.head(3))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*60)
    print("  CTGAN Training — Drowsy + Aggressive Behaviours")
    print("="*60)
    print(f"  Root dir : {ROOT_DIR}")
    print(f"  Data dir : {DATA_DIR}")

    for behaviour in BEHAVIOURS:
        run_ctgan(behaviour)

    print(f"\n{'='*60}")
    print("  ALL DONE")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for b in BEHAVIOURS:
        print(f"  • {b['out_synth']}")
    print("\nSaved models:")
    for b in BEHAVIOURS:
        print(f"  • {b['out_model']}")


if __name__ == "__main__":
    main()
