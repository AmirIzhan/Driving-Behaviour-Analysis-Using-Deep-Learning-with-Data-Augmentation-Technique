

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV  = "data/Raw.csv"
OUTPUT_CSV = "data/Preprocessed.csv"
MAP_JSON   = "data/label_mapping.json"

LABEL_COL = "behaviour"
TRIP_COL  = "trip_id"

# Columns that are metadata / targets — excluded from scaling
NON_FEATURE_COLS    = {LABEL_COL, TRIP_COL}

# Binary road-type flags kept as-is (no scaling needed)
BINARY_CONTEXT_COLS = {"type_of_road_highway", "type_of_road_rural"}

# Quantile bounds for per-trip winsorisation (clips extreme sensor spikes)
LOW_QUANTILE  = 0.005   # 0.5th percentile
HIGH_QUANTILE = 0.995   # 99.5th percentile

# Drop rows where more than this fraction of numeric features are missing
MAX_MISSING_FRACTION = 0.20

# After RobustScaler, cap residual spikes to prevent exploding loss values
NORMALISED_CAP = (-10, 10)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    
    # Infinite values break scalers and statistics — replace with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(how="all")       # Remove fully empty rows
    df = df.drop_duplicates()       # Remove exact duplicates

    # Provide a fallback trip identifier so per-trip logic never crashes
    if TRIP_COL not in df.columns:
        df[TRIP_COL] = "trip_000001"

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing required label column: '{LABEL_COL}'")

    return df


def identify_feature_columns(df: pd.DataFrame) -> list:
    
    present_binary = [col for col in BINARY_CONTEXT_COLS if col in df.columns]
    all_numeric    = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        col for col in all_numeric
        if col not in NON_FEATURE_COLS and col not in present_binary
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found to normalise.")

    return feature_cols


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def fill_missing_with_trip_median(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
   
    def _impute_group(group: pd.DataFrame) -> pd.DataFrame:
        group[feature_cols] = group[feature_cols].apply(
            lambda series: series.fillna(series.median()), axis=0
        )
        return group

    return df.groupby(TRIP_COL, group_keys=False).apply(_impute_group)


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def winsorise_per_trip(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    
    def _clip_group(group: pd.DataFrame) -> pd.DataFrame:
        for col in feature_cols:
            lo = group[col].quantile(LOW_QUANTILE)
            hi = group[col].quantile(HIGH_QUANTILE)

            # Skip columns that are constant or have degenerate quantiles
            if pd.isna(lo) or pd.isna(hi) or lo == hi:
                continue

            group[col] = group[col].clip(lower=lo, upper=hi)
        return group

    return df.groupby(TRIP_COL, group_keys=False).apply(_clip_group)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Final safety cap: rare spikes can still survive winsorisation
    df[feature_cols] = df[feature_cols].clip(*NORMALISED_CAP)

    return df, scaler


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame) -> tuple:
    
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df[LABEL_COL].astype(str))

    label_mapping = {
        cls: int(idx)
        for idx, cls in enumerate(encoder.classes_)
    }
    return df, label_mapping


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metadata(
    label_mapping: dict,
    feature_cols: list,
    present_binary: list,
) -> None:
    
    metadata = {
        "label_column":                      LABEL_COL,
        "encoded_column":                    "label",
        "mapping":                           label_mapping,
        "feature_columns_scaled":            feature_cols,
        "binary_context_columns_kept_unscaled": present_binary,
        "scaler":                            "RobustScaler(median/IQR) + per-trip quantile clipping",
        "clip_quantiles":                    {"low": LOW_QUANTILE, "high": HIGH_QUANTILE},
        "normalised_cap":                    list(NORMALISED_CAP),
    }

    with open(MAP_JSON, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full cleaning, normalisation, and encoding pipeline."""
    df = pd.read_csv(INPUT_CSV)

    # --- Step 1: Basic cleaning ---
    df = basic_clean(df)

    # --- Step 2: Identify columns to scale ---
    present_binary = [col for col in BINARY_CONTEXT_COLS if col in df.columns]
    feature_cols   = identify_feature_columns(df)

    # --- Step 3: Drop rows with excessive missing values ---
    missing_fraction = df[feature_cols].isna().mean(axis=1)
    df = df.loc[missing_fraction <= MAX_MISSING_FRACTION].copy()

    # --- Step 4: Impute remaining NaNs with per-trip median ---
    df = fill_missing_with_trip_median(df, feature_cols)

    # --- Step 5: Winsorise outliers per trip ---
    df = winsorise_per_trip(df, feature_cols)

    # --- Step 6: Normalise with RobustScaler ---
    df, _ = normalise_features(df, feature_cols)

    # --- Step 7: Encode string labels to integers ---
    df, label_mapping = encode_labels(df)

    # --- Step 8: Persist outputs ---
    df.to_csv(OUTPUT_CSV, index=False)
    save_metadata(label_mapping, feature_cols, present_binary)

    print("✅ Done.")
    print(f"Saved dataset : {OUTPUT_CSV}")
    print(f"Saved metadata: {MAP_JSON}")
    print("\nLabel mapping:")
    print(label_mapping)
    print("\nScaled feature columns:")
    print(feature_cols)


if __name__ == "__main__":
    main()