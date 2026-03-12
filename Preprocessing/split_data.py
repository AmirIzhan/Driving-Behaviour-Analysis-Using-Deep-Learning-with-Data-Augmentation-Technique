

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_CSV  = "data/preprocessed.csv"
TRAIN_CSV  = "data/Train.csv"
VAL_CSV    = "data/Validation.csv"

TRIP_COL   = "trip_id"
LABEL_COL  = "label"

TRAIN_RATIO = 0.80   # 80 % of trips go to training
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def split_trips(trip_ids: np.ndarray, train_ratio: float, seed: int) -> tuple:
  
    rng = np.random.default_rng(seed)
    rng.shuffle(trip_ids)                        # In-place shuffle

    split_idx   = int(len(trip_ids) * train_ratio)
    train_trips = set(trip_ids[:split_idx])
    val_trips   = set(trip_ids[split_idx:])

    return train_trips, val_trips


def assert_no_leakage(train_trips: set, val_trips: set) -> None:
    
    overlap = train_trips.intersection(val_trips)
    # FIX: replaced `assert` with an explicit ValueError so this check
    # cannot be silently disabled when Python runs in optimised mode (-O).
    if len(overlap) != 0:
        raise ValueError(f"❌ Trip leakage detected! Overlapping trips: {overlap}")


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_trips: set,
    val_trips: set,
) -> None:
   
    print("\n✅ Split completed (by trip — no data leakage)")
    print(f"Train trips : {len(train_trips)}")
    print(f"Val trips   : {len(val_trips)}")
    print(f"Train rows  : {len(train_df)}")
    print(f"Val rows    : {len(val_df)}")

    print("\nLabel distribution (train):")
    print(train_df[LABEL_COL].value_counts())

    print("\nLabel distribution (val):")
    print(val_df[LABEL_COL].value_counts())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the trip-level train/validation split."""
    df = pd.read_csv(INPUT_CSV)

    # FIX: validate both required columns upfront before any processing,
    # so missing-column errors surface early with a clear message.
    for col in (TRIP_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    unique_trips = df[TRIP_COL].unique()
    print(f"Total trips: {len(unique_trips)}")

    # Split trip IDs (not rows) to prevent any within-trip leakage
    train_trips, val_trips = split_trips(unique_trips, TRAIN_RATIO, RANDOM_SEED)

    # Partition the dataframe according to trip membership
    train_df = df[df[TRIP_COL].isin(train_trips)].copy()
    val_df   = df[df[TRIP_COL].isin(val_trips)].copy()

    # Safety check: confirm there is truly zero overlap
    assert_no_leakage(train_trips, val_trips)

    # Persist both splits
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    print_split_summary(train_df, val_df, train_trips, val_trips)


if __name__ == "__main__":
    main()