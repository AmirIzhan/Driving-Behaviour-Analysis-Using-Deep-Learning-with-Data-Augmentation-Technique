"""
interpolate.py
=========================
Purpose:
    Merges GPS measurements onto the accelerometer timeline by interpolating
    each GPS signal at the accelerometer's higher-frequency timestamps.

    The UAH-DRIVESET records GPS and accelerometer data at different sampling
    rates.  Direct row-by-row joining would misalign the signals.  Linear
    interpolation aligns both streams to the accelerometer timeline so every
    accelerometer row carries a corresponding GPS reading.

Inputs:
    RAW_GPS.csv             — GPS recordings in the trip folder (pre-parsed).
    RAW_ACCELEROMETERS.csv  — Accelerometer recordings in the trip folder.

Output:
    ACC_with_GPS_interpolated.csv  — Accelerometer rows enriched with
                                      interpolated GPS columns, saved in the
                                      same trip folder.


"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
TRIP_DIR = SCRIPT_DIR.parent / "Data"

GPS_FILE = TRIP_DIR / "RAW_GPS.csv"
ACC_FILE = TRIP_DIR / "RAW_ACCELEROMETERS.csv"
OUT_FILE = TRIP_DIR / "ACC_with_GPS_interpolated.csv"

TIMESTAMP_COL = "timestamp"

# GPS columns to interpolate onto the accelerometer timeline
GPS_COLUMNS_TO_INTERPOLATE = [
    "speed",        # Vehicle speed (km/h or m/s depending on dataset version)
    "lat",          # Latitude (degrees)
    "lon",          # Longitude (degrees)
    "altitude",     # Altitude above sea level (m)
    "v_acc",        # Vertical GPS accuracy estimate (m)
    "h_acc",        # Horizontal GPS accuracy estimate (m)
    "course",       # Heading direction (degrees from north)
    "difcourse",    # Rate of change of heading (degrees/s)
]

# Suffix appended to each interpolated column name in the output
INTERP_SUFFIX = "_interp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_sort(path: Path, sort_col: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    return df.sort_values(sort_col).reset_index(drop=True)


def interpolate_gps_onto_acc(
    acc: pd.DataFrame,
    gps: pd.DataFrame,
    gps_cols: list,
    timestamp_col: str,
    suffix: str,
) -> pd.DataFrame:
    
    acc_timestamps = acc[timestamp_col].values
    gps_timestamps = gps[timestamp_col].values

    result = acc.copy()
    for col in gps_cols:
        result[col + suffix] = np.interp(
            x=acc_timestamps,   # Points at which to evaluate
            xp=gps_timestamps,  # Known x-coordinates (GPS sample times)
            fp=gps[col].values, # Known y-values (GPS measurements)
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load GPS and accelerometer data, interpolate, and save the result."""
    # Load and sort both files by timestamp
    gps = load_and_sort(GPS_FILE, TIMESTAMP_COL)
    acc = load_and_sort(ACC_FILE, TIMESTAMP_COL)

    # Interpolate all GPS columns onto the accelerometer timeline
    merged = interpolate_gps_onto_acc(
        acc=acc,
        gps=gps,
        gps_cols=GPS_COLUMNS_TO_INTERPOLATE,
        timestamp_col=TIMESTAMP_COL,
        suffix=INTERP_SUFFIX,
    )

    merged.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved: {OUT_FILE} | shape={merged.shape}")


if __name__ == "__main__":
    main()