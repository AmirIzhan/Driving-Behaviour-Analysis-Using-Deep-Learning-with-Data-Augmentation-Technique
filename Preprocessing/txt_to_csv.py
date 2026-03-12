
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trip folder to process — update this path for each trip
SCRIPT_DIR = Path(__file__).parent
TRIP_DIR = SCRIPT_DIR.parent / "Data"

INPUT_FILE  = TRIP_DIR / "RAW_ACCELEROMETERS.txt"
OUTPUT_FILE = TRIP_DIR / "RAW_ACCELEROMETERS.csv"

GPS_INPUT_FILE  = TRIP_DIR / "RAW_GPS.txt"
GPS_OUTPUT_FILE = TRIP_DIR / "RAW_GPS.csv"

# Known column schema for RAW_ACCELEROMETERS.txt (in order)
BASE_COLUMNS = [
    "timestamp",    # Unix timestamp of the reading
    "sys_active",   # System-active flag
    "acc_x",        # Raw accelerometer X-axis (m/s²)
    "acc_y",        # Raw accelerometer Y-axis (m/s²)
    "acc_z",        # Raw accelerometer Z-axis (m/s²)
    "acc_x_kf",     # Kalman-filtered X-axis
    "acc_y_kf",     # Kalman-filtered Y-axis
    "acc_z_kf",     # Kalman-filtered Z-axis
    "roll",         # Roll angle (degrees)
    "pitch",        # Pitch angle (degrees)
    "yaw",          # Yaw angle (degrees)
]

# Known column schema for RAW_GPS.txt (in order)
GPS_COLUMNS = [
    "timestamp",    # Unix timestamp of the reading
    "speed",        # Vehicle speed (km/h or m/s depending on dataset version)
    "lat",          # Latitude (degrees)
    "lon",          # Longitude (degrees)
    "altitude",     # Altitude above sea level (m)
    "v_acc",        # Vertical GPS accuracy estimate (m)
    "h_acc",        # Horizontal GPS accuracy estimate (m)
    "course",       # Heading direction (degrees from north)
    "difcourse",    # Rate of change of heading (degrees/s)
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw_file(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited UAH-DRIVESET raw text file into a DataFrame."""
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def assign_columns(df: pd.DataFrame, base_cols: list, filename: str = "file") -> pd.DataFrame:
    """Assign known column names to a raw DataFrame, dropping any extra columns."""
    n_cols = df.shape[1]
    n_base = len(base_cols)

    if n_cols < n_base:
        raise ValueError(
            f"{filename} has only {n_cols} columns; "
            f"expected at least {n_base}."
        )

    # Label surplus columns generically so the assignment does not fail
    extra_cols = [f"extra{i}" for i in range(1, n_cols - n_base + 1)]
    df.columns = base_cols + extra_cols

    # Return only the documented columns
    return df[base_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse the raw accelerometer and GPS files and save them as named-column CSVs."""

    # --- Accelerometers ---
    raw_acc_df = load_raw_file(INPUT_FILE)
    acc_df = assign_columns(raw_acc_df, BASE_COLUMNS, filename="RAW_ACCELEROMETERS.txt")
    acc_df.to_csv(OUTPUT_FILE, index=False)
    print(
        f"✅ Saved: {OUTPUT_FILE} | "
        f"shape={acc_df.shape} | "
        f"total_input_cols={raw_acc_df.shape[1]}"
    )

    # --- GPS ---
    raw_gps_df = load_raw_file(GPS_INPUT_FILE)
    gps_df = assign_columns(raw_gps_df, GPS_COLUMNS, filename="RAW_GPS.txt")
    gps_df.to_csv(GPS_OUTPUT_FILE, index=False)
    print(
        f"✅ Saved: {GPS_OUTPUT_FILE} | "
        f"shape={gps_df.shape} | "
        f"total_input_cols={raw_gps_df.shape[1]}"
    )


if __name__ == "__main__":
    main()