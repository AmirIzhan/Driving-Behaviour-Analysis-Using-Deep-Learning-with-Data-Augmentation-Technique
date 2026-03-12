## Data Pipeline

The UAH-DriveSet dataset was used as the primary data source, containing
multi-sensor driving data from accelerometer and GPS signals across 3 behaviour classes:

| Label | Behaviour  |
|-------|------------|
| 0     | Aggressive |
| 1     | Drowsy     |
| 2     | Normal     |

### Step 1 — Convert Raw Text to CSV (`txt_to_csv.py`)

The raw dataset was provided as `.txt` files with no column headers, making
it unsuitable for direct analysis. This script converts each raw text file
into a structured CSV with clearly labelled columns.

### Step 2 — Sensor Synchronisation (`interpolate.py`)

The accelerometer and GPS sensors run at different sampling rates, resulting
in misaligned timestamps between the two data streams. This script applies
interpolation to project GPS signals onto the accelerometer timestamp grid,
ensuring consistent sampling across both sensors. The two datasets are then
merged into a single combined dataset.

### Step 3 — Feature Removal (`feature_drop.py`)

The following 8 irrelevant columns were removed as they do not contribute
to driving behaviour classification:

| Removed Feature    |
|--------------------|
| `timestamp`        |
| `sys_active`       |
| `lat_interp`       |
| `lon_interp`       |
| `altitude_interp`  |
| `v_acc_interp`     |
| `h_acc_interp`     |
| `course_interp`    |

### Step 4 — Cleaning and Normalisation (`preprocessing_cleaning.py`)

The dataset underwent:
- Removal of duplicate rows and rows with excessive missing values
- Per-trip winsorisation at [0.5%, 99.5%] to handle sensor spike outliers
- RobustScaler normalisation (median/IQR — robust to remaining outliers)
- Label encoding of the behaviour column

Output: `Preprocessed.csv`

### Step 5 — Train / Validation Split (`split_data.py`)

The dataset was split at the **trip level** to prevent data leakage —
entire trips are assigned to either train or validation, never split across both.

| Split      | Ratio | File           |
|------------|-------|----------------|
| Training   | 80%   | `Train.csv`    |
| Validation | 20%   | `Validation.csv` |