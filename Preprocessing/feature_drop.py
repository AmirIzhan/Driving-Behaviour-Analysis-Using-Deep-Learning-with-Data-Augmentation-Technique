"""
Feature_drop.py
===============
Purpose:
    Removes irrelevant or redundant columns from the raw combined
    driving-behaviour dataset to produce a leaner feature set for
    downstream preprocessing and model training.

Input:
    Combined_Behaviour.csv  — raw dataset containing all recorded columns.

Output:
    Combined_Behaviour_featuredrop.csv  — dataset with specified columns
                                          removed, ready for preprocessing.

"""

import pandas as pd

# Constants

INPUT_CSV = "data/Raw.csv"
OUTPUT_CSV = "data/Feature_Dropped.csv"


COLUMNS_TO_DROP = [
    "timestamp",
    "sys_active",
    "lat_interp",
    "lon_interp",
    "altitude_interp",
    "v_acc_interp",
    "h_acc_interp",
    "course_interp",
]



# Main


def drop_features(input_path: str, output_path: str, columns: list) -> pd.DataFrame:
    """Load the dataset, drop specified columns, and save the result."""
    df = pd.read_csv(input_path)

    # Drop only columns that actually exist to avoid KeyError on re-runs
    existing_columns = [col for col in columns if col in df.columns]
    df = df.drop(columns=existing_columns)

    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """Entry point: orchestrate feature dropping and report results."""
    df = drop_features(INPUT_CSV, OUTPUT_CSV, COLUMNS_TO_DROP)

    print("Features dropped successfully.")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"\nRemaining columns ({len(df.columns)}):")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()