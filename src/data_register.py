"""
Data Registration Script
-------------------------
Validates that the raw tourism dataset placed in the data/ folder has the
expected structure before it is allowed to enter the pipeline, and prints
a summary report of the dataset.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and validate its schema, then return it."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    print("=" * 60)
    print("DATA REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file            : {path}")
    print(f"Rows                   : {df.shape[0]}")
    print(f"Columns                : {df.shape[1]}")
    print(f"Expected columns found : {len(EXPECTED_COLUMNS)}/{len(EXPECTED_COLUMNS)}")
    print(f"Duplicate rows         : {df.duplicated().sum()}")
    print(f"Duplicate CustomerIDs  : {df['CustomerID'].duplicated().sum()}")
    print(f"Missing values (total) : {int(df.isnull().sum().sum())}")
    print(f"Target column          : ProdTaken")
    print(f"Target distribution    :\n{df['ProdTaken'].value_counts(normalize=True).round(3).to_string()}")
    print("=" * 60)
    print("Dataset registered successfully.")

    return df


if __name__ == "__main__":
    try:
        register_dataset()
    except Exception as exc:
        print(f"Data registration failed: {exc}")
        sys.exit(1)
