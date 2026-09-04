"""
Data Preparation Script
------------------------
Loads the raw dataset from the repository data/ folder, cleans it,
engineers the feature set, splits it into train and test sets, and
saves both splits locally so they can be picked up as a workflow
artifact by the next job in the pipeline.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "tourism.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "ProdTaken"
ID_COLUMNS = ["CustomerID", "Unnamed: 0"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns], errors="ignore")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    if "DurationOfPitch" in df.columns:
        df["DurationOfPitch"] = df["DurationOfPitch"].clip(upper=df["DurationOfPitch"].quantile(0.99))
    if "NumberOfTrips" in df.columns:
        df["NumberOfTrips"] = df["NumberOfTrips"].clip(upper=df["NumberOfTrips"].quantile(0.99))

    num_cols = df.select_dtypes(include="number").columns.drop(TARGET, errors="ignore")
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()
    return df


def prepare_data(raw_path: Path = RAW_PATH, output_dir: Path = PROCESSED_DIR, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(raw_path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Train shape           : {X_train.shape}")
    print(f"Test shape            : {X_test.shape}")
    print(f"Splits saved to       : {output_dir}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
