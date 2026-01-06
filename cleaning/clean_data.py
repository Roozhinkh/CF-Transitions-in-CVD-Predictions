from pathlib import Path

import pandas as pd
import numpy as np

def main():
    project_root = Path(__file__).resolve().parent.parent
    print(project_root)
    data_path = project_root / "data" 
    input_path = data_path / "ess.csv"
    df = pd.read_csv(input_path)

    print("Read file:", input_path)
    print("Amount of rows and columns:", df.shape)
    print()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped column: 'Unnamed: 0' (auto-generated index column).")
        print("New shape:", df.shape)
        print()

    print("Columns in the dataset:")
    print(df.columns.tolist())
    print()

    if {"height", "weighta"}.issubset(df.columns):
        df["bmi"] = df["weighta"] / (df["height"] / 100) ** 2

        before_bmi = df.shape[0]
        df = df[(df["bmi"] >= 15) & (df["bmi"] <= 60)]
        after_bmi = df.shape[0]

        print("Created column 'bmi'.")
        print(f"Rows before BMI filtering: {before_bmi}, after filtering: {after_bmi}")
        print("BMI statistics:")
        print(df["bmi"].describe())
        print()
    else:
        raise ValueError(
            "Required columns for BMI calculation ('height', 'weighta') "
            "not found in the dataset."
        )

    required_outcomes = {"hltprhc", "hltprhb", "hltprdi"}
    if not required_outcomes.issubset(df.columns):
        missing = required_outcomes - set(df.columns)
        raise ValueError(f"Required outcome columns not found in the dataset: {missing}")

    df["cvd_any"] = (
        (df["hltprhc"] == 1) |
        (df["hltprhb"] == 1) |
        (df["hltprdi"] == 1)
    ).astype(int)

    print("Created target column 'cvd_any' (combined CVD indicator).")
    print("Distribution (0 = no CVD, 1 = CVD):")
    print(df["cvd_any"].value_counts())
    print(df["cvd_any"].value_counts(normalize=True))
    print()

    # Core variables according to ESS documentation:
    #  - etfruit  : frequency of fruit consumption
    #  - eatveg   : frequency of vegetable consumption
    #  - cgtsmok  : smoking behavior
    #  - alcfreq  : alcohol consumption frequency
    #  - slprl    : sleep problems
    #  - paccnois : exposure to noise
    #  - bmi      : body mass index (newly created)
    #  - gndr     : gender
    # Additionally included:
    #  - health   : self-rated health
    #  - dosprt   : sport/physical activity
    #  - sclmeet  : frequency of social meetings
    #  - inprdsc  : perceived discrimination
    #  - ctrlife  : perceived control over life

    feature_cols = [
        "etfruit",
        "eatveg",
        "cgtsmok",
        "alcfreq",
        "slprl",
        "paccnois",
        "bmi",
        "gndr",
        "health",
        "dosprt",
        "sclmeet",
        "inprdsc",
        "ctrlife",
    ]

    missing_features = [c for c in feature_cols if c not in df.columns]
    if len(missing_features) > 0:
        raise ValueError(
            f"The following feature columns are missing from the dataset: {missing_features}"
        )

    cols_for_model = feature_cols + ["cvd_any"]
    df_model = df[cols_for_model].copy()

    print("Number of rows before removing NaN in features/target:", df_model.shape[0])
    df_model = df_model.dropna()
    print("Number of rows after dropna:", df_model.shape[0])
    print()

    print("Descriptive statistics for features:")
    print(df_model[feature_cols].describe())
    print()

    print("Distribution of 'cvd_any' after cleaning:")
    print(df_model["cvd_any"].value_counts())
    print(df_model["cvd_any"].value_counts(normalize=True))
    print()

    full_clean_path = data_path / "ess_clean_full.csv"
    df.to_csv(full_clean_path, index=False)
    print(f"Saved fully cleaned dataset to: {full_clean_path}")

    model_ready_path = data_path / "ess_model_ready.csv"
    df_model.to_csv(model_ready_path, index=False)
    print(f"Saved model-ready dataset to: {model_ready_path}")

    print("\n Data cleaning completed.")


if __name__ == "__main__":
    main()