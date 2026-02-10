import pandas as pd
import numpy as np
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "ess.csv"

    full_clean_path = project_root / "data" / "ess_clean_full.csv"
    model_ready_path = project_root / "data" / "ess_model_ready.csv"


    df = pd.read_csv(input_path)
    print(f"Read file: {input_path}")
    print("Rows and columns:", df.shape)
    print()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped column: 'Unnamed: 0' (auto-generated index column).")
        print("New shape:", df.shape)
        print()


    if {"height", "weighta"}.issubset(df.columns):
        df["bmi"] = df["weighta"] / (df["height"] / 100) ** 2

        before_bmi = df.shape[0]
        df = df[(df["bmi"] >= 15) & (df["bmi"] <= 60)]
        after_bmi = df.shape[0]

        print("Created column 'bmi'.")
        print(f"Rows before BMI filtering: {before_bmi}, after filtering: {after_bmi}")
        print()
    else:
        raise ValueError("Required columns for BMI calculation not found: 'height' and 'weighta'.")


    outcome_cols = ["hltprhc", "hltprhb", "hltprdi"]
    missing_outcomes = [c for c in outcome_cols if c not in df.columns]
    if missing_outcomes:
        raise ValueError(f"Missing required outcome columns: {missing_outcomes}")

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
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")


    cols_for_model = feature_cols + outcome_cols
    df_model = df[cols_for_model].copy()

    print("Rows before dropna:", df_model.shape[0])
    df_model = df_model.dropna()
    print("Rows after dropna:", df_model.shape[0])
    print()


    df.to_csv(full_clean_path, index=False)
    print(f"Saved fully cleaned dataset to: {full_clean_path}")

    df_model.to_csv(model_ready_path, index=False)
    print(f"Saved model-ready dataset to: {model_ready_path}")

    print("\n Data cleaning completed.")


if __name__ == "__main__":
    main()
