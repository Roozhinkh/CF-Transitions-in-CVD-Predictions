# blood pressure = hltprhb
from pathlib import Path
import pandas as pd
import joblib
import dice_ml


def main():
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "ess_model_ready.csv"
    model_path = project_root / "models" / "rf_hltprhb.pkl"
    out_dir = project_root / "outputs" / "counterfactuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_col = "hltprhb"
    outcome_cols = ["hltprhb", "hltprhc", "hltprdi"]

    df = pd.read_csv(data_path)
    print("Loaded data:", data_path)
    print("Shape:", df.shape)
    print()

    feature_cols = [c for c in df.columns if c not in outcome_cols]

    df_dice = df[feature_cols + [target_col]].copy()


    model = joblib.load(model_path)
    print("Loaded model:", model_path)
    print()

    # Immutable feature: gender (do not allow changes)
    immutable_features = ["gndr"]
    features_to_vary = [c for c in feature_cols if c not in immutable_features]

    print("Immutable features:", immutable_features)
    print("Features allowed to vary:", features_to_vary)
    print()

    dice_data = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=["bmi"],
        outcome_name=target_col
    )

    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier"
    )

    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    positives = df_dice[df_dice[target_col] == 1]
    if positives.empty:
        raise RuntimeError(f"No positive cases found for {target_col}=1.")

    query_instance = positives.sample(n=1, random_state=42)[feature_cols]
    print("Selected query instance (first row shown):")
    print(query_instance.head(1))
    print()

    cf = explainer.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class=0,
        features_to_vary=features_to_vary
    )

    cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
    out_path = out_dir / "dice_bp_counterfactuals.csv"
    cf_df.to_csv(out_path, index=False)

    print("Counterfactuals generated and saved to:", out_path)
    print("\n DiCE (blood pressure) completed.")


if __name__ == "__main__":
    main()
