from pathlib import Path
import sys
import numpy as np 
import pandas as pd 
import joblib
import dice_ml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from explain.alibi_anch import explain_with_alibi_anchors
from explain.shap import explain_with_shap
from explain.alibi_shap import explain_with_alibi_shap


def main():
    data_path = project_root / "data" / "ess_model_ready.csv"
    model_path = project_root / "models" / "rf_cvd.pkl"
    out_dir = project_root / "outputs" / "counterfactuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    target_col = "cvd_any"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    continuous_features = feature_cols  #This can be redefined later, e.g. if some features are categorical (such as gender)

    probs = model.predict_proba(X)[:, 1]
    high_idx = np.where(probs >= 0.5)[0]

    if len(high_idx) == 0:
        query_idx = int(np.argmax(probs))    
    else:
        rng = np.random.default_rng(42)
        query_idx = int(rng.choice(high_idx))

    query_instance = X.iloc[[query_idx]].copy()
    current_prob = float(probs[query_idx])
    target_prob = current_prob / 2.0

    print(f"Selected index: {query_idx}")
    print(f"Current predicted risk (P(cvd_any=1)): {current_prob:.3f}")
    print(f"Target (halve risk): <= {target_prob:.3f}")
    print()

    explain_with_alibi_anchors(model, X, query_instance, feature_cols)

    # These SHAP explanations are both very similar, but they do 
    # not get the exact same values. I have kept both for comparison.
    explain_with_shap(model, X, query_instance, feature_cols, output_path=out_dir)

    explain_with_alibi_shap(model, X, query_instance, feature_cols, output_path=out_dir)


    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name=target_col
    )

    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier"
    )

    dice = dice_ml.Dice(dice_data, dice_model, method="random")

    cf = dice.generate_counterfactuals(
        query_instance,
        total_CFs=5,
        desired_class=0
    )

    cf_df = cf.cf_examples_list[0].final_cfs_df

    cf_probs = model.predict_proba(cf_df[feature_cols])[:, 1]
    cf_df = cf_df.copy()
    cf_df["predicted_risk"] = cf_probs
    cf_df["meets_half_risk_target"] = cf_df["predicted_risk"] <= target_prob

    print("Counterfactuals generated (showing predicted risks):")
    print(cf_df[["predicted_risk", "meets_half_risk_target"] + feature_cols].head())
    print()

    query_out = out_dir / "dice_query_instance.csv"
    cf_out = out_dir / "dice_counterfactuals.csv"

    query_instance_out = query_instance.copy()
    query_instance_out["predicted_risk"] = current_prob
    query_instance_out.to_csv(query_out, index=False)
    cf_df.to_csv(cf_out, index=False)

    print(f"Saved query instance to: {query_out}")
    print(f"Saved counterfactuals to: {cf_out}")
    print("\n DiCE counterfactual generation completed.")

if __name__ == "__main__":
    main()


