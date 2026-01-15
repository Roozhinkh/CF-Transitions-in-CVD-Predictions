from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
import dice_ml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DiCE counterfactual explanations for a chosen individual (row index) "
            "in ess_model_ready.csv using a trained RF model."
        )
    )
    parser.add_argument(
        "--person-index",
        type=int,
        default=None,
        help=(
            "Row index (0-based) of the individual to explain. "
            "If omitted, a high-risk individual (p>=0.5) is selected deterministically using --seed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only when --person-index is not provided. Default: 42.",
    )
    parser.add_argument(
        "--total-cfs",
        type=int,
        default=5,
        help="Number of counterfactuals to generate. Default: 5.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "ess_model_ready.csv"
    model_path = project_root / "models" / "rf_cvd.pkl"
    out_dir = project_root / "outputs" / "counterfactuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    target_col = "cvd_any"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}.")

    # Immutable features in the dataset
    # Gender is treated as immutable when present in the dataset. 
    immutable_features = []
    if "gndr" in df.columns:
        immutable_features.append("gndr")
    else:
        print(
            "Note: 'gndr' not found in dataset. "
            "No gender-based immutability constraint applied."
        )

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()

    model = joblib.load(model_path)

    # Pick individual
    probs = model.predict_proba(X)[:, 1]

    if args.person_index is not None:
        idx = int(args.person_index)
        if idx < 0 or idx >= len(df):
            raise IndexError(f"--person-index {idx} is out of bounds (0..{len(df)-1}).")
    else:
        high_idx = np.where(probs >= 0.5)[0]
        if len(high_idx) == 0:
            raise ValueError(
                "No individuals with predicted risk >= 0.5 were found. "
                "Provide --person-index or adjust the selection rule."
            )
        rng = np.random.default_rng(args.seed)
        idx = int(rng.choice(high_idx))

    query_instance = X.iloc[[idx]].copy()

    current_prob = float(probs[idx])
    target_prob = current_prob / 2.0  # "halve the risk" target

    print(f"\nSelected row index: {idx}")
    print(f"Current predicted risk P({target_col}=1): {current_prob:.4f}")
    print(f"Target threshold (halve risk): <= {target_prob:.4f}")

    # Decide what DiCE may change 
    # Everything except immutable features may vary.
    features_to_vary = [c for c in feature_cols if c not in set(immutable_features)]
    print(f"Immutable (locked): {immutable_features}")
    print(f"Features allowed to vary: {len(features_to_vary)} / {len(feature_cols)}")

    continuous_features = [c for c in ["bmi"] if c in feature_cols]
    categorical_features = [c for c in feature_cols if c not in set(continuous_features)]

    #explain_with_alibi_anchors(model, X, query_instance, feature_cols)

    # These SHAP explanations are both very similar, but they do 
    # not get the exact same values. I have kept both for comparison.
    #explain_with_shap(model, X, query_instance, feature_cols, output_path=out_dir)

    #explain_with_alibi_shap(model, X, query_instance, feature_cols, output_path=out_dir)


    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name=target_col,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")

    # method="random" is typically robust with RF; this can later be tested with other methods
    exp = dice_ml.Dice(dice_data, dice_model, method="random")


    cf = exp.generate_counterfactuals(
    query_instances=query_instance,
    total_CFs=args.total_cfs,
    desired_class=0,
    features_to_vary=features_to_vary,
    )

    # Extract CFs
    cf_df = cf.cf_examples_list[0].final_cfs_df.copy()

    cf_probs = model.predict_proba(cf_df[feature_cols])[:, 1]
    cf_df["predicted_risk"] = cf_probs
    cf_df["meets_half_risk_target"] = cf_df["predicted_risk"] <= target_prob

    print("\nCounterfactuals (with predicted risks):")
    cols_to_show = ["predicted_risk", "meets_half_risk_target"] + feature_cols
    print(cf_df[cols_to_show].head(args.total_cfs))

    query_out = out_dir / f"dice_query_row{idx}.csv"
    cf_out = out_dir / f"dice_counterfactuals_row{idx}.csv"

    query_instance_out = query_instance.copy()
    query_instance_out["predicted_risk"] = current_prob
    query_instance_out.to_csv(query_out, index=False)
    cf_df.to_csv(cf_out, index=False)

    print(f"\nSaved query instance to: {query_out}")
    print(f"Saved counterfactuals to: {cf_out}")
    print("\nDiCE counterfactual generation completed.\n")


if __name__ == "__main__":
    main()
