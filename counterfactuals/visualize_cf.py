from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


OUTCOME_COLS = ["hltprhb", "hltprhc", "hltprdi"]

TASK_CONFIG = {
    "bp": {
        "target_col": "hltprhb",
        "model_file": "rf_hltprhb.pkl",
        "cf_file": "dice_bp_counterfactuals.csv",
    },
    "hd": {
        "target_col": "hltprhc",
        "model_file": "rf_hltprhc.pkl",
        "cf_file": "dice_hd_counterfactuals.csv",
    },
    "dm": {
        "target_col": "hltprdi",
        "model_file": "rf_hltprdi.pkl",
        "cf_file": "dice_dm_counterfactuals.csv",
    },
}


def _infer_feature_cols(query_df: pd.DataFrame, cf_df: pd.DataFrame) -> list[str]:
    """Pick columns that represent features (exclude helper columns)."""
    helper_cols = {"predicted_risk", "meets_half_risk_target"}
    cols = [c for c in query_df.columns if c not in helper_cols]
    cols = [c for c in cols if c in cf_df.columns]
    return cols


def _print_recommendations(
    query_row: pd.Series, cf_df: pd.DataFrame, feature_cols: list[str], top_k: int = 10
) -> None:
    """Print changes per counterfactual in a human-readable way."""
    print("\n=== Changes per Counterfactual (vs original) ===")
    for i, row in cf_df.iterrows():
        changes = []
        for col in feature_cols:
            orig = query_row[col]
            new = row[col]
            if pd.isna(orig) or pd.isna(new):
                continue
            if orig != new:
                changes.append((col, orig, new))

        def _change_mag(t):
            _, o, n = t
            try:
                return abs(float(n) - float(o))
            except Exception:
                return 0.0

        changes_sorted = sorted(changes, key=_change_mag, reverse=True)

        risk = row["predicted_risk"] if "predicted_risk" in row else None
        hit = row["meets_half_risk_target"] if "meets_half_risk_target" in row else None

        if risk is not None:
            print(f"\nCF #{i}: predicted_risk={risk:.4f}  meets_half_risk_target={hit}")
        else:
            print(f"\nCF #{i}: (no predicted_risk column found)")

        if not changes_sorted:
            print("  (No feature changes detected)")
            continue

        for col, orig, new in changes_sorted[:top_k]:
            print(f"  - {col}: {orig} → {new}")

        if len(changes_sorted) > top_k:
            print(f"  ... and {len(changes_sorted) - top_k} more changes.")


def _save_heatmap(df_matrix: pd.DataFrame, out_path: Path, title: str) -> None:
    """Save a basic heatmap using matplotlib only (no seaborn)."""
    numeric = df_matrix.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(
        figsize=(max(10, 0.6 * numeric.shape[1]), max(4, 0.6 * numeric.shape[0]))
    )
    im = ax.imshow(numeric.values, aspect="auto")

    ax.set_title(title)
    ax.set_yticks(range(numeric.shape[0]))
    ax.set_yticklabels(df_matrix.index.tolist())
    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(df_matrix.columns.tolist(), rotation=90)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value (numeric)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize DiCE counterfactual outputs (heatmaps + change list)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),
        help="Which outcome/task to visualize: bp, hd, or dm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to reproduce the same query instance selection as dice_*.py (default: 42).",
    )
    args = parser.parse_args()

    cfg = TASK_CONFIG[args.task]
    target_col = cfg["target_col"]

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "ess_model_ready.csv"
    model_path = project_root / "models" / cfg["model_file"]
    out_dir = project_root / "outputs" / "counterfactuals"
    cf_path = out_dir / cfg["cf_file"]

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not cf_path.exists():
        raise FileNotFoundError(f"Counterfactual file not found: {cf_path}")

    cf_df_raw = pd.read_csv(cf_path)

    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}.")
    for oc in OUTCOME_COLS:
        if oc not in df.columns:
            raise ValueError(f"Expected outcome column '{oc}' not found in {data_path}.")

    feature_cols = [c for c in df.columns if c not in OUTCOME_COLS]

    df_dice = df[feature_cols + [target_col]].copy()
    positives = df_dice[df_dice[target_col] == 1]
    if positives.empty:
        raise RuntimeError(f"No positive cases found for {target_col}=1; cannot reconstruct query instance.")

    query_instance = positives.sample(n=1, random_state=args.seed)[feature_cols].copy()

    print(f"\nTask: {args.task}  (target={target_col})")
    print(f"Reconstructed query instance using positives.sample(random_state={args.seed})")
    print("Selected query instance (first row shown):")
    print(query_instance.head(1))
    print()

    model = joblib.load(model_path)

    query_prob = float(model.predict_proba(query_instance[feature_cols])[:, 1][0])
    target_prob = query_prob / 2.0

    cf_features = cf_df_raw[feature_cols].copy()
    cf_probs = model.predict_proba(cf_features)[:, 1]

    cf_df = cf_df_raw.copy()
    cf_df["predicted_risk"] = cf_probs
    cf_df["meets_half_risk_target"] = cf_df["predicted_risk"] <= target_prob

    print(f"Original predicted risk (P({target_col}=1)): {query_prob:.4f}")
    print(f"Half-risk threshold: <= {target_prob:.4f}")

    query_df = query_instance.copy()
    query_df["predicted_risk"] = query_prob
    query_df["meets_half_risk_target"] = True  

    feature_cols_for_plot = _infer_feature_cols(query_df, cf_df)

    matrix = pd.concat(
        [
            query_df[feature_cols_for_plot].assign(_label="Original"),
            cf_df[feature_cols_for_plot].assign(_label=[f"CF{i}" for i in range(len(cf_df))]),
        ],
        ignore_index=True,
    ).set_index("_label")

    orig_vals = pd.to_numeric(query_df[feature_cols_for_plot].iloc[0], errors="coerce")
    deltas = cf_df[feature_cols_for_plot].apply(pd.to_numeric, errors="coerce").sub(orig_vals, axis=1)
    deltas.index = [f"CF{i}" for i in range(len(deltas))]

    heatmap_path = out_dir / f"{args.task}_heatmap.png"
    delta_path = out_dir / f"{args.task}_delta_heatmap.png"

    _save_heatmap(matrix, heatmap_path, title=f"Original vs Counterfactuals ({args.task})")
    _save_heatmap(deltas, delta_path, title=f"Counterfactual Changes (CF - Original) ({args.task})")

    _print_recommendations(query_df.iloc[0], cf_df, feature_cols_for_plot, top_k=10)

    print(f"\nSaved heatmap to: {heatmap_path}")
    print(f"Saved delta-heatmap to: {delta_path}\n")


if __name__ == "__main__":
    main()
