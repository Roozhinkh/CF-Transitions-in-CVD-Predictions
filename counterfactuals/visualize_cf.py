from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        print(f"\nCF #{i}: predicted_risk={risk:.4f}  meets_half_risk_target={hit}")
        if not changes_sorted:
            print("  (No feature changes detected)")
            continue

        for col, orig, new in changes_sorted[:top_k]:
            print(f"  - {col}: {orig} → {new}")

        if len(changes_sorted) > top_k:
            print(f"  ... and {len(changes_sorted) - top_k} more changes.")


def _save_heatmap(
    df_matrix: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    """Save a basic heatmap using matplotlib only (no seaborn)."""
    numeric = df_matrix.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * numeric.shape[1]), max(4, 0.6 * numeric.shape[0])))
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
        "--row",
        type=int,
        required=True,
        help="Row index used when generating the DiCE outputs (e.g., 0, 42).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "outputs" / "counterfactuals"

    query_path = out_dir / f"dice_query_row{args.row}.csv"
    cf_path = out_dir / f"dice_counterfactuals_row{args.row}.csv"

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    if not cf_path.exists():
        raise FileNotFoundError(f"Counterfactual file not found: {cf_path}")

    query_df = pd.read_csv(query_path)
    cf_df = pd.read_csv(cf_path)

    if len(query_df) != 1:
        raise ValueError(f"Expected query file to contain exactly 1 row, got {len(query_df)}.")

    query_row = query_df.iloc[0]

    feature_cols = _infer_feature_cols(query_df, cf_df)

    matrix = pd.concat(
        [
            query_df[feature_cols].assign(_label="Original"),
            cf_df[feature_cols].assign(_label=[f"CF{i}" for i in range(len(cf_df))]),
        ],
        ignore_index=True,
    ).set_index("_label")

    orig_vals = pd.to_numeric(query_df[feature_cols].iloc[0], errors="coerce")
    deltas = cf_df[feature_cols].apply(pd.to_numeric, errors="coerce").sub(orig_vals, axis=1)
    deltas.index = [f"CF{i}" for i in range(len(deltas))]

    heatmap_path = out_dir / f"row{args.row}_heatmap.png"
    delta_path = out_dir / f"row{args.row}_delta_heatmap.png"

    _save_heatmap(matrix, heatmap_path, title=f"Original vs Counterfactuals (row {args.row})")
    _save_heatmap(deltas, delta_path, title=f"Counterfactual Changes (CF - Original) (row {args.row})")

    _print_recommendations(query_row, cf_df, feature_cols, top_k=10)

    print(f"\nSaved heatmap to: {heatmap_path}")
    print(f"Saved delta-heatmap to: {delta_path}\n")


if __name__ == "__main__":
    main()
