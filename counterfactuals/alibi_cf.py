import argparse
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

from alibi.explainers import Counterfactual

TASK_CONFIG = {
    "bp": "hltprhb",
    "hd": "hltprhc",
    "dm": "hltprdi",
}

DATA_PATH = "data/ess_model_ready.csv"
NN_PATH_TEMPLATE = "models/nn_{target}.keras"
SCALER_PATH_TEMPLATE = "models/scaler_nn_{target}.pkl"
META_PATH_TEMPLATE = "models/nn_{target}_meta.pkl"

OUTPUT_DIR = "outputs/counterfactuals"
IMMUTABLE_FEATURES = ["gndr"]


def select_query_instance(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    positives = df[df[target_col] == 1]
    return positives.sample(n=1, random_state=seed)


def compute_feature_ranges_unscaled(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    fr = {}
    for c in feature_cols:
        fr[c] = (float(df[c].min()), float(df[c].max()))
    return fr


def to_scaled_feature_range(fr_unscaled: dict, feature_cols: list[str], scaler) -> tuple[np.ndarray, np.ndarray]:
    mins = np.array([fr_unscaled[c][0] for c in feature_cols], dtype=np.float32).reshape(1, -1)
    maxs = np.array([fr_unscaled[c][1] for c in feature_cols], dtype=np.float32).reshape(1, -1)
    mins_s = scaler.transform(mins).reshape(-1).astype(np.float32)
    maxs_s = scaler.transform(maxs).reshape(-1).astype(np.float32)
    return mins_s, maxs_s


def print_cf_changes(orig_row: pd.Series, cf_row: pd.Series, predicted_risk: float, half_threshold: float, idx: int):
    meets_half = predicted_risk <= half_threshold
    print(f"\nCF #{idx}: predicted_risk={predicted_risk:.4f}  meets_half_risk_target={meets_half}")
    for col in orig_row.index:
        a = float(orig_row[col])
        b = float(cf_row[col])
        if not np.isclose(a, b):
            print(f"  - {col}: {a} → {b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["bp", "hd", "dm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cfs", type=int, default=3)
    args = parser.parse_args()

    task = args.task
    seed = args.seed
    n_cfs = args.n_cfs
    target_col = TASK_CONFIG[task]

    df = pd.read_csv(DATA_PATH)

    nn_path = NN_PATH_TEMPLATE.format(target=target_col)
    scaler_path = SCALER_PATH_TEMPLATE.format(target=target_col)
    meta_path = META_PATH_TEMPLATE.format(target=target_col)

    model = tf.keras.models.load_model(nn_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]

    # Pick same individual as DiCE (positive class, seeded)
    query = select_query_instance(df, target_col, seed)
    row_index = int(query.index[0])

    print(f"\nTask: {task}  (target={target_col})")
    print(f"Reconstructed query instance using positives.sample(random_state={seed})")
    print("Selected query instance (first row shown):")
    print(query[feature_cols])

    x_orig = query[feature_cols].values.astype(np.float32)          
    x_orig_scaled = scaler.transform(x_orig).astype(np.float32)     

    orig_risk = float(model.predict(x_orig_scaled, verbose=0).reshape(-1)[0])
    half_threshold = orig_risk / 2.0

    print(f"\nOriginal predicted risk (P({target_col}=1)): {orig_risk:.4f}")
    print(f"Half-risk threshold: <= {half_threshold:.4f}")

    fr_unscaled = compute_feature_ranges_unscaled(df, feature_cols)

    # Lock immutable features by min=max to the individual's value
    for c in IMMUTABLE_FEATURES:
        v = float(query[c].iloc[0])
        fr_unscaled[c] = (v, v)

    # Convert ranges to scaled space (NN input)
    mins_s, maxs_s = to_scaled_feature_range(fr_unscaled, feature_cols, scaler)
    feature_range_scaled = (mins_s, maxs_s)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        p1 = model.predict(X, verbose=0).astype(np.float32)
        if p1.ndim == 1:
            p1 = p1.reshape(-1, 1)
        p0 = 1.0 - p1
        return np.concatenate([p0, p1], axis=1)


    target_probas = [0.55, 0.65, 0.75]

    print("\n=== Changes per Counterfactual (vs original) ===")

    results = []
    seen = set()
    cf_idx = 0

    for tp in target_probas:
        if cf_idx >= n_cfs:
            break

        cf_explainer = Counterfactual(
            predict_fn,                              
            shape=(1, x_orig_scaled.shape[1]),
            distance_fn="l1",
            target_class=0,                          # class 0 = lower risk
            target_proba=float(tp),
            feature_range=feature_range_scaled,
            max_iter=2000,
            lam_init=1e-1,
            max_lam_steps=10,
            tol=1e-3,
            learning_rate_init=0.05,
            eps=0.01,
            init="identity",                         # REQUIRED by this alibi version
            sess=sess,
        )

        explanation = cf_explainer.explain(x_orig_scaled)

# --- robust extraction of CF from explanation (alibi versions differ) ---
        if explanation is None or getattr(explanation, "cf", None) is None:
            continue

        cf_obj = explanation.cf

# In some alibi versions, cf is a dict; in others it's already the array.
        if isinstance(cf_obj, dict):
            x_cf_scaled = cf_obj.get("X", None)
        else:
            x_cf_scaled = cf_obj

        if x_cf_scaled is None:
            continue

        x_cf_scaled = np.array(x_cf_scaled, dtype=np.float32).reshape(1, -1)
        x_cf = scaler.inverse_transform(x_cf_scaled).reshape(-1).astype(np.float32)

        key = tuple(np.round(x_cf, 4))
        if key in seen:
            continue
        seen.add(key)

        cf_risk = float(model.predict(x_cf_scaled, verbose=0).reshape(-1)[0])

        cf_series = pd.Series(x_cf, index=feature_cols)
        print_cf_changes(query[feature_cols].iloc[0], cf_series, cf_risk, half_threshold, cf_idx)

        row_dict = cf_series.to_dict()
        row_dict.update({
            "row_index": row_index,
            "task": task,
            "method": "alibi_counterfactual",
            "seed": seed,
            "target_proba_used": float(tp),
            "original_predicted_risk": orig_risk,
            "cf_predicted_risk": cf_risk,
            "meets_half_risk_target": (cf_risk <= half_threshold),
        })
        results.append(row_dict)
        cf_idx += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"alibi_{task}_counterfactuals.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved counterfactuals to: {out_path}")

    if cf_idx < n_cfs:
        print(f"WARNING: Only generated {cf_idx}/{n_cfs} CFs. Try lowering target_probas or increasing max_iter.")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()