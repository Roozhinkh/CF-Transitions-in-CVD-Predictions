from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def train_one_target(df: pd.DataFrame, target_col: str, out_dir: Path) -> None:
    """
    Train a Random Forest classifier for a single target column and save the model.
    """

    outcome_cols = ["hltprhb", "hltprhc", "hltprdi"]
    if target_col not in outcome_cols:
        raise ValueError(f"Unknown target '{target_col}'. Expected one of: {outcome_cols}")

    feature_cols = [c for c in df.columns if c not in outcome_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    uniq = sorted(y.dropna().unique())
    print(f"\n==============================")
    print(f"Training target: {target_col}")
    print(f"Unique values in target: {uniq}")
    print("Target distribution (normalized):")
    print(y.value_counts(normalize=True))
    print("==============================\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    roc_auc = None
    if hasattr(rf_model, "predict_proba") and len(rf_model.classes_) == 2:
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model performance on test set:")
    print(f"Accuracy: {accuracy:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC:  {roc_auc:.3f}")
    else:
        print("ROC-AUC:  not computed (target is not binary in the expected way)")
    print()

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"rf_{target_col}.pkl"
    joblib.dump(rf_model, model_path)
    print(f"Saved trained model to: {model_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "ess_model_ready.csv"
    models_dir = project_root / "models"

    df = pd.read_csv(data_path)
    print("Loaded dataset:", data_path)
    print("Shape:", df.shape)

    targets = ["hltprhb", "hltprhc", "hltprdi"]
    for t in targets:
        train_one_target(df, t, models_dir)

    print("\n Training completed for all targets.")


if __name__ == "__main__":
    main()
