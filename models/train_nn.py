import os
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

TASK_CONFIG = {
    "bp": "hltprhb",
    "hd": "hltprhc",
    "dm": "hltprdi",
}

DATA_PATH = "data/ess_model_ready.csv"
OUT_DIR = "models"
RANDOM_STATE = 42

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TASK_CONFIG.values()]

def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bp", "hd", "dm", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    feature_cols = get_feature_cols(df)

    tasks = ["bp", "hd", "dm"] if args.task == "all" else [args.task]

    for task in tasks:
        target = TASK_CONFIG[task]
        print(f"\n=== Training NN for task={task} target={target} ===")

        X = df[feature_cols].values.astype(np.float32)
        y = df[target].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_model(input_dim=X_train_s.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
        ]

        model.fit(
            X_train_s, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=256,
            verbose=1,
            callbacks=callbacks
        )

        y_pred = model.predict(X_test_s, verbose=0).reshape(-1)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))
        print(f"Test AUC: {auc:.4f} | Test ACC: {acc:.4f}")

        model_path = os.path.join(OUT_DIR, f"nn_{target}.keras")
        scaler_path = os.path.join(OUT_DIR, f"scaler_nn_{target}.pkl")
        meta_path = os.path.join(OUT_DIR, f"nn_{target}_meta.pkl")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump({"feature_cols": feature_cols}, meta_path)

        print(f"Saved: {model_path}")
        print(f"Saved: {scaler_path}")
        print(f"Saved: {meta_path}")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()