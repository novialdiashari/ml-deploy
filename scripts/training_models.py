import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# Lokasi file
DATA_FILE = Path("data/diabetes_regression.csv")

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "best_regression_model.pkl"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
FEATURES_FILE = MODEL_DIR / "feature_names.json"

RANDOM_STATE = 42


def load_dataset():
    """Membaca dataset dari file CSV."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_FILE}"
        )

    with DATA_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        # Mengambil nama fitur
        feature_names = [
            name
            for name in reader.fieldnames
            if name != "target"
        ]

        # Membaca semua baris
        rows = list(reader)

    # Mengambil data fitur
    X = np.array([
        [
            float(row[feature])
            for feature in feature_names
        ]
        for row in rows
    ])

    # Mengambil target
    y = np.array([
        float(row["target"])
        for row in rows
    ])

    return X, y, feature_names


def build_models():
    """Membuat beberapa model regresi."""

    models = {

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=30,
                epsilon=8
            ))
        ]),

        "decision_tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "xgboost": XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE
        )
    }

    return models


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Melatih dan mengevaluasi model."""

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    predictions = model.predict(X_test)

    # Evaluasi
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "model": name,
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4)
    }


def main():

    # Membaca dataset
    X, y, feature_names = load_dataset()

    print("Dataset berhasil dibaca")
    print(f"Jumlah data  : {len(y)}")
    print(f"Jumlah fitur : {len(feature_names)}")

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print(f"Data training : {len(X_train)}")
    print(f"Data testing  : {len(X_test)}")

    # Membuat model
    models = build_models()

    print()
    print("Model yang digunakan:")

    for name in models:
        print(f"- {name}")

    # Menyimpan hasil evaluasi
    results = []

    # Menyimpan model terbaik
    best_model = None
    best_model_name = None
    best_rmse = float("inf")

    # Melatih dan membandingkan model
    print()
    print("Hasil evaluasi:")

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

        print(
            f"{name}: "
            f"RMSE={result['rmse']}, "
            f"MAE={result['mae']}, "
            f"R2={result['r2']}"
        )

        # Model dengan RMSE terkecil dipilih
        if result["rmse"] < best_rmse:
            best_rmse = result["rmse"]
            best_model_name = name
            best_model = model

    # Informasi hasil eksperimen
    metadata = {
        "dataset": "Diabetes Regression Dataset",
        "task": "regression",
        "feature_names": feature_names,
        "target": "diabetes disease progression score",
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "selection_metric": "lowest_rmse",
        "selected_model": best_model_name,
        "results": sorted(
            results,
            key=lambda item: item["rmse"]
        )
    }

    # Membuat folder models
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Menyimpan model terbaik
    joblib.dump(
        best_model,
        MODEL_FILE
    )

    # Menyimpan hasil evaluasi
    METRICS_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # Menyimpan nama fitur
    FEATURES_FILE.write_text(
        json.dumps(
            feature_names,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("Training selesai")
    print(f"Model terbaik : {best_model_name}")
    print(f"RMSE terbaik  : {best_rmse}")
    print(f"Model disimpan: {MODEL_FILE}")


if __name__ == "__main__":
    main()