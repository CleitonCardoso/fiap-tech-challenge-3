from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "DEP_HOUR",
    "DISTANCE",
    "IS_WEEKEND",
]

CATEGORICAL = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUMERICAL = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR", "DISTANCE", "IS_WEEKEND"]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "processed" / "flights_sample.parquet"
    out_path = repo_root / "models" / "random_forest_delay_model.pkl"
    out_path.parent.mkdir(exist_ok=True)

    data = pd.read_parquet(data_path)
    X = data[FEATURES]
    y = data["DELAYED"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), NUMERICAL),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150, max_depth=18, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC: {auc:.3f}")

    joblib.dump(model, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
