"""
Model Building Script with Experiment Tracking
------------------------------------------------
Loads the train/test split produced by data_prep.py (delivered to this
job as a workflow artifact in the GitHub Actions pipeline), builds a
preprocessing + model pipeline for several candidate algorithms, tunes
each with GridSearchCV, logs every run and its parameters with MLflow,
evaluates on the held-out test set, and commits the best-performing
model to the repository as model/best_model.joblib.
"""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
MLRUNS_DIR = ROOT / "mlruns"

CANDIDATE_MODELS = {
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_split": [2, 5, 10],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [8, 12, None],
            "model__min_samples_split": [2, 5],
        },
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1, 1.0],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [150, 250],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
        },
    },
}


def load_splits():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def run_experiments():
    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor(X_train)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("visit-with-us-wellness-package")

    results = []
    best_score = -1
    best_model = None
    best_name = None
    best_metrics = None

    for name, cfg in CANDIDATE_MODELS.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", cfg["estimator"])])
        search = GridSearchCV(pipeline, param_grid=cfg["params"], scoring="f1", cv=5, n_jobs=-1)

        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            metrics = evaluate(search.best_estimator_, X_test, y_test)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_family", name)

            results.append({"model": name, "best_params": search.best_params_, **metrics})

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = search.best_estimator_
                best_name = name
                best_metrics = metrics

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(results_df.to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    with open(MODEL_DIR / "best_model_metadata.json", "w") as f:
        json.dump({"model_name": best_name, "metrics": best_metrics}, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Metrics   : {best_metrics}")
    print(f"Saved to  : {MODEL_DIR / 'best_model.joblib'}")

    return results_df, best_name, best_metrics


if __name__ == "__main__":
    run_experiments()
