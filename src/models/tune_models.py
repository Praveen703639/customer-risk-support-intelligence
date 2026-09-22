from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


RANDOM_STATE = 42
TEST_SIZE = 0.20


NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_dataset(data_path):

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = (
        df["Churn"]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    return X, y


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def main():

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [
                    200,
                    400,
                ],
                "classifier__max_depth": [
                    None,
                    6,
                    10,
                    15,
                ],
                "classifier__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        ),
    }

    all_results = []

    print()
    print("CROSS-VALIDATION + HYPERPARAMETER TUNING")
    print("=" * 70)

    for name, (classifier, parameter_grid) in models.items():

        print()
        print(f"Tuning: {name}")
        print("-" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
            return_train_score=False,
        )

        search.fit(
            X_train,
            y_train,
        )

        print()
        print("Best parameters:")
        print(search.best_params_)

        print(
            f"Best CV ROC-AUC: "
            f"{search.best_score_:.4f}"
        )

        test_metrics = evaluate_model(
            search.best_estimator_,
            X_test,
            y_test,
        )

        print()
        print("Untouched test-set evaluation:")

        for metric, value in test_metrics.items():
            print(
                f"{metric:12}: {value:.4f}"
            )

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        model_path = (
            models_dir
            / f"{safe_name}_tuned.joblib"
        )

        joblib.dump(
            search.best_estimator_,
            model_path,
        )

        print()
        print(f"Saved model:")
        print(model_path)

        all_results.append(
            {
                "model": name,
                "cv_roc_auc": search.best_score_,
                **test_metrics,
            }
        )

        cv_results = pd.DataFrame(
            search.cv_results_
        )

        cv_path = (
            reports_dir
            / f"{safe_name}_grid_search_results.csv"
        )

        cv_results.to_csv(
            cv_path,
            index=False,
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        reports_dir
        / "tuned_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL TUNED MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"Results saved to:")
    print(results_path)


if __name__ == "__main__":
    main()
