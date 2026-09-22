from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate one classification model."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
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

    return model, metrics, predictions


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

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    X = df.drop(
        columns=["customerID", "Churn"],
    )

    y = (
        df["Churn"]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    from sklearn.model_selection import train_test_split
    from src.data.preprocess import build_preprocessor

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    print()
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)

    for name, model in models.items():

        trained_model, metrics, predictions = (
            evaluate_model(
                model,
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
            )
        )

        print()
        print(name)
        print("-" * 40)

        for metric_name, value in metrics.items():
            print(
                f"{metric_name:12}: {value:.4f}"
            )

        print()
        print("Confusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions,
            )
        )

        print()
        print("Classification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "No Churn",
                    "Churn",
                ],
                zero_division=0,
            )
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

        model_filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            trained_model,
            models_dir / model_filename,
        )

    results_df = pd.DataFrame(results)

    results_path = (
        reports_dir
        / "baseline_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
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
