from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and clean the Telco dataset."""

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df


def prepare_test_set(df: pd.DataFrame):
    """
    Reproduce the same stratified test split used during
    model tuning so that the saved model is evaluated on
    the same untouched test population.
    """

    from sklearn.model_selection import train_test_split

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = (
        df["Churn"]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    customer_ids = df["customerID"].copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        ids_train,
        ids_test,
    ) = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_test,
        y_test,
        ids_test,
    )


def calculate_threshold_metrics(
    y_true,
    probabilities,
    thresholds,
):
    """Calculate classification metrics at selected thresholds."""

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def assign_risk_level(probability: float) -> str:
    """
    Convert churn probability into an application-level
    risk category.

    These categories are project-defined presentation
    labels, not claims of calibrated probability bands.
    """

    if probability < 0.25:
        return "Low"

    if probability < 0.50:
        return "Medium"

    if probability < 0.75:
        return "High"

    return "Critical"


def create_customer_risk_table(
    customer_ids,
    y_true,
    probabilities,
):
    """Create customer-level risk records."""

    risk_scores = (
        probabilities * 100
    )

    risk_levels = [
        assign_risk_level(
            probability
        )
        for probability in probabilities
    ]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    risk_df = pd.DataFrame(
        {
            "customerID": customer_ids.values,
            "actual_churn": y_true.values,
            "churn_probability": probabilities,
            "risk_score": risk_scores,
            "risk_level": risk_levels,
            "predicted_churn": predictions,
        }
    )

    risk_df = risk_df.sort_values(
        "churn_probability",
        ascending=False,
    )

    return risk_df


def main():

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    model_path = (
        project_root
        / "models"
        / "logistic_regression_tuned.joblib"
    )

    reports_dir = (
        project_root
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("CUSTOMER RISK SCORING")
    print("=" * 70)

    print()
    print("Loading model:")
    print(model_path)

    model = joblib.load(
        model_path
    )

    df = load_data(
        data_path
    )

    (
        X_test,
        y_test,
        customer_ids,
    ) = prepare_test_set(df)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print(
        f"Test ROC-AUC: {roc_auc:.4f}"
    )

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    threshold_results = (
        calculate_threshold_metrics(
            y_test,
            probabilities,
            thresholds,
        )
    )

    threshold_path = (
        reports_dir
        / "risk_threshold_analysis.csv"
    )

    threshold_results.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print("THRESHOLD ANALYSIS")
    print("-" * 70)

    print(
        threshold_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    risk_df = create_customer_risk_table(
        customer_ids,
        y_test,
        probabilities,
    )

    risk_path = (
        reports_dir
        / "customer_risk_scores.csv"
    )

    risk_df.to_csv(
        risk_path,
        index=False,
    )

    print()
    print("CUSTOMER RISK SUMMARY")
    print("-" * 70)

    print(
        risk_df[
            "risk_level"
        ].value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("TOP 20 HIGHEST-RISK CUSTOMERS")
    print("-" * 70)

    print(
        risk_df[
            [
                "customerID",
                "churn_probability",
                "risk_score",
                "risk_level",
                "actual_churn",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"Threshold analysis:"
    )
    print(threshold_path)

    print()
    print(
        f"Customer risk scores:"
    )
    print(risk_path)


if __name__ == "__main__":
    main()
