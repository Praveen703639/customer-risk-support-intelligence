from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset schema does not match expected columns."
        )


def validate_customer_ids(df: pd.DataFrame) -> None:
    if df["customerID"].isna().any():
        raise ValueError("customerID contains missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")


def validate_target(df: pd.DataFrame) -> None:
    allowed_values = {"Yes", "No"}
    actual_values = set(df["Churn"].dropna().unique())

    unexpected = actual_values - allowed_values

    if unexpected:
        raise ValueError(
            f"Unexpected Churn values found: {unexpected}"
        )


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    if (df["tenure"] < 0).any():
        raise ValueError("Negative tenure values detected.")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("Negative MonthlyCharges detected.")

    total_charges = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    if (total_charges < 0).any():
        raise ValueError("Negative TotalCharges detected.")


def validate_total_charges(df: pd.DataFrame) -> int:
    total_charges = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return int(total_charges.isna().sum())


def validate_dataset(df: pd.DataFrame) -> dict:
    validate_schema(df)
    validate_customer_ids(df)
    validate_target(df)
    validate_numeric_ranges(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(
            df["customerID"].duplicated().sum()
        ),
        "invalid_total_charges": validate_total_charges(df),
        "churn_yes": int((df["Churn"] == "Yes").sum()),
        "churn_no": int((df["Churn"] == "No").sum()),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(data_path)

    report = validate_dataset(df)

    print("DATA VALIDATION REPORT")
    print("=" * 40)

    for key, value in report.items():
        print(f"{key}: {value}")
