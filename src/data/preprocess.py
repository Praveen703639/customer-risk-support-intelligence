from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


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


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load the raw Telco customer churn dataset."""

    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform deterministic cleaning before model preprocessing.

    TotalCharges contains blank strings in the raw dataset.
    Convert them to numeric values; invalid values become NaN.
    """

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate customer IDs, features, and target.

    customerID is retained separately for customer-level
    tracing but is never used as an ML feature.
    """

    customer_ids = df[ID_COLUMN].copy()

    X = df.drop(
        columns=[ID_COLUMN, TARGET_COLUMN],
    )

    y = (
        df[TARGET_COLUMN]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    return X, y, customer_ids


def build_preprocessor() -> ColumnTransformer:
    """Build the numerical and categorical preprocessing pipeline."""

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
                    strategy="most_frequent",
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


def prepare_data(data_path: Path):
    """Load, clean, split, and build the preprocessing object."""

    df = load_raw_data(data_path)

    df = clean_data(df)

    X, y, customer_ids = split_features_target(df)

    X_train, X_test, y_train, y_test, ids_train, ids_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        ids_train,
        ids_test,
        preprocessor,
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        ids_train,
        ids_test,
        preprocessor,
    ) = prepare_data(data_path)

    preprocessor_path = (
        models_dir / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print("PREPROCESSING REPORT")
    print("=" * 40)

    print(f"Training samples : {X_train_processed.shape[0]}")
    print(f"Test samples     : {X_test_processed.shape[0]}")
    print(f"Processed features: {X_train_processed.shape[1]}")

    print()
    print("Training target distribution:")
    print(y_train.value_counts().sort_index())

    print()
    print("Test target distribution:")
    print(y_test.value_counts().sort_index())

    print()
    print(f"Preprocessor saved to:")
    print(preprocessor_path)
