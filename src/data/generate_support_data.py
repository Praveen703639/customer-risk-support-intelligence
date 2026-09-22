from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


ISSUE_TYPES = [
    "Billing",
    "Internet Connectivity",
    "Technical Support",
    "Account Access",
    "Service Change",
    "Payment Issue",
    "Equipment",
    "Streaming",
]

CHANNELS = [
    "Phone",
    "Web",
    "Chat",
    "Email",
]

STATUSES = [
    "Open",
    "In Progress",
    "Resolved",
    "Closed",
]

RESOLUTIONS = [
    "Issue resolved remotely",
    "Customer provided troubleshooting steps",
    "Billing adjustment applied",
    "Payment method updated",
    "Service configuration updated",
    "Replacement equipment requested",
    "Account information updated",
    "Escalated to specialist",
]


def load_customer_data(
    project_root: Path,
) -> pd.DataFrame:

    customer_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    risk_path = (
        project_root
        / "reports"
        / "customer_risk_scores.csv"
    )

    customers = pd.read_csv(
        customer_path
    )

    risk_scores = pd.read_csv(
        risk_path
    )

    customers["TotalCharges"] = pd.to_numeric(
        customers["TotalCharges"],
        errors="coerce",
    )

    customer_data = customers.merge(
        risk_scores[
            [
                "customerID",
                "churn_probability",
                "risk_score",
                "risk_level",
            ]
        ],
        on="customerID",
        how="inner",
    )

    return customer_data


def choose_issue_type(
    row: pd.Series,
    rng: np.random.Generator,
) -> str:

    weighted_issues = [
        "Billing",
        "Internet Connectivity",
        "Technical Support",
        "Account Access",
        "Service Change",
        "Payment Issue",
        "Equipment",
        "Streaming",
    ]

    weights = np.array(
        [
            0.15,
            0.20,
            0.20,
            0.08,
            0.10,
            0.10,
            0.10,
            0.07,
        ]
    )

    # Increase technical/support-related interactions
    # for customers whose profile contains relevant services.
    if row["InternetService"] != "No":
        weights[1] += 0.05

    if row["TechSupport"] == "No":
        weights[2] += 0.04

    if row["PaymentMethod"] == "Electronic check":
        weights[0] += 0.04
        weights[5] += 0.04

    if row["StreamingTV"] == "Yes" or row["StreamingMovies"] == "Yes":
        weights[7] += 0.04

    weights = weights / weights.sum()

    return rng.choice(
        weighted_issues,
        p=weights,
    )


def choose_priority(
    risk_level: str,
    rng: np.random.Generator,
) -> str:

    if risk_level == "Critical":

        return rng.choice(
            [
                "High",
                "Urgent",
            ],
            p=[
                0.55,
                0.45,
            ],
        )

    if risk_level == "High":

        return rng.choice(
            [
                "Medium",
                "High",
            ],
            p=[
                0.45,
                0.55,
            ],
        )

    if risk_level == "Medium":

        return rng.choice(
            [
                "Low",
                "Medium",
            ],
            p=[
                0.45,
                0.55,
            ],
        )

    return rng.choice(
        [
            "Low",
            "Medium",
        ],
        p=[
            0.75,
            0.25,
        ],
    )


def choose_status(
    rng: np.random.Generator,
) -> str:

    return rng.choice(
        STATUSES,
        p=[
            0.12,
            0.18,
            0.38,
            0.32,
        ],
    )


def generate_ticket(
    row: pd.Series,
    ticket_number: int,
    rng: np.random.Generator,
) -> dict:

    issue_type = choose_issue_type(
        row,
        rng,
    )

    priority = choose_priority(
        row["risk_level"],
        rng,
    )

    status = choose_status(
        rng,
    )

    channel = rng.choice(
        CHANNELS,
        p=[
            0.35,
            0.25,
            0.25,
            0.15,
        ],
    )

    created_days_ago = int(
        rng.integers(
            1,
            365,
        )
    )

    # Higher-priority tickets tend to have
    # shorter intended resolution targets.
    if priority == "Urgent":
        resolution_hours = round(
            float(
                rng.uniform(
                    1,
                    12,
                )
            ),
            2,
        )

    elif priority == "High":
        resolution_hours = round(
            float(
                rng.uniform(
                    4,
                    36,
                )
            ),
            2,
        )

    elif priority == "Medium":
        resolution_hours = round(
            float(
                rng.uniform(
                    8,
                    72,
                )
            ),
            2,
        )

    else:
        resolution_hours = round(
            float(
                rng.uniform(
                    12,
                    120,
                )
            ),
            2,
        )

    resolution = rng.choice(
        RESOLUTIONS
    )

    # Open and in-progress tickets do not yet
    # have a completed resolution.
    if status in {
        "Open",
        "In Progress",
    }:

        resolution = "Pending"

    return {
        "ticket_id": (
            f"TKT-{ticket_number:06d}"
        ),
        "customerID": row[
            "customerID"
        ],
        "issue_type": issue_type,
        "priority": priority,
        "channel": channel,
        "status": status,
        "created_days_ago": (
            created_days_ago
        ),
        "resolution_hours": (
            resolution_hours
        ),
        "resolution": resolution,
    }


def generate_support_tickets(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:

    tickets = []

    ticket_number = 1

    for _, row in customers.iterrows():

        # Every customer gets at least one
        # synthetic interaction.
        ticket_count = int(
            rng.choice(
                [
                    1,
                    2,
                    3,
                ],
                p=[
                    0.65,
                    0.25,
                    0.10,
                ],
            )
        )

        # Higher-risk customers receive somewhat
        # more synthetic support interactions.
        if row["risk_level"] == "Critical":
            ticket_count += 2

        elif row["risk_level"] == "High":
            ticket_count += 1

        for _ in range(
            ticket_count
        ):

            tickets.append(
                generate_ticket(
                    row,
                    ticket_number,
                    rng,
                )
            )

            ticket_number += 1

    return pd.DataFrame(
        tickets
    )


def validate_tickets(
    tickets: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:

    assert len(tickets) > 0

    assert (
        tickets["ticket_id"]
        .is_unique
    )

    valid_customer_ids = set(
        customers["customerID"]
    )

    assert set(
        tickets["customerID"]
    ).issubset(
        valid_customer_ids
    )

    assert (
        tickets["issue_type"]
        .isin(ISSUE_TYPES)
        .all()
    )

    assert (
        tickets["channel"]
        .isin(CHANNELS)
        .all()
    )

    assert (
        tickets["status"]
        .isin(STATUSES)
        .all()
    )

    assert (
        tickets["created_days_ago"]
        > 0
    ).all()

    assert (
        tickets["resolution_hours"]
        > 0
    ).all()


def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    output_dir = (
        project_root
        / "data"
        / "raw"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "support_tickets.csv"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    customers = load_customer_data(
        project_root
    )

    print()
    print(
        "SYNTHETIC SUPPORT DATA GENERATION"
    )
    print("=" * 70)

    print(
        f"Customers used: "
        f"{len(customers)}"
    )

    tickets = generate_support_tickets(
        customers,
        rng,
    )

    validate_tickets(
        tickets,
        customers,
    )

    tickets.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Support tickets generated: "
        f"{len(tickets)}"
    )

    print()
    print(
        "ISSUE DISTRIBUTION"
    )
    print("-" * 70)

    print(
        tickets[
            "issue_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "PRIORITY DISTRIBUTION"
    )
    print("-" * 70)

    print(
        tickets[
            "priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "STATUS DISTRIBUTION"
    )
    print("-" * 70)

    print(
        tickets[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "FILES CREATED"
    )
    print("-" * 70)

    print(output_path)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These support tickets are synthetic "
        "and are generated for project "
        "development and demonstration."
    )


if __name__ == "__main__":
    main()
