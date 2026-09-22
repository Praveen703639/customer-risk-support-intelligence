from pathlib import Path

import json

import networkx as nx
import pandas as pd


RANDOM_STATE = 42


def load_customer_data(
    data_path: Path,
    risk_path: Path,
) -> pd.DataFrame:

    customers = pd.read_csv(
        data_path
    )

    risk_scores = pd.read_csv(
        risk_path
    )

    customers["TotalCharges"] = pd.to_numeric(
        customers["TotalCharges"],
        errors="coerce",
    )

    merged = customers.merge(
        risk_scores[
            [
                "customerID",
                "churn_probability",
                "risk_score",
                "risk_level",
                "predicted_churn",
            ]
        ],
        on="customerID",
        how="inner",
    )

    return merged


def add_customer_node(
    graph: nx.Graph,
    row: pd.Series,
) -> None:

    customer_id = row["customerID"]

    graph.add_node(
        f"customer:{customer_id}",
        node_type="customer",
        customer_id=customer_id,
        gender=row["gender"],
        senior_citizen=int(
            row["SeniorCitizen"]
        ),
        partner=row["Partner"],
        dependents=row["Dependents"],
        tenure=int(row["tenure"]),
        monthly_charges=float(
            row["MonthlyCharges"]
        ),
        total_charges=(
            ""
        ),
        contract=row["Contract"],
        churn=row["Churn"],
        churn_probability=float(
            row["churn_probability"]
        ),
        risk_score=float(
            row["risk_score"]
        ),
        risk_level=row["risk_level"],
        predicted_churn=int(
            row["predicted_churn"]
        ),
    )


def add_relationship(
    graph: nx.Graph,
    customer_id: str,
    relationship_type: str,
    value,
    node_type: str,
) -> None:

    value = str(value)

    node_id = (
        f"{node_type}:{value}"
    )

    graph.add_node(
        node_id,
        node_type=node_type,
        value=value,
    )

    graph.add_edge(
        f"customer:{customer_id}",
        node_id,
        relationship=relationship_type,
    )


def build_graph(
    customers: pd.DataFrame,
) -> nx.Graph:

    graph = nx.Graph()

    for _, row in customers.iterrows():

        customer_id = row["customerID"]

        add_customer_node(
            graph,
            row,
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_GENDER",
            row["gender"],
            "gender",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_CONTRACT",
            row["Contract"],
            "contract",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_INTERNET_SERVICE",
            row["InternetService"],
            "internet_service",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_PAYMENT_METHOD",
            row["PaymentMethod"],
            "payment_method",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_PHONE_SERVICE",
            row["PhoneService"],
            "phone_service",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_TECH_SUPPORT",
            row["TechSupport"],
            "tech_support",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_ONLINE_SECURITY",
            row["OnlineSecurity"],
            "online_security",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_ONLINE_BACKUP",
            row["OnlineBackup"],
            "online_backup",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_DEVICE_PROTECTION",
            row["DeviceProtection"],
            "device_protection",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_STREAMING_TV",
            row["StreamingTV"],
            "streaming_tv",
        )

        add_relationship(
            graph,
            customer_id,
            "HAS_STREAMING_MOVIES",
            row["StreamingMovies"],
            "streaming_movies",
        )

    return graph


def calculate_graph_statistics(
    graph: nx.Graph,
) -> dict:

    node_type_counts = {}

    for _, attributes in graph.nodes(
        data=True
    ):

        node_type = attributes.get(
            "node_type",
            "unknown",
        )

        node_type_counts[node_type] = (
            node_type_counts.get(
                node_type,
                0,
            )
            + 1
        )

    relationship_counts = {}

    for _, _, attributes in graph.edges(
        data=True
    ):

        relationship = attributes.get(
            "relationship",
            "unknown",
        )

        relationship_counts[
            relationship
        ] = (
            relationship_counts.get(
                relationship,
                0,
            )
            + 1
        )

    degree_centrality = (
        nx.degree_centrality(graph)
    )

    top_nodes = sorted(
        degree_centrality.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_type_counts": node_type_counts,
        "relationship_counts": relationship_counts,
        "top_degree_centrality_nodes": [
            {
                "node": node,
                "centrality": score,
                "node_type": graph.nodes[
                    node
                ].get(
                    "node_type"
                ),
            }
            for node, score in top_nodes
        ],
    }


def save_graph(
    graph: nx.Graph,
    output_path: Path,
) -> None:

    for _, attributes in graph.nodes(data=True):

        for key, value in list(attributes.items()):

            if value is None:
                attributes[key] = ""

            elif pd.isna(value):
                attributes[key] = ""

            elif hasattr(value, "item"):
                attributes[key] = value.item()

            else:
                attributes[key] = value

    for _, _, attributes in graph.edges(data=True):

        for key, value in list(attributes.items()):

            if value is None:
                attributes[key] = ""

            elif pd.isna(value):
                attributes[key] = ""

            elif hasattr(value, "item"):
                attributes[key] = value.item()

            else:
                attributes[key] = value

    nx.write_graphml(
        graph,
        output_path,
    )

def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    data_path = (
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

    graph_dir = (
        project_root
        / "reports"
        / "graph"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_path = (
        graph_dir
        / "customer_service_graph.graphml"
    )

    statistics_path = (
        graph_dir
        / "graph_statistics.json"
    )

    print()
    print(
        "CUSTOMER-SERVICE KNOWLEDGE GRAPH"
    )
    print("=" * 70)

    customers = load_customer_data(
        data_path,
        risk_path,
    )

    print()
    print(
        f"Customers loaded: "
        f"{len(customers)}"
    )

    graph = build_graph(
        customers
    )

    statistics = (
        calculate_graph_statistics(
            graph
        )
    )

    save_graph(
        graph,
        graph_path,
    )

    with open(
        statistics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4,
        )

    print()
    print("GRAPH STATISTICS")
    print("-" * 70)

    print(
        f"Nodes : "
        f"{statistics['nodes']}"
    )

    print(
        f"Edges : "
        f"{statistics['edges']}"
    )

    print()
    print("NODE TYPES")

    for node_type, count in (
        statistics[
            "node_type_counts"
        ].items()
    ):

        print(
            f"{node_type:25}: "
            f"{count}"
        )

    print()
    print("RELATIONSHIP TYPES")

    for relationship, count in (
        statistics[
            "relationship_counts"
        ].items()
    ):

        print(
            f"{relationship:30}: "
            f"{count}"
        )

    print()
    print(
        "TOP DEGREE-CENTRALITY NODES"
    )
    print("-" * 70)

    for item in statistics[
        "top_degree_centrality_nodes"
    ]:

        print(
            f"{item['node']:45} "
            f"{item['centrality']:.6f}"
        )

    print()
    print("FILES CREATED")
    print("-" * 70)

    print(graph_path)
    print(statistics_path)


if __name__ == "__main__":
    main()


