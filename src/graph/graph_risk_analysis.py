from pathlib import Path

import networkx as nx
import pandas as pd


def load_graph(graph_path: Path) -> nx.Graph:
    return nx.read_graphml(graph_path)


def get_customer_nodes(graph: nx.Graph):
    return [
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("node_type") == "customer"
    ]


def analyze_risk_by_attribute(
    graph: nx.Graph,
) -> pd.DataFrame:

    rows = []

    for node, attributes in graph.nodes(data=True):

        if attributes.get("node_type") != "customer":
            continue

        risk_level = attributes.get(
            "risk_level",
            "",
        )

        risk_score = float(
            attributes.get(
                "risk_score",
                0,
            )
        )

        for neighbor in graph.neighbors(node):

            neighbor_attributes = graph.nodes[
                neighbor
            ]

            node_type = neighbor_attributes.get(
                "node_type"
            )

            if node_type in {
                "customer",
            }:
                continue

            rows.append(
                {
                    "customer_id": attributes.get(
                        "customer_id"
                    ),
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "attribute_type": node_type,
                    "attribute_value": neighbor_attributes.get(
                        "value"
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_attribute_risk(
    relationships: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        relationships
        .groupby(
            [
                "attribute_type",
                "attribute_value",
            ]
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),
            average_risk_score=(
                "risk_score",
                "mean",
            ),
            high_risk_customers=(
                "risk_level",
                lambda values: (
                    values.isin(
                        [
                            "High",
                            "Critical",
                        ]
                    ).sum()
                ),
            ),
            critical_customers=(
                "risk_level",
                lambda values: (
                    (
                        values
                        == "Critical"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    summary[
        "high_risk_rate"
    ] = (
        summary[
            "high_risk_customers"
        ]
        / summary["customers"]
    )

    summary[
        "critical_rate"
    ] = (
        summary[
            "critical_customers"
        ]
        / summary["customers"]
    )

    return summary.sort_values(
        "average_risk_score",
        ascending=False,
    )


def get_customer_neighborhood(
    graph: nx.Graph,
    customer_id: str,
) -> dict:

    customer_node = (
        f"customer:{customer_id}"
    )

    if customer_node not in graph:
        raise ValueError(
            f"Customer {customer_id} "
            f"not found in graph."
        )

    customer_attributes = (
        graph.nodes[
            customer_node
        ]
    )

    connected_attributes = []

    for neighbor in graph.neighbors(
        customer_node
    ):

        attributes = graph.nodes[
            neighbor
        ]

        connected_attributes.append(
            {
                "node": neighbor,
                "node_type": attributes.get(
                    "node_type"
                ),
                "value": attributes.get(
                    "value"
                ),
                "relationship": graph[
                    customer_node
                ][neighbor].get(
                    "relationship"
                ),
            }
        )

    return {
        "customer": customer_id,
        "risk_score": float(
            customer_attributes.get(
                "risk_score",
                0,
            )
        ),
        "risk_level": customer_attributes.get(
            "risk_level"
        ),
        "churn_probability": float(
            customer_attributes.get(
                "churn_probability",
                0,
            )
        ),
        "connected_attributes": (
            connected_attributes
        ),
    }


def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    graph_path = (
        project_root
        / "reports"
        / "graph"
        / "customer_service_graph.graphml"
    )

    reports_dir = (
        project_root
        / "reports"
        / "graph"
    )

    graph = load_graph(
        graph_path
    )

    print()
    print(
        "GRAPH-BASED CUSTOMER RISK ANALYSIS"
    )
    print("=" * 70)

    customer_nodes = (
        get_customer_nodes(graph)
    )

    print()
    print(
        f"Customer nodes: "
        f"{len(customer_nodes)}"
    )

    relationships = (
        analyze_risk_by_attribute(
            graph
        )
    )

    summary = (
        summarize_attribute_risk(
            relationships
        )
    )

    summary_path = (
        reports_dir
        / "attribute_risk_analysis.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "HIGHEST-RISK ATTRIBUTE GROUPS"
    )
    print("-" * 70)

    print(
        summary[
            [
                "attribute_type",
                "attribute_value",
                "customers",
                "average_risk_score",
                "high_risk_rate",
                "critical_rate",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Find the highest-risk customer
    highest_risk_customer = max(
        customer_nodes,
        key=lambda node: float(
            graph.nodes[node].get(
                "risk_score",
                0,
            )
        ),
    )

    customer_id = graph.nodes[
        highest_risk_customer
    ].get(
        "customer_id"
    )

    neighborhood = (
        get_customer_neighborhood(
            graph,
            customer_id,
        )
    )

    print()
    print(
        "HIGHEST-RISK CUSTOMER NEIGHBORHOOD"
    )
    print("-" * 70)

    print(
        f"Customer: "
        f"{neighborhood['customer']}"
    )

    print(
        f"Risk score: "
        f"{neighborhood['risk_score']:.2f}"
    )

    print(
        f"Risk level: "
        f"{neighborhood['risk_level']}"
    )

    print(
        f"Churn probability: "
        f"{neighborhood['churn_probability']:.4f}"
    )

    print()
    print("CONNECTED ATTRIBUTES:")

    for item in neighborhood[
        "connected_attributes"
    ]:

        print(
            f"{item['node_type']:25} "
            f"{item['value']}"
        )

    print()
    print(
        "FILE CREATED:"
    )

    print(summary_path)


if __name__ == "__main__":
    main()
