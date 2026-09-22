from pathlib import Path

import json

import networkx as nx
import pandas as pd


def sanitize_graph_attributes(
    graph: nx.DiGraph,
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


def load_base_graph(
    graph_path: Path,
) -> nx.Graph:

    return nx.read_graphml(
        graph_path
    )


def load_support_tickets(
    support_path: Path,
) -> pd.DataFrame:

    tickets = pd.read_csv(
        support_path
    )

    required_columns = {
        "ticket_id",
        "customerID",
        "issue_type",
        "priority",
        "channel",
        "status",
        "created_days_ago",
        "resolution_hours",
        "resolution",
    }

    missing_columns = (
        required_columns
        - set(tickets.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing support-ticket columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return tickets


def build_support_graph(
    base_graph: nx.Graph,
    tickets: pd.DataFrame,
) -> nx.DiGraph:

    graph = nx.DiGraph()

    # Copy all existing customer/service nodes.
    for node, attributes in (
        base_graph.nodes(data=True)
    ):

        graph.add_node(
            node,
            **dict(attributes),
        )

    # Copy the existing service relationships.
    for source, target, attributes in (
        base_graph.edges(data=True)
    ):

        graph.add_edge(
            source,
            target,
            **dict(attributes),
        )

    for _, ticket in tickets.iterrows():

        customer_id = str(
            ticket["customerID"]
        )

        ticket_id = str(
            ticket["ticket_id"]
        )

        customer_node = (
            f"customer:{customer_id}"
        )

        ticket_node = (
            f"ticket:{ticket_id}"
        )

        if customer_node not in graph:

            print(
                f"WARNING: customer not found: "
                f"{customer_id}"
            )

            continue

        graph.add_node(
            ticket_node,
            node_type="support_ticket",
            ticket_id=ticket_id,
            customer_id=customer_id,
            issue_type=str(
                ticket["issue_type"]
            ),
            priority=str(
                ticket["priority"]
            ),
            channel=str(
                ticket["channel"]
            ),
            status=str(
                ticket["status"]
            ),
            created_days_ago=int(
                ticket["created_days_ago"]
            ),
            resolution_hours=float(
                ticket["resolution_hours"]
            ),
            resolution=str(
                ticket["resolution"]
            ),
        )

        graph.add_edge(
            customer_node,
            ticket_node,
            relationship="CREATED_TICKET",
        )

        issue_node = (
            "issue:"
            + str(ticket["issue_type"])
        )

        graph.add_node(
            issue_node,
            node_type="issue",
            value=str(
                ticket["issue_type"]
            ),
        )

        graph.add_edge(
            ticket_node,
            issue_node,
            relationship="HAS_ISSUE",
        )

        priority_node = (
            "priority:"
            + str(ticket["priority"])
        )

        graph.add_node(
            priority_node,
            node_type="priority",
            value=str(
                ticket["priority"]
            ),
        )

        graph.add_edge(
            ticket_node,
            priority_node,
            relationship="HAS_PRIORITY",
        )

        channel_node = (
            "channel:"
            + str(ticket["channel"])
        )

        graph.add_node(
            channel_node,
            node_type="channel",
            value=str(
                ticket["channel"]
            ),
        )

        graph.add_edge(
            ticket_node,
            channel_node,
            relationship="VIA_CHANNEL",
        )

        status_node = (
            "status:"
            + str(ticket["status"])
        )

        graph.add_node(
            status_node,
            node_type="ticket_status",
            value=str(
                ticket["status"]
            ),
        )

        graph.add_edge(
            ticket_node,
            status_node,
            relationship="HAS_STATUS",
        )

        resolution_node = (
            "resolution:"
            + str(ticket["resolution"])
        )

        graph.add_node(
            resolution_node,
            node_type="resolution",
            value=str(
                ticket["resolution"]
            ),
        )

        graph.add_edge(
            ticket_node,
            resolution_node,
            relationship="HAS_RESOLUTION",
        )

    return graph


def calculate_statistics(
    graph: nx.DiGraph,
) -> dict:

    node_type_counts = {}

    for _, attributes in (
        graph.nodes(data=True)
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

    for _, _, attributes in (
        graph.edges(data=True)
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

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_type_counts": node_type_counts,
        "relationship_counts": relationship_counts,
        "is_directed": graph.is_directed(),
    }


def save_graph(
    graph: nx.DiGraph,
    output_path: Path,
) -> None:

    sanitize_graph_attributes(
        graph
    )

    nx.write_graphml(
        graph,
        output_path,
    )


def main():

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    base_graph_path = (
        project_root
        / "reports"
        / "graph"
        / "customer_service_graph.graphml"
    )

    support_path = (
        project_root
        / "data"
        / "raw"
        / "support_tickets.csv"
    )

    output_dir = (
        project_root
        / "reports"
        / "graph"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_graph_path = (
        output_dir
        / "customer_support_knowledge_graph.graphml"
    )

    statistics_path = (
        output_dir
        / "customer_support_graph_statistics.json"
    )

    print()
    print(
        "CUSTOMER SUPPORT KNOWLEDGE GRAPH"
    )
    print("=" * 70)

    base_graph = load_base_graph(
        base_graph_path
    )

    tickets = load_support_tickets(
        support_path
    )

    print(
        f"Base graph nodes: "
        f"{base_graph.number_of_nodes()}"
    )

    print(
        f"Base graph edges: "
        f"{base_graph.number_of_edges()}"
    )

    print(
        f"Support tickets: "
        f"{len(tickets)}"
    )

    graph = build_support_graph(
        base_graph,
        tickets,
    )

    statistics = calculate_statistics(
        graph
    )

    save_graph(
        graph,
        output_graph_path,
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
    print(
        "EXPANDED GRAPH STATISTICS"
    )
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
        "FILES CREATED"
    )
    print("-" * 70)

    print(output_graph_path)
    print(statistics_path)


if __name__ == "__main__":
    main()
