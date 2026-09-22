from pathlib import Path

import networkx as nx


class CustomerGraphRetriever:

    def __init__(
        self,
        graph_path: Path,
    ) -> None:

        self.graph_path = graph_path

        self.graph = nx.read_graphml(
            graph_path
        )

    def get_customer_node(
        self,
        customer_id: str,
    ) -> str:

        customer_node = (
            f"customer:{customer_id}"
        )

        if customer_node not in self.graph:

            raise ValueError(
                f"Customer not found: "
                f"{customer_id}"
            )

        return customer_node

    def get_customer_services(
        self,
        customer_id: str,
    ) -> list[dict]:

        customer_node = (
            self.get_customer_node(
                customer_id
            )
        )

        services = []

        # Check outgoing relationships.
        for neighbor in (
            self.graph.successors(
                customer_node
            )
        ):

            edge_data = self.graph.edges[
                customer_node,
                neighbor
            ]

            relationship = edge_data.get(
                "relationship",
                "",
            )

            if relationship.startswith(
                "HAS_"
            ):

                services.append(
                    {
                        "node": neighbor,
                        "relationship": relationship,
                        "direction": "outgoing",
                        "attributes": dict(
                            self.graph.nodes[
                                neighbor
                            ]
                        ),
                    }
                )

        # Check incoming relationships.
        for neighbor in (
            self.graph.predecessors(
                customer_node
            )
        ):

            edge_data = self.graph.edges[
                neighbor,
                customer_node
            ]

            relationship = edge_data.get(
                "relationship",
                "",
            )

            if relationship.startswith(
                "HAS_"
            ):

                services.append(
                    {
                        "node": neighbor,
                        "relationship": relationship,
                        "direction": "incoming",
                        "attributes": dict(
                            self.graph.nodes[
                                neighbor
                            ]
                        ),
                    }
                )

        return services

    def get_customer_tickets(
        self,
        customer_id: str,
    ) -> list[dict]:

        customer_node = (
            self.get_customer_node(
                customer_id
            )
        )

        tickets = []

        for neighbor in (
            self.graph.successors(
                customer_node
            )
        ):

            edge_data = self.graph.edges[
                customer_node,
                neighbor
            ]

            relationship = edge_data.get(
                "relationship",
                "",
            )

            if relationship != (
                "CREATED_TICKET"
            ):

                continue

            ticket_data = dict(
                self.graph.nodes[
                    neighbor
                ]
            )

            tickets.append(
                {
                    "ticket_id": neighbor,
                    **ticket_data,
                }
            )

        return tickets

    def get_customer_risk(
        self,
        customer_id: str,
    ) -> dict:

        customer_node = (
            self.get_customer_node(
                customer_id
            )
        )

        data = dict(
            self.graph.nodes[
                customer_node
            ]
        )

        return {
            "customer_id": customer_id,
            "churn_probability": data.get(
                "churn_probability"
            ),
            "risk_score": data.get(
                "risk_score"
            ),
            "risk_level": data.get(
                "risk_level"
            ),
            "predicted_churn": data.get(
                "predicted_churn"
            ),
            "actual_churn": data.get(
                "churn"
            ),
        }

    def get_customer_issues(
        self,
        customer_id: str,
    ) -> list[str]:

        tickets = (
            self.get_customer_tickets(
                customer_id
            )
        )

        issues = []

        for ticket in tickets:

            issue_type = ticket.get(
                "issue_type"
            )

            if issue_type:
                issues.append(
                    issue_type
                )

        return issues

    def build_retrieval_context(
        self,
        customer_id: str,
    ) -> dict:

        return {
            "customer_id": customer_id,
            "customer": dict(
                self.graph.nodes[
                    self.get_customer_node(
                        customer_id
                    )
                ]
            ),
            "risk": self.get_customer_risk(
                customer_id
            ),
            "services": self.get_customer_services(
                customer_id
            ),
            "support_tickets": self.get_customer_tickets(
                customer_id
            ),
            "issues": self.get_customer_issues(
                customer_id
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
        / "customer_support_knowledge_graph.graphml"
    )

    customer_id = "5178-LMXOP"

    print()
    print("GRAPH RETRIEVAL TEST")
    print("=" * 70)

    retriever = CustomerGraphRetriever(
        graph_path
    )

    print(
        f"Graph nodes: "
        f"{retriever.graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: "
        f"{retriever.graph.number_of_edges()}"
    )

    context = (
        retriever.build_retrieval_context(
            customer_id
        )
    )

    print()
    print("CUSTOMER")
    print("-" * 70)
    print(context["customer"])

    print()
    print("RISK")
    print("-" * 70)
    print(context["risk"])

    print()
    print("SERVICES")
    print("-" * 70)

    for service in context["services"]:

        print(
            f"{service['relationship']:25} "
            f"{service['direction']:10} "
            f"{service['node']}"
        )

    print()
    print("SUPPORT TICKETS")
    print("-" * 70)

    for ticket in context[
        "support_tickets"
    ]:

        print(ticket)

    print()
    print("ISSUES")
    print("-" * 70)
    print(context["issues"])


if __name__ == "__main__":
    main()
