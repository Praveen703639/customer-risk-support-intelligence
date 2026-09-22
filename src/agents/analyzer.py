from pathlib import Path
from typing import Any

from src.rag.graph_retriever import CustomerGraphRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "reports"
    / "graph"
    / "customer_support_knowledge_graph.graphml"
)


_retriever = None


def get_retriever() -> CustomerGraphRetriever:
    """
    Return a cached customer graph retriever.
    """

    global _retriever

    if _retriever is None:
        _retriever = CustomerGraphRetriever(
            GRAPH_PATH
        )

    return _retriever


def analyze_customer(
    customer_id: str,
) -> dict[str, Any]:
    """
    Perform deterministic customer risk and
    support analysis using the knowledge graph.
    """

    retriever = get_retriever()

    # ---------------------------------------------------------
    # Find customer
    # ---------------------------------------------------------

    customer_node = retriever.get_customer_node(
        customer_id
    )

    if not customer_node:
        return {
            "error": (
                f"Customer {customer_id} "
                "was not found."
            )
        }

    customer = dict(
        retriever.graph.nodes[customer_node]
    )

    # ---------------------------------------------------------
    # Retrieve customer data
    # ---------------------------------------------------------

    risk = retriever.get_customer_risk(
        customer_id
    )

    services = retriever.get_customer_services(
        customer_id
    )

    tickets = retriever.get_customer_tickets(
        customer_id
    )

    # ---------------------------------------------------------
    # Classify support tickets
    # ---------------------------------------------------------

    unresolved_tickets = [
        ticket
        for ticket in tickets
        if ticket.get("status")
        in {
            "Open",
            "In Progress",
            "Pending",
        }
    ]

    high_priority_tickets = [
        ticket
        for ticket in tickets
        if ticket.get("priority") == "High"
    ]

    urgent_tickets = [
        ticket
        for ticket in tickets
        if ticket.get("priority") == "Urgent"
    ]

    # ---------------------------------------------------------
    # Issue distribution
    # ---------------------------------------------------------

    issue_distribution: dict[str, int] = {}

    for ticket in tickets:

        issue = ticket.get(
            "issue_type",
            "Unknown",
        )

        issue_distribution[issue] = (
            issue_distribution.get(issue, 0) + 1
        )

    # ---------------------------------------------------------
    # Average resolution time
    # ---------------------------------------------------------

    average_resolution_time = round(
        (
            sum(
                ticket.get(
                    "resolution_hours",
                    0,
                )
                for ticket in tickets
            )
            / len(tickets)
            if tickets
            else 0
        ),
        2,
    )

    # ---------------------------------------------------------
    # Build final analysis
    # ---------------------------------------------------------

    return {
        "customer_id": customer_id,

        "risk": risk,

        "support_summary": {
            "total_tickets": len(tickets),

            "open_tickets": sum(
                1
                for ticket in tickets
                if ticket.get("status") == "Open"
            ),

            "unresolved_tickets": len(
                unresolved_tickets
            ),

            "high_priority_tickets": len(
                high_priority_tickets
            ),

            "urgent_tickets": len(
                urgent_tickets
            ),

            "recent_tickets": sum(
                1
                for ticket in tickets
                if ticket.get(
                    "created_days_ago",
                    9999,
                ) <= 30
            ),

            "average_resolution_time":
                average_resolution_time,
        },

        "issue_distribution":
            issue_distribution,

        "unresolved_tickets":
            unresolved_tickets,

        "customer_profile": {
            "gender": customer.get(
                "gender"
            ),

            "senior_citizen": customer.get(
                "senior_citizen"
            ),

            "partner": customer.get(
                "partner"
            ),

            "dependents": customer.get(
                "dependents"
            ),

            "tenure": customer.get(
                "tenure"
            ),

            "monthly_charges": customer.get(
                "monthly_charges"
            ),

            "contract": customer.get(
                "contract"
            ),
        },

        "services": services,
    }