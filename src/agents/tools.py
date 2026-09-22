
from pathlib import Path
from typing import Any

from src.agents.analyzer import analyze_customer
from src.rag.context_builder import RAGContextBuilder
from src.rag.context_ranker import CustomerContextRanker
from src.rag.graph_retriever import CustomerGraphRetriever


# =============================================================
# PROJECT PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "reports"
    / "graph"
    / "customer_support_knowledge_graph.graphml"
)


# =============================================================
# RETRIEVER
# =============================================================

_retriever: CustomerGraphRetriever | None = None


def get_retriever() -> CustomerGraphRetriever:
    """
    Return the shared customer knowledge-graph retriever.

    The retriever is initialized lazily so that the knowledge
    graph is loaded only when one of the customer tools actually
    needs it.
    """

    global _retriever

    if _retriever is None:
        _retriever = CustomerGraphRetriever(
            GRAPH_PATH
        )

    return _retriever


# =============================================================
# CUSTOMER PROFILE
# =============================================================

def get_customer_profile(
    customer_id: str,
) -> dict[str, Any]:
    """
    Retrieve the basic profile of a customer.
    """

    retriever = get_retriever()

    customer_node = retriever.get_customer_node(
        customer_id
    )

    if not customer_node:
        return {
            "error": (
                f"Customer {customer_id} "
                f"was not found."
            )
        }

    customer = dict(
        retriever.graph.nodes[customer_node]
    )

    return {
        "customer_id": customer_id,
        "gender": customer.get("gender"),
        "senior_citizen": customer.get(
            "senior_citizen"
        ),
        "partner": customer.get("partner"),
        "dependents": customer.get(
            "dependents"
        ),
        "tenure": customer.get("tenure"),
        "monthly_charges": customer.get(
            "monthly_charges"
        ),
        "contract": customer.get("contract"),
    }


# =============================================================
# CUSTOMER RISK
# =============================================================

def get_customer_risk(
    customer_id: str,
) -> dict[str, Any]:
    """
    Retrieve the machine-learning risk assessment
    for a customer.
    """

    retriever = get_retriever()

    risk = retriever.get_customer_risk(
        customer_id
    )

    if not risk:
        return {
            "error": (
                f"Risk information for "
                f"{customer_id} was not found."
            )
        }

    return risk


# =============================================================
# CUSTOMER SERVICES
# =============================================================

def get_customer_services(
    customer_id: str,
) -> list[dict[str, Any]]:
    """
    Retrieve the customer's service configuration
    from the knowledge graph.
    """

    retriever = get_retriever()

    return retriever.get_customer_services(
        customer_id
    )


# =============================================================
# CUSTOMER TICKETS
# =============================================================

def get_customer_tickets(
    customer_id: str,
) -> list[dict[str, Any]]:
    """
    Retrieve support tickets associated with
    a customer.
    """

    retriever = get_retriever()

    return retriever.get_customer_tickets(
        customer_id
    )


# =============================================================
# CUSTOMER ISSUES
# =============================================================

def get_customer_issues(
    customer_id: str,
) -> list[str]:
    """
    Retrieve the issue types associated with
    a customer's support history.
    """

    retriever = get_retriever()

    return retriever.get_customer_issues(
        customer_id
    )


# =============================================================
# CUSTOMER CONTEXT / RAG
# =============================================================

def get_customer_context(
    customer_id: str,
    question: str,
) -> str:
    """
    Retrieve and rank customer information relevant
    to a support-agent question.

    The context builder provides grounded evidence
    that can later be passed to an LLM response generator.
    """

    retriever = get_retriever()

    ranker = CustomerContextRanker(
        retriever
    )

    builder = RAGContextBuilder(
        ranker
    )

    return builder.build_context(
        customer_id=customer_id,
        query=question,
        top_k=5,
    )


# =============================================================
# CUSTOMER ANALYSIS
# =============================================================

def get_customer_analysis(
    customer_id: str,
) -> dict[str, Any]:
    """
    Perform deterministic analysis of a customer's
    risk and support history.

    This combines the ML risk assessment with
    structured support-history analysis.
    """

    return analyze_customer(
        customer_id
    )


# =============================================================
# TOOL REGISTRY
# =============================================================

TOOL_FUNCTIONS = {
    "get_customer_profile": get_customer_profile,
    "get_customer_risk": get_customer_risk,
    "get_customer_services": get_customer_services,
    "get_customer_tickets": get_customer_tickets,
    "get_customer_issues": get_customer_issues,
    "get_customer_context": get_customer_context,
    "get_customer_analysis": get_customer_analysis,
}

