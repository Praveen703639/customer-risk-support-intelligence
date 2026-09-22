from pathlib import Path

from src.rag.graph_retriever import CustomerGraphRetriever
from src.rag.context_ranker import CustomerContextRanker


class RAGContextBuilder:

    def __init__(
        self,
        ranker: CustomerContextRanker,
    ) -> None:
        self.ranker = ranker

    def build_context(
        self,
        customer_id: str,
        query: str,
        top_k: int = 5,
    ) -> str:

        context = self.ranker.retrieve_relevant_context(
            customer_id=customer_id,
            query=query,
            top_k=top_k,
        )

        customer = context.get("customer", {})
        risk = context.get("risk", {})
        services = context.get("services", [])
        tickets = context.get(
            "ranked_support_tickets",
            context.get("support_tickets", []),
        )

        sections = []

        sections.append(
            "## CUSTOMER PROFILE\n"
            f"Customer ID: {customer.get('customer_id', 'Unknown')}\n"
            f"Gender: {customer.get('gender', 'Unknown')}\n"
            f"Senior citizen: {customer.get('senior_citizen', 'Unknown')}\n"
            f"Partner: {customer.get('partner', 'Unknown')}\n"
            f"Dependents: {customer.get('dependents', 'Unknown')}\n"
            f"Tenure: {customer.get('tenure', 'Unknown')} months\n"
            f"Contract: {customer.get('contract', 'Unknown')}\n"
            f"Monthly charges: {customer.get('monthly_charges', 'Unknown')}"
        )

        probability = risk.get("churn_probability")

        if probability is not None:
            probability_text = f"{float(probability) * 100:.2f}%"
        else:
            probability_text = "Unknown"

        sections.append(
            "## RISK ASSESSMENT\n"
            f"Churn probability: {probability_text}\n"
            f"Risk score: {risk.get('risk_score', 'Unknown')}\n"
            f"Risk level: {risk.get('risk_level', 'Unknown')}\n"
            f"Predicted churn: {risk.get('predicted_churn', 'Unknown')}\n"
            f"Actual churn label: {risk.get('actual_churn', 'Unknown')}"
        )

        service_lines = [
            "## SERVICE CONFIGURATION"
        ]

        for service in services:

            relationship = service.get(
                "relationship",
                "UNKNOWN",
            )

            attributes = service.get(
                "attributes",
                {},
            )

            value = attributes.get(
                "value",
                service.get("node", "Unknown"),
            )

            service_type = attributes.get(
                "node_type",
                relationship,
            )

            service_lines.append(
                f"- {service_type}: {value}"
            )

        sections.append(
            "\n".join(service_lines)
        )

        ticket_lines = [
            "## RANKED SUPPORT EVIDENCE"
        ]

        if not tickets:
            ticket_lines.append(
                "No support-ticket evidence was retrieved."
            )

        for index, ticket in enumerate(
            tickets,
            start=1,
        ):

            ticket_lines.extend(
                [
                    "",
                    f"### Evidence {index}",
                    f"Ticket ID: {ticket.get('ticket_id', 'Unknown')}",
                    f"Issue: {ticket.get('issue_type', 'Unknown')}",
                    f"Priority: {ticket.get('priority', 'Unknown')}",
                    f"Status: {ticket.get('status', 'Unknown')}",
                    f"Channel: {ticket.get('channel', 'Unknown')}",
                    f"Age: {ticket.get('created_days_ago', 'Unknown')} days",
                    f"Resolution: {ticket.get('resolution', 'Unknown')}",
                    f"Retrieval score: {ticket.get('retrieval_score', 'Unknown')}",
                ]
            )

        sections.append(
            "\n".join(ticket_lines)
        )

        sections.append(
            "## GROUNDING INSTRUCTIONS\n"
            "Answer the user's question using only "
            "the evidence provided above.\n"
            "Do not invent customer information.\n"
            "Distinguish model predictions from observed "
            "customer data.\n"
            "If the evidence is insufficient, explicitly "
            "state that the available context is insufficient."
        )

        return "\n\n".join(sections)


def main():

    project_root = Path(__file__).resolve().parents[2]

    graph_path = (
        project_root
        / "reports"
        / "graph"
        / "customer_support_knowledge_graph.graphml"
    )

    customer_id = "5178-LMXOP"

    query = (
        "What should a support agent know about "
        "this customer's current problems and risk?"
    )

    print()
    print("RAG CONTEXT BUILDER")
    print("=" * 70)

    retriever = CustomerGraphRetriever(graph_path)

    ranker = CustomerContextRanker(retriever)

    builder = RAGContextBuilder(ranker)

    context = builder.build_context(
        customer_id=customer_id,
        query=query,
        top_k=5,
    )

    print()
    print("GENERATED GROUNDING CONTEXT")
    print("=" * 70)

    print(context)


if __name__ == "__main__":
    main()
