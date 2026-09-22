from pathlib import Path
import re

from src.rag.graph_retriever import (
    CustomerGraphRetriever,
)


class CustomerContextRanker:

    PRIORITY_SCORES = {
        "Urgent": 4.0,
        "High": 3.0,
        "Medium": 2.0,
        "Low": 1.0,
    }

    STATUS_SCORES = {
        "Open": 4.0,
        "In Progress": 3.0,
        "Resolved": 1.0,
        "Closed": 0.5,
    }

    ACTIVE_TICKET_BONUS = {
        "Open": 2.0,
        "In Progress": 1.5,
        "Resolved": 0.0,
        "Closed": -0.5,
    }

    def __init__(
        self,
        retriever: CustomerGraphRetriever,
    ) -> None:

        self.retriever = retriever

    @staticmethod
    def tokenize(
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"[a-zA-Z0-9]+",
                text.lower(),
            )
        )

    def query_relevance(
        self,
        query: str,
        ticket: dict,
    ) -> float:

        query_tokens = self.tokenize(
            query
        )

        ticket_text = " ".join(
            [
                str(
                    ticket.get(
                        "issue_type",
                        "",
                    )
                ),
                str(
                    ticket.get(
                        "priority",
                        "",
                    )
                ),
                str(
                    ticket.get(
                        "status",
                        "",
                    )
                ),
                str(
                    ticket.get(
                        "resolution",
                        "",
                    )
                ),
                str(
                    ticket.get(
                        "channel",
                        "",
                    )
                ),
            ]
        )

        ticket_tokens = self.tokenize(
            ticket_text
        )

        if not query_tokens:
            return 0.0

        overlap = (
            query_tokens
            & ticket_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    @staticmethod
    def recency_score(
        created_days_ago,
    ) -> float:

        days = float(
            created_days_ago
        )

        # Exponential-style decay.
        # Recent tickets receive significantly
        # more retrieval weight.
        return max(
            0.0,
            5.0 * (
                1.0
                / (1.0 + days / 30.0)
            ),
        )

    def rank_tickets(
        self,
        query: str,
        tickets: list[dict],
    ) -> list[dict]:

        ranked = []

        for ticket in tickets:

            priority_score = (
                self.PRIORITY_SCORES.get(
                    ticket.get(
                        "priority"
                    ),
                    0.0,
                )
            )

            status_score = (
                self.STATUS_SCORES.get(
                    ticket.get(
                        "status"
                    ),
                    0.0,
                )
            )

            active_bonus = (
                self.ACTIVE_TICKET_BONUS.get(
                    ticket.get(
                        "status"
                    ),
                    0.0,
                )
            )

            recency = (
                self.recency_score(
                    ticket.get(
                        "created_days_ago",
                        9999,
                    )
                )
            )

            relevance = (
                self.query_relevance(
                    query,
                    ticket,
                )
            )

            final_score = (
                priority_score * 0.25
                + status_score * 0.25
                + recency * 0.20
                + relevance * 20.0 * 0.20
                + active_bonus
            )

            ranked_ticket = dict(
                ticket
            )

            ranked_ticket[
                "priority_score"
            ] = round(
                priority_score,
                4,
            )

            ranked_ticket[
                "status_score"
            ] = round(
                status_score,
                4,
            )

            ranked_ticket[
                "active_bonus"
            ] = round(
                active_bonus,
                4,
            )

            ranked_ticket[
                "recency_score"
            ] = round(
                recency,
                4,
            )

            ranked_ticket[
                "query_relevance"
            ] = round(
                relevance,
                4,
            )

            ranked_ticket[
                "retrieval_score"
            ] = round(
                final_score,
                4,
            )

            ranked.append(
                ranked_ticket
            )

        ranked.sort(
            key=lambda item: item[
                "retrieval_score"
            ],
            reverse=True,
        )

        return ranked

    def retrieve_relevant_context(
        self,
        customer_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict:

        context = (
            self.retriever.build_retrieval_context(
                customer_id
            )
        )

        ranked_tickets = (
            self.rank_tickets(
                query,
                context[
                    "support_tickets"
                ],
            )
        )

        context[
            "ranked_support_tickets"
        ] = ranked_tickets[
            :top_k
        ]

        context[
            "retrieval_query"
        ] = query

        return context


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

    customer_id = (
        "5178-LMXOP"
    )

    query = (
        "What should a support agent "
        "know about this customer's "
        "current problems and risk?"
    )

    print()
    print(
        "CUSTOMER CONTEXT RANKING"
    )
    print("=" * 70)

    retriever = (
        CustomerGraphRetriever(
            graph_path
        )
    )

    ranker = (
        CustomerContextRanker(
            retriever
        )
    )

    context = (
        ranker.retrieve_relevant_context(
            customer_id,
            query,
            top_k=5,
        )
    )

    print()
    print("QUERY")
    print("-" * 70)
    print(query)

    print()
    print("CUSTOMER RISK")
    print("-" * 70)
    print(context["risk"])

    print()
    print("RANKED SUPPORT EVIDENCE")
    print("-" * 70)

    for index, ticket in enumerate(
        context[
            "ranked_support_tickets"
        ],
        start=1,
    ):

        print()
        print(
            f"#{index} "
            f"{ticket['ticket_id']}"
        )

        print(
            f"  Issue       : "
            f"{ticket['issue_type']}"
        )

        print(
            f"  Priority    : "
            f"{ticket['priority']}"
        )

        print(
            f"  Status      : "
            f"{ticket['status']}"
        )

        print(
            f"  Channel     : "
            f"{ticket['channel']}"
        )

        print(
            f"  Age         : "
            f"{ticket['created_days_ago']} days"
        )

        print(
            f"  Resolution  : "
            f"{ticket['resolution']}"
        )

        print(
            f"  Retrieval   : "
            f"{ticket['retrieval_score']}"
        )


if __name__ == "__main__":
    main()
