from __future__ import annotations

from typing import Any


class CustomerResponseGenerator:
    """
    Deterministic response generator for customer-support analysis.

    Converts structured customer evidence into human-readable
    support-agent responses.

    This component does not call an LLM and does not generate
    information that is absent from the supplied evidence.
    """

    # =========================================================
    # RISK RESPONSE
    # =========================================================

    def generate_risk_response(
        self,
        risk: dict[str, Any],
    ) -> str:
        """
        Generate a human-readable customer risk response.
        """

        customer_id = risk["customer_id"]
        churn_probability = risk["churn_probability"]
        risk_score = risk["risk_score"]
        risk_level = risk["risk_level"]
        predicted_churn = risk["predicted_churn"]
        actual_churn = risk["actual_churn"]

        predicted_label = (
            "Yes"
            if predicted_churn == 1
            else "No"
        )

        return (
            f"Customer {customer_id} has a "
            f"{risk_level.lower()} model-predicted churn risk "
            f"of {churn_probability:.2%}. "
            f"The risk score is {risk_score:.2f}. "
            f"The model predicts churn: {predicted_label}. "
            f"The recorded actual churn value is {actual_churn}."
        )

    # =========================================================
    # SUPPORT RESPONSE
    # =========================================================

    def generate_support_response(
        self,
        support_summary: dict[str, Any],
        unresolved_tickets: list[dict[str, Any]],
    ) -> str:
        """
        Generate a human-readable support response.
        """

        total_tickets = support_summary["total_tickets"]
        unresolved_count = support_summary["unresolved_tickets"]
        high_priority = support_summary["high_priority_tickets"]
        urgent = support_summary["urgent_tickets"]

        lines = []

        lines.append(
            f"There are {total_tickets} recorded support tickets, "
            f"with {unresolved_count} currently unresolved."
        )

        lines.append(
            f"{high_priority} tickets are marked High priority "
            f"and {urgent} are marked Urgent."
        )

        if unresolved_tickets:
            lines.append(
                "The unresolved tickets are:"
            )

            for ticket in unresolved_tickets:
                lines.append(
                    f"- {ticket['ticket_id']} — "
                    f"{ticket['issue_type']} — "
                    f"{ticket['priority']} priority — "
                    f"{ticket['status']}"
                )
        else:
            lines.append(
                "There are currently no unresolved support tickets."
            )

        lines.append(
            "Ticket statuses and priorities are based on "
            "the recorded support data."
        )

        return "\n".join(lines)

    # =========================================================
    # CUSTOMER CONTEXT RESPONSE
    # =========================================================

    def generate_context_response(
        self,
        context: str,
    ) -> str:
        """
        Generate a human-readable response from retrieved
        customer context.
        """

        return (
            "Here is the relevant customer context:\n\n"
            f"{context}"
        )

    # =========================================================
    # COMPLETE ANALYSIS RESPONSE
    # =========================================================

    def generate_analysis_response(
        self,
        analysis: dict[str, Any],
    ) -> str:
        """
        Generate a complete customer support briefing.
        """

        customer_id = analysis["customer_id"]

        risk = analysis["risk"]
        support = analysis["support_summary"]
        profile = analysis["customer_profile"]
        unresolved = analysis["unresolved_tickets"]

        risk_score = risk["risk_score"]
        risk_level = risk["risk_level"]

        contract = profile["contract"]
        tenure = profile["tenure"]
        monthly_charges = profile["monthly_charges"]

        total_tickets = support["total_tickets"]
        unresolved_count = support["unresolved_tickets"]

        lines = []

        lines.append(
            f"Customer {customer_id} has a "
            f"{risk_level.lower()} model-predicted churn risk "
            f"of {risk_score:.2f}%."
        )

        tenure_label = (
            "month"
            if tenure == 1
            else "months"
        )

        lines.append(
            f"The customer is on a {contract} contract "
            f"and has been with the service for "
            f"{tenure} {tenure_label}. "
            f"Monthly charges are {monthly_charges:.2f}."
        )

        lines.append(
            f"There are {total_tickets} recorded support tickets, "
            f"with {unresolved_count} currently unresolved."
        )

        if unresolved:
            lines.append(
                "The unresolved tickets are:"
            )

            for ticket in unresolved:
                lines.append(
                    f"- {ticket['ticket_id']} — "
                    f"{ticket['issue_type']} — "
                    f"{ticket['priority']} priority — "
                    f"{ticket['status']}"
                )

        else:
            lines.append(
                "There are currently no unresolved support tickets."
            )

        lines.append(
            "The churn prediction is a model output; "
            "the support ticket information and statuses "
            "are observed support records."
        )

        return "\n".join(lines)