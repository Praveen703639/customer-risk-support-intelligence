from __future__ import annotations

from typing import Any

from src.agents.response_generator import CustomerResponseGenerator
from src.agents.tools import TOOL_FUNCTIONS


class LocalCustomerAgent:
    """
    Deterministic local agent used for development and testing.

    This agent does not call an LLM.

    It simulates the tool-selection/orchestration layer so that
    the rest of the system can be tested without external API calls.
    """

    def __init__(self) -> None:
        self.trace: list[dict[str, Any]] = []
        self.response_generator = CustomerResponseGenerator()

    # =========================================================
    # TRACE
    # =========================================================

    def _record(
        self,
        step: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an event in the agent execution trace."""

        self.trace.append(
            {
                "step": step,
                "details": details or {},
            }
        )

    # =========================================================
    # TOOL EXECUTION
    # =========================================================

    def _call_tool(
        self,
        tool_name: str,
        **arguments: Any,
    ) -> Any:
        """Execute a registered tool and record its execution."""

        self._record(
            "tool_call",
            {
                "tool": tool_name,
                "arguments": arguments,
            },
        )

        function = TOOL_FUNCTIONS.get(tool_name)

        if function is None:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        result = function(**arguments)

        self._record(
            "tool_result",
            {
                "tool": tool_name,
            },
        )

        return result

    # =========================================================
    # INTENT DETECTION
    # =========================================================

    def detect_intent(
        self,
        question: str,
    ) -> str:
        """
        Deterministic intent router.

        Routing priority:

        1. Complete analysis
        2. Risk
        3. Support
        4. Customer context
        """

        normalized = question.lower().strip()

        # =====================================================
        # COMPLETE ANALYSIS
        # =====================================================

        complete_analysis_keywords = [
            "complete briefing",
            "complete analysis",
            "complete overview",
            "complete summary",
            "full briefing",
            "full analysis",
            "full overview",
            "full summary",
            "support-agent briefing",
            "customer briefing",
            "customer analysis",
            "analyze this customer",
            "analyse this customer",
            "analyze the customer",
            "analyse the customer",
            "analyze everything",
            "analyse everything",
            "everything you know",
            "everything about this customer",
        ]

        if any(
            keyword in normalized
            for keyword in complete_analysis_keywords
        ):
            return "complete_analysis"

        # =====================================================
        # RISK
        # =====================================================

        risk_keywords = [
            "risk",
            "churn",
            "probability",
            "likely to leave",
            "will leave",
            "customer leave",
            "going to leave",
            "might leave",
        ]

        if any(
            keyword in normalized
            for keyword in risk_keywords
        ):
            return "risk"

        # =====================================================
        # SUPPORT
        # =====================================================

        support_keywords = [
            "ticket",
            "tickets",
            "issue",
            "issues",
            "support",
            "unresolved",
            "problem",
            "problems",
            "complaint",
            "complaints",
        ]

        if any(
            keyword in normalized
            for keyword in support_keywords
        ):
            return "support"

        # =====================================================
        # CUSTOMER CONTEXT
        # =====================================================

        context_keywords = [
            "service",
            "services",
            "plan",
            "contract",
            "internet",
            "phone",
            "subscription",
            "package",
            "customer details",
            "customer information",
            "account details",
        ]

        if any(
            keyword in normalized
            for keyword in context_keywords
        ):
            return "customer_context"

        # =====================================================
        # DEFAULT
        # =====================================================

        return "complete_analysis"

    # =========================================================
    # MAIN AGENT ENTRY POINT
    # =========================================================

    def run(
        self,
        customer_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Execute the local customer-support agent.
        """

        self.trace = []

        self._record(
            "question_received",
            {
                "customer_id": customer_id,
                "question": question,
            },
        )

        # =====================================================
        # DETECT INTENT
        # =====================================================

        intent = self.detect_intent(question)

        self._record(
            "intent_detected",
            {
                "intent": intent,
            },
        )

        # =====================================================
        # EXECUTE WORKFLOW
        # =====================================================

        if intent == "risk":

            result = self._run_risk(
                customer_id,
            )

        elif intent == "support":

            result = self._run_support(
                customer_id,
            )

        elif intent == "customer_context":

            result = self._run_context(
                customer_id,
                question,
            )

        else:

            result = self._run_complete_analysis(
                customer_id,
                question,
            )

        # =====================================================
        # GENERATE HUMAN-READABLE RESPONSE
        # =====================================================

        if intent == "risk":

            response = (
                self.response_generator.generate_risk_response(
                    result["risk"]
                )
            )

        elif intent == "support":

            response = (
                self.response_generator.generate_support_response(
                    result["support_summary"],
                    result["unresolved_tickets"],
                )
            )

        elif intent == "customer_context":

            response = (
                self.response_generator.generate_context_response(
                    result["context"]
                )
            )

        else:

            response = (
                self.response_generator.generate_analysis_response(
                    result["analysis"]
                )
            )

        # =====================================================
        # COMPLETION TRACE
        # =====================================================

        self._record(
            "agent_completed",
            {
                "intent": intent,
            },
        )

        return {
            "customer_id": customer_id,
            "question": question,
            "intent": intent,
            "result": result,
            "response": response,
            "trace": self.trace,
        }

    # =========================================================
    # RISK WORKFLOW
    # =========================================================

    def _run_risk(
        self,
        customer_id: str,
    ) -> dict[str, Any]:
        """Retrieve customer churn/risk information."""

        risk = self._call_tool(
            "get_customer_risk",
            customer_id=customer_id,
        )

        return {
            "risk": risk,
        }

    # =========================================================
    # SUPPORT WORKFLOW
    # =========================================================

    def _run_support(
        self,
        customer_id: str,
    ) -> dict[str, Any]:
        """Retrieve support-related customer information."""

        analysis = self._call_tool(
            "get_customer_analysis",
            customer_id=customer_id,
        )

        return {
            "support_summary": analysis[
                "support_summary"
            ],
            "issue_distribution": analysis[
                "issue_distribution"
            ],
            "unresolved_tickets": analysis[
                "unresolved_tickets"
            ],
        }

    # =========================================================
    # CUSTOMER CONTEXT WORKFLOW
    # =========================================================

    def _run_context(
        self,
        customer_id: str,
        question: str,
    ) -> dict[str, Any]:
        """Retrieve customer-specific contextual information."""

        context = self._call_tool(
            "get_customer_context",
            customer_id=customer_id,
            question=question,
        )

        return {
            "context": context,
        }

    # =========================================================
    # COMPLETE ANALYSIS WORKFLOW
    # =========================================================

    def _run_complete_analysis(
        self,
        customer_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Execute the complete customer intelligence workflow.

        This currently uses deterministic tools.

        Later, an LLM can use this structured evidence to
        generate a natural-language customer briefing.
        """

        analysis = self._call_tool(
            "get_customer_analysis",
            customer_id=customer_id,
        )

        context = self._call_tool(
            "get_customer_context",
            customer_id=customer_id,
            question=question,
        )

        return {
            "analysis": analysis,
            "grounding_context": context,
        }


# =============================================================
# CLI RESULT PRINTER
# =============================================================

def print_agent_result(
    response: dict[str, Any],
) -> None:
    """
    Pretty-print the structured agent response for CLI testing.
    """

    print()
    print("=" * 75)
    print("LOCAL CUSTOMER SUPPORT AGENT")
    print("=" * 75)

    print()
    print("CUSTOMER")
    print("-" * 75)
    print(response["customer_id"])

    print()
    print("QUESTION")
    print("-" * 75)
    print(response["question"])

    print()
    print("INTENT")
    print("-" * 75)
    print(response["intent"])

    result = response["result"]

    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    if response.get("response"):

        print()
        print("FINAL RESPONSE")
        print("-" * 75)
        print(response["response"])

    # =========================================================
    # RISK
    # =========================================================

    if "risk" in result:

        print()
        print("RISK")
        print("-" * 75)

        risk = result["risk"]

        print(
            f"Churn probability: "
            f"{risk['churn_probability']:.2%}"
        )

        print(
            f"Risk score: "
            f"{risk['risk_score']:.2f}"
        )

        print(
            f"Risk level: "
            f"{risk['risk_level']}"
        )

        print(
            f"Predicted churn: "
            f"{risk['predicted_churn']}"
        )

        print(
            f"Actual churn: "
            f"{risk['actual_churn']}"
        )

    # =========================================================
    # SUPPORT SUMMARY
    # =========================================================

    if "support_summary" in result:

        print()
        print("SUPPORT SUMMARY")
        print("-" * 75)

        for key, value in result[
            "support_summary"
        ].items():

            print(
                f"{key}: {value}"
            )

    # =========================================================
    # ISSUE DISTRIBUTION
    # =========================================================

    if "issue_distribution" in result:

        print()
        print("ISSUE DISTRIBUTION")
        print("-" * 75)

        for key, value in result[
            "issue_distribution"
        ].items():

            print(
                f"{key}: {value}"
            )

    # =========================================================
    # UNRESOLVED TICKETS
    # =========================================================

    if "unresolved_tickets" in result:

        print()
        print("UNRESOLVED TICKETS")
        print("-" * 75)

        for ticket in result[
            "unresolved_tickets"
        ]:

            print(
                f"{ticket['ticket_id']} | "
                f"{ticket['issue_type']} | "
                f"{ticket['priority']} | "
                f"{ticket['status']}"
            )

    # =========================================================
    # CUSTOMER CONTEXT
    # =========================================================

    if "context" in result:

        print()
        print("CUSTOMER CONTEXT")
        print("-" * 75)

        context = result["context"]

        if isinstance(context, dict):

            for key, value in context.items():

                print(
                    f"{key}: {value}"
                )

        else:

            print(context)

    # =========================================================
    # COMPLETE ANALYSIS
    # =========================================================

    if "analysis" in result:

        analysis = result["analysis"]

        print()
        print("COMPLETE ANALYSIS")
        print("-" * 75)

        risk = analysis["risk"]
        profile = analysis["customer_profile"]

        print(
            f"Risk: "
            f"{risk['risk_level']} "
            f"({risk['churn_probability']:.2%})"
        )

        print(
            f"Contract: "
            f"{profile['contract']}"
        )

        tenure = profile["tenure"]

        tenure_label = (
            "month"
            if tenure == 1
            else "months"
        )

        print(
            f"Tenure: "
            f"{tenure} {tenure_label}"
        )

        print(
            f"Monthly charges: "
            f"{profile['monthly_charges']}"
        )

        print(
            f"Total tickets: "
            f"{analysis['support_summary']['total_tickets']}"
        )

        print(
            f"Unresolved tickets: "
            f"{analysis['support_summary']['unresolved_tickets']}"
        )

    # =========================================================
    # GROUNDING CONTEXT
    # =========================================================

    if "grounding_context" in result:

        print()
        print("GROUNDING CONTEXT")
        print("-" * 75)

        context = result["grounding_context"]

        if isinstance(context, dict):

            for key, value in context.items():

                print(
                    f"{key}: {value}"
                )

        else:

            print(context)

    # =========================================================
    # AGENT TRACE
    # =========================================================

    print()
    print("AGENT TRACE")
    print("-" * 75)

    for index, trace_item in enumerate(
        response["trace"],
        start=1,
    ):

        print(
            f"{index}. "
            f"{trace_item['step']} "
            f"{trace_item['details']}"
        )