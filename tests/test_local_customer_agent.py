from src.agents.local_agent import LocalCustomerAgent


def test_risk_intent():
    agent = LocalCustomerAgent()

    assert (
        agent.detect_intent(
            "What is this customer's current risk?"
        )
        == "risk"
    )


def test_support_intent():
    agent = LocalCustomerAgent()

    assert (
        agent.detect_intent(
            "What unresolved support issues need attention?"
        )
        == "support"
    )


def test_complete_analysis_intent():
    agent = LocalCustomerAgent()

    assert (
        agent.detect_intent(
            "Give me a complete support-agent briefing for this customer."
        )
        == "complete_analysis"
    )


def test_complete_analysis_variations():
    agent = LocalCustomerAgent()

    questions = [
        "Analyze this customer completely.",
        "Give me everything you know about this customer.",
        "Give me a complete analysis including support issues and unresolved tickets.",
        "Give me a full analysis including customer risk and churn.",
    ]

    for question in questions:
        assert (
            agent.detect_intent(question)
            == "complete_analysis"
        )


def test_support_variations():
    agent = LocalCustomerAgent()

    questions = [
        "What support tickets are unresolved?",
        "Show me all unresolved tickets.",
    ]

    for question in questions:
        assert (
            agent.detect_intent(question)
            == "support"
        )


def test_risk_variations():
    agent = LocalCustomerAgent()

    assert (
        agent.detect_intent(
            "What is the customer's churn probability?"
        )
        == "risk"
    )


def test_customer_context_intent():
    agent = LocalCustomerAgent()

    assert (
        agent.detect_intent(
            "What internet plan does this customer have?"
        )
        == "customer_context"
    )