import os

import pytest

from src.agents.agent_runner import CustomerSupportAgent


RUN_GEMINI_TESTS = os.getenv("RUN_GEMINI_TESTS", "").lower() == "true"


@pytest.mark.skipif(
    not RUN_GEMINI_TESTS,
    reason="Live Gemini tests are disabled. Set RUN_GEMINI_TESTS=true to run them.",
)
def test_gemini_function_calling_agent():
    agent = CustomerSupportAgent()

    result = agent.ask(
        customer_id="5178-LMXOP",
        question="What is this customer's current risk?",
    )

    assert result
    assert len(result) > 20