from src.agents.local_agent import (
    LocalCustomerAgent,
    print_agent_result,
)


def main() -> None:

    customer_id = "5178-LMXOP"

    agent = LocalCustomerAgent()

    questions = [
        "What is this customer's current risk?",
        "What unresolved support issues need attention?",
        "Give me a complete support-agent briefing for this customer.",
    ]

    for question in questions:

        response = agent.run(
            customer_id=customer_id,
            question=question,
        )

        print_agent_result(response)


if __name__ == "__main__":
    main()