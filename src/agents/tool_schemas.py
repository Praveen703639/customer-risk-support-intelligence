CUSTOMER_AGENT_TOOLS = [
    {
        "name": "get_customer_risk",
        "description": (
            "Get the machine-learning risk assessment "
            "for a specific customer, including churn "
            "probability, risk score, risk level, "
            "predicted churn, and actual churn label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "The unique customer ID."
                    ),
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_customer_tickets",
        "description": (
            "Get all support tickets associated with "
            "a specific customer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "The unique customer ID."
                    ),
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_customer_context",
        "description": (
            "Retrieve and rank customer information "
            "relevant to a specific support question. "
            "Use this when the question requires "
            "context-aware support evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "The unique customer ID."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "The support question that "
                        "the retrieved context should "
                        "address."
                    ),
                },
            },
            "required": [
                "customer_id",
                "question",
            ],
        },
    },
    {
        "name": "get_customer_analysis",
        "description": (
            "Perform a deterministic comprehensive "
            "customer analysis combining customer "
            "profile, ML risk, services, support "
            "tickets, unresolved issues, and issue "
            "distribution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "The unique customer ID."
                    ),
                }
            },
            "required": ["customer_id"],
        },
    },
]