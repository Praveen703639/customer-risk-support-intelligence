from src.agents.response_generator import CustomerResponseGenerator


def test_generate_analysis_response():
    analysis = {
        "customer_id": "5178-LMXOP",
        "risk": {
            "risk_score": 85.39980067307114,
            "risk_level": "Critical",
        },
        "support_summary": {
            "total_tickets": 5,
            "unresolved_tickets": 2,
        },
        "customer_profile": {
            "contract": "Month-to-month",
            "tenure": 1,
            "monthly_charges": 95.1,
        },
        "unresolved_tickets": [
            {
                "ticket_id": "TKT-001115",
                "issue_type": "Service Change",
                "priority": "High",
                "status": "In Progress",
            },
            {
                "ticket_id": "TKT-001117",
                "issue_type": "Internet Connectivity",
                "priority": "High",
                "status": "Open",
            },
        ],
    }

    response = CustomerResponseGenerator().generate_analysis_response(
        analysis
    )

    assert "5178-LMXOP" in response
    assert "85.40%" in response
    assert "Month-to-month" in response
    assert "1 month" in response
    assert "95.10" in response
    assert "5 recorded support tickets" in response
    assert "2 currently unresolved" in response
    assert "TKT-001115" in response
    assert "TKT-001117" in response
    assert "Service Change" in response
    assert "Internet Connectivity" in response
    assert "model output" in response


def test_generate_risk_response():
    risk = {
        "customer_id": "5178-LMXOP",
        "churn_probability": 0.8539980067307115,
        "risk_score": 85.39980067307114,
        "risk_level": "Critical",
        "predicted_churn": 1,
        "actual_churn": "Yes",
    }

    response = CustomerResponseGenerator().generate_risk_response(
        risk
    )

    assert "5178-LMXOP" in response
    assert "85.40%" in response
    assert "85.40" in response
    assert "critical" in response
    assert "model predicts churn: Yes" in response
    assert "actual churn value is Yes" in response