
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.agents.local_agent import LocalCustomerAgent
from src.agents.tools import (
    get_customer_analysis,
    get_customer_context,
    get_customer_profile,
    get_customer_risk,
    get_customer_services,
    get_customer_tickets,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Risk & Support Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def get_customer_ids() -> list[str]:
    """
    Load available customer IDs from the existing
    customer risk report.

    No model training happens here.
    """

    risk_path = (
        PROJECT_ROOT
        / "reports"
        / "customer_risk_scores.csv"
    )

    if not risk_path.exists():
        raise FileNotFoundError(
            f"Risk report not found: {risk_path}"
        )

    df = pd.read_csv(risk_path)

    if "customerID" not in df.columns:
        raise ValueError(
            "customer_risk_scores.csv does not contain "
            "'customerID'."
        )

    return sorted(
        df["customerID"]
        .astype(str)
        .tolist()
    )


def format_tenure(value: Any) -> str:
    """Format customer tenure."""

    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if value == 1:
        return "1 month"

    return f"{value} months"


def render_ticket(
    ticket: dict[str, Any],
) -> None:
    """Render one support ticket using native Streamlit UI."""

    ticket_id = ticket.get(
        "ticket_id",
        "Unknown",
    )

    issue = ticket.get(
        "issue_type",
        "Unknown",
    )

    priority = ticket.get(
        "priority",
        "Unknown",
    )

    status = ticket.get(
        "status",
        "Unknown",
    )

    channel = ticket.get(
        "channel",
        "Unknown",
    )

    resolution = ticket.get(
        "resolution",
        "Unknown",
    )

    created_days = ticket.get(
        "created_days_ago",
        "Unknown",
    )

    with st.container(border=True):

        st.markdown(
            f"**{ticket_id} — {issue}**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(
                f"Priority: **{priority}**"
            )

        with col2:
            st.write(
                f"Status: **{status}**"
            )

        with col3:
            st.write(
                f"Channel: **{channel}**"
            )

        st.caption(
            f"Created {created_days} days ago • "
            f"Resolution: {resolution}"
        )


# ============================================================
# GEMINI ERROR HANDLING
# ============================================================

def format_gemini_error(
    error: Exception,
) -> str:
    """
    Convert raw Gemini API errors into a clean
    user-facing message.
    """

    error_text = str(error)
    lowered = error_text.lower()

    if (
        "429" in lowered
        and (
            "quota" in lowered
            or "resource_exhausted" in lowered
            or "per day" in lowered
        )
    ):
        return (
            "Gemini's daily API quota has been exhausted. "
            "The local Customer Intelligence Agent is still "
            "available."
        )

    if "429" in lowered:
        return (
            "Gemini is temporarily rate-limited. "
            "Please try again shortly."
        )

    if "503" in lowered:
        return (
            "Gemini is temporarily unavailable because "
            "the model is experiencing high demand. "
            "Please try again shortly."
        )

    if "504" in lowered:
        return (
            "Gemini took too long to respond. "
            "Please try again."
        )

    if "401" in lowered:
        return (
            "Gemini authentication failed. "
            "Please check the API key configuration."
        )

    if "403" in lowered:
        return (
            "Gemini rejected the request because the "
            "configured API key or project does not "
            "have the required permissions."
        )

    if "404" in lowered:
        return (
            "The configured Gemini model could not be found."
        )

    return (
        "Gemini could not complete the request. "
        "The local Customer Intelligence Agent is "
        "still available."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


if "agent_customer_id" not in st.session_state:
    st.session_state.agent_customer_id = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Customer Risk & Support Intelligence"
)

st.caption(
    "Machine Learning • Knowledge Graph • RAG • "
    "Tool Calling • AI Agents"
)

st.divider()


# ============================================================
# LOAD CUSTOMER IDS
# ============================================================

try:

    customer_ids = get_customer_ids()

except Exception as error:

    st.error(
        "Unable to load customer IDs."
    )

    st.exception(error)

    st.stop()


if not customer_ids:

    st.error(
        "No customer IDs were found."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Customer Explorer"
    )

    default_index = 0

    if "5178-LMXOP" in customer_ids:

        default_index = customer_ids.index(
            "5178-LMXOP"
        )

    customer_id = st.selectbox(
        "Select Customer",
        options=customer_ids,
        index=default_index,
    )

    st.divider()

    st.subheader(
        "System Architecture"
    )

    st.markdown(
        """
        **ML Risk Model**

        ↓

        **Knowledge Graph**

        ↓

        **RAG Retrieval**

        ↓

        **Customer Tools**

        ↓

        **AI Agents**
        """
    )

    st.divider()

    st.caption(
        "The dashboard uses the existing backend "
        "without retraining the ML model."
    )


# ============================================================
# RESET AGENT CHAT WHEN CUSTOMER CHANGES
# ============================================================

if (
    st.session_state.agent_customer_id
    != customer_id
):

    st.session_state.agent_messages = []

    st.session_state.agent_customer_id = (
        customer_id
    )


# ============================================================
# LOAD CUSTOMER INTELLIGENCE
# ============================================================

try:

    risk = get_customer_risk(
        customer_id
    )

    profile = get_customer_profile(
        customer_id
    )

    services = get_customer_services(
        customer_id
    )

    tickets = get_customer_tickets(
        customer_id
    )

    analysis = get_customer_analysis(
        customer_id
    )

except Exception as error:

    st.error(
        "Unable to load customer intelligence."
    )

    st.exception(error)

    st.stop()


if "error" in risk:

    st.error(
        risk["error"]
    )

    st.stop()


# ============================================================
# CUSTOMER INTELLIGENCE AGENT
# ============================================================

st.header(
    "🤖 Customer Intelligence Agent"
)

st.caption(
    f"Ask questions about customer **{customer_id}**. "
    "The deterministic agent uses the existing customer "
    "analysis, knowledge graph, RAG, and trusted tools."
)


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.markdown(
    "**Quick questions**"
)

quick_col1, quick_col2, quick_col3 = st.columns(3)

quick_questions = [
    (
        quick_col1,
        "📊 Risk Assessment",
        "What is this customer's current risk and why?",
    ),
    (
        quick_col2,
        "🎫 Support Issues",
        "What unresolved support issues need attention?",
    ),
    (
        quick_col3,
        "📋 Complete Briefing",
        "Give me a complete support-agent briefing "
        "for this customer.",
    ),
]


for column, label, question in quick_questions:

    with column:

        if st.button(
            label,
            use_container_width=True,
        ):

            st.session_state.agent_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            try:

                local_agent = (
                    LocalCustomerAgent()
                )

                result = local_agent.run(
                    customer_id=customer_id,
                    question=question,
                )

                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": result["response"],
                    }
                )

            except Exception as error:

                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I could not complete the "
                            "customer analysis."
                        ),
                    }
                )

                st.session_state.agent_error = (
                    str(error)
                )

            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.agent_messages:

    st.subheader(
        "Conversation"
    )

    for message in (
        st.session_state.agent_messages
    ):

        if message["role"] == "user":

            with st.chat_message(
                "user",
                avatar="👤",
            ):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message(
                "assistant",
                avatar="🤖",
            ):

                st.write(
                    message["content"]
                )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    f"Ask anything about {customer_id}..."
)


if prompt:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.agent_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.write(
            prompt
        )

    # --------------------------------------------------------
    # AGENT RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        with st.spinner(
            "Analyzing customer intelligence..."
        ):

            try:

                local_agent = (
                    LocalCustomerAgent()
                )

                result = local_agent.run(
                    customer_id=customer_id,
                    question=prompt,
                )

                answer = result[
                    "response"
                ]

                st.write(
                    answer
                )

                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                with st.expander(
                    "Agent Execution Trace"
                ):

                    st.write(
                        f"Detected intent: "
                        f"`{result['intent']}`"
                    )

                    for index, trace_item in enumerate(
                        result["trace"],
                        start=1,
                    ):

                        st.write(
                            f"{index}. "
                            f"{trace_item['step']}"
                        )

                        if trace_item.get(
                            "details"
                        ):

                            st.json(
                                trace_item[
                                    "details"
                                ]
                            )

            except Exception as error:

                error_message = (
                    "I could not complete the "
                    "customer analysis."
                )

                st.error(
                    error_message
                )

                st.session_state.agent_messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ============================================================
# RISK OVERVIEW
# ============================================================

st.divider()

st.header(
    "📊 Risk Overview"
)

predicted_churn = (
    "Yes"
    if risk["predicted_churn"] == 1
    else "No"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Risk Level",
        risk["risk_level"],
    )

with col2:

    st.metric(
        "Churn Probability",
        f"{risk['churn_probability']:.2%}",
    )

with col3:

    st.metric(
        "Risk Score",
        f"{risk['risk_score']:.2f}",
    )

with col4:

    st.metric(
        "Predicted Churn",
        predicted_churn,
    )

st.caption(
    "Churn probability and risk level are machine-learning "
    "outputs and are not guarantees of future customer behavior."
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.header(
    "👤 Customer Profile"
)

profile_col1, profile_col2 = st.columns(2)


with profile_col1:

    profile_left = pd.DataFrame(
        {
            "Attribute": [
                "Customer ID",
                "Gender",
                "Senior Citizen",
                "Partner",
                "Dependents",
            ],
            "Value": [
                profile.get(
                    "customer_id"
                ),
                profile.get(
                    "gender"
                ),
                profile.get(
                    "senior_citizen"
                ),
                profile.get(
                    "partner"
                ),
                profile.get(
                    "dependents"
                ),
            ],
        }
    )

    st.dataframe(
        profile_left,
        use_container_width=True,
        hide_index=True,
    )


with profile_col2:

    monthly_charges = profile.get(
        "monthly_charges"
    )

    if monthly_charges is not None:

        monthly_charges_text = (
            f"{float(monthly_charges):.2f}"
        )

    else:

        monthly_charges_text = "N/A"

    profile_right = pd.DataFrame(
        {
            "Attribute": [
                "Contract",
                "Tenure",
                "Monthly Charges",
            ],
            "Value": [
                profile.get(
                    "contract"
                ),
                format_tenure(
                    profile.get(
                        "tenure"
                    )
                ),
                monthly_charges_text,
            ],
        }
    )

    st.dataframe(
        profile_right,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SUPPORT INTELLIGENCE
# ============================================================

support = analysis[
    "support_summary"
]

st.header(
    "🎫 Support Intelligence"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Tickets",
        support["total_tickets"],
    )

with col2:

    st.metric(
        "Unresolved",
        support["unresolved_tickets"],
    )

with col3:

    st.metric(
        "High Priority",
        support["high_priority_tickets"],
    )

with col4:

    st.metric(
        "Urgent",
        support["urgent_tickets"],
    )

with col5:

    st.metric(
        "Avg Resolution",
        f"{support['average_resolution_time']:.2f}h",
    )


# ============================================================
# UNRESOLVED ISSUES
# ============================================================

st.header(
    "🚨 Unresolved Issues"
)

unresolved_tickets = analysis[
    "unresolved_tickets"
]

if unresolved_tickets:

    for ticket in unresolved_tickets:

        render_ticket(
            ticket
        )

else:

    st.success(
        "No unresolved support tickets."
    )


# ============================================================
# ISSUE DISTRIBUTION
# ============================================================

st.header(
    "📈 Issue Distribution"
)

issue_distribution = analysis[
    "issue_distribution"
]

if issue_distribution:

    issue_df = pd.DataFrame(
        {
            "Issue": list(
                issue_distribution.keys()
            ),
            "Tickets": list(
                issue_distribution.values()
            ),
        }
    )

    st.bar_chart(
        issue_df.set_index(
            "Issue"
        )
    )

else:

    st.info(
        "No issue distribution data available."
    )


# ============================================================
# CUSTOMER SERVICES
# ============================================================

st.header(
    "🛠️ Customer Services"
)

if services:

    service_rows = []

    for service in services:

        service_rows.append(
            {
                key: value
                for key, value in service.items()
                if key != "node_type"
            }
        )

    st.dataframe(
        pd.DataFrame(
            service_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No service information available."
    )


# ============================================================
# SUPPORT TICKET HISTORY
# ============================================================

st.header(
    "📋 Support Ticket History"
)

if tickets:

    ticket_rows = []

    for ticket in tickets:

        ticket_rows.append(
            {
                key: value
                for key, value in ticket.items()
                if key != "node_type"
            }
        )

    st.dataframe(
        pd.DataFrame(
            ticket_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No support tickets found."
    )


# ============================================================
# RAG CUSTOMER CONTEXT
# ============================================================

st.header(
    "🔎 RAG Customer Context"
)

st.caption(
    "Retrieve and rank customer-specific evidence "
    "from the existing knowledge graph."
)

rag_question = st.text_input(
    "Retrieval question",
    value=(
        "What support issues and customer information "
        "are most relevant?"
    ),
    key="rag_question",
)

if st.button(
    "Retrieve Relevant Context",
    key="rag_button",
):

    with st.spinner(
        "Retrieving customer context..."
    ):

        try:

            context = get_customer_context(
                customer_id=customer_id,
                question=rag_question,
            )

            st.subheader(
                "Retrieved Context"
            )

            st.code(
                context,
                language="text",
            )

        except Exception as error:

            st.error(
                "RAG retrieval failed."
            )

            st.exception(error)


# ============================================================
# GEMINI FUNCTION-CALLING AGENT
# ============================================================

st.header(
    "✨ Gemini Function-Calling Agent"
)

st.caption(
    "Optional LLM-powered agent using the same "
    "trusted customer tools."
)

gemini_question = st.text_input(
    "Ask Gemini about this customer",
    value=(
        "What unresolved support issues need attention?"
    ),
    key="gemini_question",
)

if st.button(
    "✨ Ask Gemini",
    key="gemini_button",
):

    with st.spinner(
        "Gemini is processing..."
    ):

        try:

            from src.agents.agent_runner import (
                CustomerSupportAgent,
            )

            gemini_agent = (
                CustomerSupportAgent()
            )

            answer = gemini_agent.ask(
                customer_id=customer_id,
                question=gemini_question,
            )

            st.success(
                "Gemini response received."
            )

            with st.chat_message(
                "assistant",
                avatar="✨",
            ):

                st.write(
                    answer
                )

        except Exception as error:

            st.warning(
                "✨ Gemini is temporarily unavailable."
            )

            st.info(
                format_gemini_error(
                    error
                )
            )

            st.markdown(
                """
                **Your customer intelligence is still available.**

                You can continue using:

                - 🤖 Customer Intelligence Agent
                - 🔎 RAG Customer Context
                - 📊 ML Risk Analysis
                - 🎫 Support Intelligence
                - 🛠️ Customer Services
                """
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


# ============================================================
# COMPLETE CUSTOMER ANALYSIS
# ============================================================

st.header(
    "📦 Complete Customer Analysis"
)

with st.expander(
    "View structured analysis"
):

    st.json(
        analysis
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Risk & Support Intelligence • "
    "Machine Learning • Knowledge Graph • RAG • "
    "Tool Calling • AI Agents"
)

