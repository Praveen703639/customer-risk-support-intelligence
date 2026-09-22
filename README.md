# 🧠 Customer Risk & Support Intelligence

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&height=230&text=CUSTOMER%20RISK%20%26%20SUPPORT%20INTELLIGENCE&fontSize=34&fontColor=00F3FF&stroke=00F3FF&strokeWidth=2&desc=ML%20%7C%20KNOWLEDGE%20GRAPH%20%7C%20RAG%20%7C%20AI%20AGENTS&descAlignY=68&descSize=15&theme=matrix" width="100%" alt="Customer Risk and Support Intelligence">

[![Python](https://img.shields.io/badge/Python-3.13-0E1128?style=for-the-badge&logo=python&logoColor=00F3FF)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Knowledge%20Graph-1F425F?style=for-the-badge)](https://networkx.org/)
[![Gemini](https://img.shields.io/badge/Gemini-Function%20Calling-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-9%20Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

### 🚀 [LIVE STREAMLIT DEMO](https://customer-risk-support-intelligence-fbvpyylopxhc69vt2frjem.streamlit.app/)

</div>

---

## 🧠 What Is This?

**Customer Risk & Support Intelligence** is an end-to-end AI engineering project that combines:

- Machine-learning churn risk prediction
- Customer support knowledge graphs
- Query-aware RAG retrieval
- Deterministic analysis tools
- A deterministic local AI agent
- Gemini function-calling
- A production-style Streamlit dashboard

The goal is simple:

> Turn scattered customer, service, support-ticket, and churn information into one grounded customer-intelligence system.

This project goes beyond a traditional ML notebook by connecting the model to a knowledge layer, retrieval system, tool layer, agents, and a deployed application.

---

## ⚡ System At A Glance

| Layer | Technology / Component |
|---|---|
| Risk Prediction | Scikit-Learn |
| Production Risk Model | Tuned Logistic Regression |
| Customer Data | Telco Customer Churn Dataset |
| Support Data | Support Ticket Dataset |
| Knowledge Graph | NetworkX |
| Retrieval | Graph-based RAG |
| Tools | Deterministic Python functions |
| Local Agent | Intent routing + trusted tools |
| LLM Agent | Gemini function calling |
| UI | Streamlit |
| Testing | Pytest |
| Deployment | Streamlit Community Cloud |

---

## 🏗️ Architecture

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

</div>

    ┌──────────────────────────────────────────────────────┐
    │              STREAMLIT DASHBOARD                     │
    │                                                      │
    │  Customer Selector • Risk • Support • RAG • Agents  │
    └───────────────────────┬──────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────┐
    │                    AGENT LAYER                       │
    │                                                      │
    │   LocalCustomerAgent      Gemini Support Agent       │
    └───────────────────────┬──────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────┐
    │                    TOOL LAYER                        │
    │                                                      │
    │ Profile • Risk • Services • Tickets • Issues        │
    │ Context Retrieval • Complete Analysis                │
    └───────────────┬───────────────────────┬──────────────┘
                    │                       │
                    ▼                       ▼
           ┌────────────────┐      ┌────────────────────┐
           │   ML RISK      │      │ KNOWLEDGE GRAPH    │
           │    MODEL       │      │       + RAG        │
           └────────────────┘      └────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                     CUSTOMER INTELLIGENCE

---

## 📊 Machine Learning Risk Engine

The ML layer predicts customer churn probability and converts the probability into an interpretable risk level.

### Pipeline

    Customer Data
          ↓
    Preprocessing
          ↓
    Tuned Logistic Regression
          ↓
    Churn Probability
          ↓
    Risk Score
          ↓
    Risk Category

### Risk Categories

| Probability | Category |
|---:|---|
| < 25% | Low |
| 25% – <50% | Medium |
| 50% – <75% | High |
| ≥ 75% | Critical |

The application clearly separates **machine-learning predictions** from **observed support records**.

> A churn probability is a model output, not a guarantee of future customer behavior.

### Model Artifacts

- decision_tree.joblib
- logistic_regression.joblib
- logistic_regression_tuned.joblib
- preprocessor.joblib
- random_forest_tuned.joblib

The production scoring pipeline uses the tuned logistic-regression artifact.

---

## 🕸️ Customer Support Knowledge Graph

Customer support information is represented as a graph so that relationships between customers, services, tickets, issues, and risk information can be queried together.

    Customer
       │
       ├──────────► Services
       │
       ├──────────► Support Tickets
       │
       ├──────────► Issues
       │
       └──────────► Risk Information

Graph artifacts include:

- customer_support_knowledge_graph.graphml
- customer_service_graph.graphml
- graph_statistics.json
- customer_support_graph_statistics.json
- attribute_risk_analysis.csv

This graph becomes the foundation for customer-specific retrieval.

---

## 🔎 RAG Retrieval

The RAG layer retrieves evidence relevant to a customer's question.

    User Question
          ↓
    Customer ID + Query
          ↓
    Knowledge Graph
          ↓
    Candidate Evidence
          ↓
    Context Ranking
          ↓
    Top-K Relevant Context
          ↓
    Agent / Application

The retrieval implementation is organized under:

    src/rag/
      context_builder.py
      context_ranker.py
      generator.py
      graph_retriever.py

Example:

> What support issues and customer information are most relevant?

The system retrieves evidence for the selected customer rather than relying on a generic global response.

---

## 🧰 Trusted Tool Layer

The agent layer exposes deterministic functions for accessing customer intelligence.

| Tool | Purpose |
|---|---|
| get_customer_profile | Customer attributes |
| get_customer_risk | ML risk information |
| get_customer_services | Customer services |
| get_customer_tickets | Support tickets |
| get_customer_issues | Issue types |
| get_customer_context | Ranked RAG context |
| get_customer_analysis | Comprehensive deterministic analysis |

The architecture is:

    Agent
      ↓
    Tool Schema
      ↓
    Python Function
      ↓
    Trusted Data
      ↓
    Tool Result
      ↓
    Grounded Response

This keeps data access and business logic outside the LLM.

---

## 🤖 Two-Agent Architecture

### 1. Deterministic Local Agent

The local agent requires no external LLM.

    Question
       ↓
    Intent Detection
       ↓
    Tool Selection
       ↓
    Tool Execution
       ↓
    Evidence Collection
       ↓
    Response Generation
       ↓
    Execution Trace

Supported intents include:

- complete_analysis
- risk
- support
- customer_context

This provides a reliable fallback when Gemini is unavailable.

### 2. Gemini Function-Calling Agent

The Gemini agent uses the same trusted tools.

    User Question
          ↓
        Gemini
          ↓
     Function Call
          ↓
     Local Tool
          ↓
   Customer Evidence
          ↓
    Tool Response
          ↓
        Gemini
          ↓
   Grounded Answer

The agent is instructed to:

- Never invent customer information
- Treat tool results as the source of truth
- Distinguish model predictions from observed records
- Reference relevant ticket evidence
- Avoid claiming actions that did not happen

---

## 🖥️ Streamlit Application

### 🚀 Live Application

**[Open Customer Risk & Support Intelligence](https://customer-risk-support-intelligence-fbvpyylopxhc69vt2frjem.streamlit.app/)**

The deployed dashboard contains:

### 🤖 Customer Intelligence Agent
- Risk assessment
- Support issue investigation
- Complete customer briefing
- Interactive agent chat

### 📊 Risk Overview
- Risk level
- Churn probability
- Risk score
- Predicted churn

### 👤 Customer Profile
- Customer identity
- Demographics
- Contract
- Tenure
- Monthly charges

### 🎫 Support Intelligence
- Total tickets
- Unresolved tickets
- High-priority tickets
- Urgent tickets
- Average resolution time

### 🚨 Support Investigation
- Unresolved issue cards
- Issue distribution
- Complete ticket history
- Customer services

### 🔎 RAG Interface
- Natural-language retrieval question
- Ranked customer context

### ✨ Gemini Agent
- LLM-powered tool calling
- Same trusted customer tools
- Graceful fallback when Gemini is unavailable

### 📦 Complete Analysis
- Full structured customer intelligence object

---

## 🔍 Example Customer

Example customer: **5178-LMXOP**

| Signal | Value |
|---|---:|
| Churn Probability | 85.40% |
| Risk Score | 85.40 |
| Risk Level | Critical |
| Predicted Churn | Yes |
| Total Tickets | 5 |
| Unresolved Tickets | 2 |
| High Priority | 4 |
| Urgent | 1 |
| Average Resolution | 16.36 hours |
| Contract | Month-to-month |
| Tenure | 1 month |
| Monthly Charges | 95.10 |

### Unresolved support evidence

**TKT-001115**

- Issue: Service Change
- Priority: High
- Status: In Progress
- Resolution: Pending

**TKT-001117**

- Issue: Internet Connectivity
- Priority: High
- Status: Open
- Resolution: Pending

This demonstrates the central architecture:

    MODEL RISK
         +
    CUSTOMER PROFILE
         +
    SUPPORT HISTORY
         +
    UNRESOLVED ISSUES
         +
    RETRIEVED CONTEXT
         =
    CUSTOMER INTELLIGENCE

---

## 🧪 Testing

The project includes automated tests covering the agent layer and response generation.

Current local result:

**9 passed, 1 skipped**

The live Gemini test is intentionally skipped by default because it requires an external API request.

Run:

    python -m pytest

To explicitly enable the live Gemini test:

    $env:RUN_GEMINI_TESTS="true"
    python -m pytest

---

## 📁 Project Structure

    customer-risk-support-intelligence/
    │
    ├── api/
    │   └── __init__.py
    │
    ├── app/
    │   ├── __init__.py
    │   └── streamlit_app.py
    │
    ├── configs/
    │
    ├── data/
    │   └── raw/
    │       ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
    │       └── support_tickets.csv
    │
    ├── models/
    │
    ├── reports/
    │   ├── customer_risk_scores.csv
    │   └── graph/
    │
    ├── src/
    │   ├── agents/
    │   ├── data/
    │   ├── graph/
    │   ├── models/
    │   ├── rag/
    │   └── utils/
    │
    ├── tests/
    │
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## 🚀 Run Locally

### Clone

    git clone https://github.com/Praveen703639/customer-risk-support-intelligence.git
    cd customer-risk-support-intelligence

### Create environment

    python -m venv .venv

### Windows PowerShell

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

### Install

    python -m pip install -r requirements.txt

### Configure Gemini

Create a local .env file:

    GEMINI_API_KEY=your_api_key_here

Never commit the .env file.

### Start Streamlit

    python -m streamlit run .\app\streamlit_app.py

Then open:

    http://localhost:8501

---

## 🔐 Security

Secrets are excluded from source control.

    .env
      ↓
    .gitignore
      ↓
    NOT COMMITTED

For cloud deployment, provide the Gemini API key through the platform's secret-management system.

---

## ⚠️ Limitations

- Churn probabilities are model outputs, not guarantees.
- Support intelligence depends on the available support dataset.
- Gemini depends on external API availability and quota.
- The deterministic local agent remains available without an external LLM.
- RAG retrieval operates on the project's available graph and customer data.
- This is a portfolio/research system rather than an enterprise production support platform.

---

## 🔭 Future Improvements

- Richer knowledge-graph visualizations
- Retrieval-quality evaluation
- Agent evaluation benchmarks
- SHAP/model explainability
- More support-domain tools
- Persistent conversation memory
- Authentication and role-based access
- External API endpoints
- Monitoring and observability
- Human-in-the-loop action workflows

---

## 💼 Why This Project?

This project demonstrates the progression from traditional machine learning toward a complete AI application:

    Traditional ML
         ↓
    Churn Prediction
         ↓
    Knowledge Graph
         ↓
    RAG
         ↓
    Trusted Tool Layer
         ↓
    AI Agents
         ↓
    Streamlit Product
         ↓
    Cloud Deployment

It brings together:

- Machine learning
- Data preprocessing
- Model persistence
- Knowledge graphs
- Information retrieval
- RAG
- Tool calling
- Deterministic agents
- LLM agents
- Automated testing
- Streamlit
- Cloud deployment
- Git and GitHub engineering

The point is not only to build a churn model.

The point is to demonstrate how multiple AI components can be composed into a usable customer-intelligence product.

---

## 👨‍💻 Author

**Korra Praveen**

B.Tech Computer Science Engineering — IIIT Ranchi

[GitHub @Praveen703639](https://github.com/Praveen703639)

---

<div align="center">

### ⭐ Built with Python • Scikit-Learn • NetworkX • RAG • Gemini • Streamlit

**If you find this project useful, consider starring the repository.**

</div>
