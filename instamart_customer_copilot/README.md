# Instamart Customer Intelligence & Support Copilot

**Capstone submission package | Parts A–D | Offline, reproducible, synthetic-data project**

## Overview

This repository builds an end-to-end customer intelligence and support copilot for a fictional quick-commerce business. It combines data quality and SQL analytics, customer segmentation and churn modelling, LLM-style ticket triage with privacy and token-budget controls, and a policy-grounded refund agent with retrieval, tool use, checkpoints, and safe fallbacks.

All customer, order, ticket, and policy-evaluation records are synthetic. No real customer information or live payment/refund service is used. `mock_llm.py` is a deterministic offline stand-in; it does not call a hosted LLM.

### Repository map

```text
instamart_customer_copilot/
├── README.md
├── requirements.txt
├── .gitignore
├── generate_data.py
├── mock_llm.py
├── part_a.py
├── part_b.py
├── part_c.py
├── part_d.py
├── run_all.py
├── sql/queries.sql
├── data/
│   ├── raw/                 # generated source CSVs
│   ├── clean/               # validated/cleaned CSVs
│   ├── policies/            # policy knowledge base
│   └── instamart.db         # SQLite database
├── outputs/                 # pre-generated deliverables and evaluation logs
└── state/                   # agent checkpoints for crash/resume demonstration
```

## Reproduce

The complete generated data and output artifacts are included in this submission. The commands below reproduce them from source on a local Python environment.

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Run the complete pipeline from the repository root.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

The scripts use relative paths and should be run from the repository root. `run_all.py` executes Parts A, B, C, and D in sequence. The pre-generated `data/`, `outputs/`, and `state/` directories are included so reviewers can inspect results without executing the pipeline.

## Part A — Data engineering and SQL analytics

### Pipeline

- Creates deterministic synthetic customers, products, orders, support tickets, policy-evaluation questions, and agent scenarios (`generate_data.py`).
- Cleans inconsistent city/gender values, invalid ages, malformed amounts, duplicate and pending order rows, and invalid delivery times (`part_a.py`).
- Loads validated tables into `data/instamart.db`.
- Executes the analytical SQL in `sql/queries.sql` and writes clean datasets and validation outputs.

### Data quality and audit notes

The deterministic generator reports **1,200 customers, 40 products, 13,084 raw order rows, and 164 tickets**. The raw order file intentionally includes duplicate rows, pending records, malformed numeric values, impossible delivery times, and inconsistent categorical formatting. The cleaning pipeline is designed to make these issues visible and resolve them before analysis.

The included clean order table contains **12,309 rows**: **11,110 completed** and **1,199 cancelled** orders. Completed-order value totals **₹2,998,668.80**. These figures describe the generated synthetic dataset, not real Instamart business performance.

### SQL deliverables

`sql/queries.sql` contains the seven required analytical queries. The generated query outputs and checkpoints are written by `part_a.py`; the run log reports:

| Query | Result rows |
|---|---:|
| Q1 | 30 |
| Q2 | 30 |
| Q3 | 11 |
| Q4 | 25 |
| Q5 | 550 |
| Q6 | 1,175 |
| Q7 | 183 |

The pipeline prints `ALL CHECKPOINTS PASSED` when the expected validation checks succeed.

## Part B — Customer segmentation and retention modelling

### Method

- Builds customer-level behavioural features from cleaned orders.
- Standardises features and compares K-means solutions for K=2 through K=8.
- Uses inertia and silhouette score to assess candidate cluster counts.
- Produces a two-component PCA representation for visual inspection.
- Trains logistic-regression and random-forest lapse classifiers using a stratified train/test split.
- Compares the models with a majority-class baseline and evaluates an oversampled logistic-regression variant.
- Exports customer segment assignments and the retention outreach list.

### K-means comparison

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.7606 | 0.46497 |
| 3 | 1268.2346 | 0.44516 |
| 4 | 743.0711 | 0.52554 |
| 5 | 617.3666 | 0.48050 |
| 6 | 503.1706 | 0.48187 |
| 7 | 408.4605 | 0.41634 |
| 8 | 366.5035 | 0.41233 |

K=4 has the highest silhouette score among the tested solutions. The final exported segment labels are in `outputs/customer_segments.csv`.

### Lapse model evaluation

The modelling cohort contains **1,156 customers**, with **443 labelled lapsed**. The split contains 924 training and 232 test customers.

| Model | Test precision (lapsed) | Test recall (lapsed) | Test F1 (lapsed) |
|---|---:|---:|---:|
| Logistic regression (best C=10) | 0.5938 | 0.6404 | 0.6162 |
| Random forest (depth=8, min leaf=10) | 0.6591 | 0.6517 | 0.6554 |
| Logistic regression with oversampling | 0.5954 | 0.8764 | 0.7091 |
| Majority-class baseline | — | — | 0.0000 |

The oversampled logistic model improves minority-class recall and F1, at the cost of lower precision. This is useful for a retention workflow where missing a potentially lapsed customer may be more expensive than reviewing some false positives. `outputs/retention_list.csv` contains the prioritised outreach list, and `outputs/lapse_model.joblib` contains the fitted model artifact.

## Part C — Support ticket triage copilot

### Design

- Uses deterministic `mock_llm.py` responses so the project can be evaluated offline.
- Classifies tickets into billing, delivery, technical, and fraud categories.
- Compares zero-shot and few-shot prompts.
- Parses and validates model JSON, flags malformed/invalid outputs, and records empty or overlong responses.
- Redacts phone numbers and email addresses before model processing.
- Tracks estimated token use and stops processing when the configured budget is exhausted.
- Routes tickets using category, customer segment, premium status, and safety rules.

### Included artifacts

- `outputs/triage_results.csv` — ticket-level classification, status, segment, priority, route, and token count.
- `outputs/triage_budget_demo.csv` — budget-stop demonstration.
- The script prints classification quality, parse-status counts, token totals, and budget exhaustion details when run.

The mock model is intentionally imperfect: it injects malformed, irrelevant, and transient-error cases to exercise defensive parsing and retry handling. Results should therefore be interpreted as a controlled engineering evaluation, not a benchmark of a commercial language model.

## Part D — Policy RAG and tool-using refund agent

### Retrieval-augmented policy assistant

- Splits the policy text files into overlapping chunks.
- Uses TF-IDF vectors and cosine similarity to retrieve the top policy passages.
- Evaluates retrieval against the supplied policy questions and expected facts.
- Returns a grounded answer or an explicit “could not find” fallback when policy evidence is unavailable.

The included run reports **15 chunks**, **12/12 expected-document retrieval matches**, **12/12 expected-fact checks**, and **2/2 correct fallback checks** for the standard overlapping-chunk configuration.

### Refund agent and safety controls

The agent uses these tools:

1. `get_order` — retrieves an order from the local SQLite database.
2. `search_policy` — retrieves relevant policy passages.
3. `calculate_refund` — applies the simulated late-delivery refund rule.
4. `issue_refund` — writes an approved simulated refund to the local ledger.
5. `escalate_to_human` — records a human-review escalation.

The agent includes a maximum-iteration limit, token limit, time limit, checkpoint persistence, and crash/resume support. Refund actions are simulated and only append records to `outputs/refund_ledger.jsonl`; no payment system is contacted.

### Scenario and resilience outputs

`outputs/agent_runs.jsonl` contains the seven required scenario results, including status, final response, tool observations, model-call count, and token use. `outputs/refund_ledger.jsonl` records simulated refund/escalation actions. `state/` contains saved checkpoints, including the crash/resume example.

The run demonstrates successful refunds, human escalation for high-value refunds, denial for ineligible orders, not-found handling, maximum-iteration fallback, token-limit fallback, timeout fallback, and checkpoint-based recovery.

## Limitations and Next Steps

- The dataset is synthetic and deterministic; it does not represent actual Instamart customers, orders, policies, or operational performance.
- `mock_llm.py` simulates an LLM interface and failure modes. Replace it with a secured hosted model adapter only after adding authentication, privacy review, monitoring, and cost controls.
- Refund tools are local simulations. A production integration would require payment-provider APIs, idempotency keys, audit permissions, and human approval controls.
- TF-IDF is a transparent baseline retriever. A production RAG system should evaluate stronger embedding models, chunking strategies, reranking, and citation-level grounding.
- Model performance depends on synthetic labels and feature assumptions. Validate on representative, consented, production-quality data before operational use.
- Add automated unit tests, CI, environment locking, and a privacy/security review before production deployment.

## Submission note

This folder is structured as a single repository intended for the public GitHub `main` branch. Review the files and generated outputs, then upload the **contents of this folder** to one public GitHub repository. Do not upload personal credentials or real customer data.
