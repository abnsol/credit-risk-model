# Credit Risk Model for High-Risk Customer Identification

This repository houses a comprehensive credit risk modeling project developed as part of the Kifiya AI Mastery training program. The primary objective is to build a robust, interpretable, and deployable credit scoring system that leverages alternative transaction data to identify high-risk customer segments. This is crucial for financial institutions to effectively manage credit risk, optimize capital allocation, and ensure compliance with regulatory frameworks like Basel II.

## Project Structure

```plaintext
credit-risk-model/
├── .github/workflows/ci.yml         # GitHub Actions CI/CD workflow for automated testing
├── data/
│   ├── raw/                         # Raw, unprocessed input data (e.g., data.csv)
│   └── processed/                   # Cleaned data for modeling (model_features.csv, is_high_risk_target.csv)
├── models/
│   ├── final_best_credit_risk_model.pkl
│   └── feature_engineering_pipeline.pkl
├── notebooks/
│   └── 1.0-eda.ipynb                # EDA and initial insights
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   ├── transformers.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

```


## Credit Scoring Business Understanding

Credit scoring models are vital tools in the financial industry, enabling lenders to assess the creditworthiness of borrowers and manage credit risk effectively. This section delves into key considerations for developing such a model, particularly within a regulated environment.

### Influence of Basel II Accord on Model Interpretability and Documentation

Basel II emphasizes:

- Minimum Capital Requirements
- Supervisory Review Process
- Market Discipline

These require that credit risk models be interpretable, validated, and auditable.

**Key Points:**

- **Risk Measurement Accuracy:** Banks must quantify credit risk transparently.
- **Model Validation and Auditability:** Models must be understandable and well-documented.
- **Supervisory Review Process:** Evaluates the robustness of internal models.
- **Market Discipline:** Encourages transparency and market confidence.

Basel II transforms credit risk modeling into a regulatory and risk management function.

## Necessity of Proxy Variables for Default and Associated Business Risks

When true default labels are unavailable, proxy variables must be used.

### Common Proxy Events:

- Days Past Due (DPD), e.g., 90 or 180 days
- Restructuring or Write-off
- Legal Action

### Reasons for Proxy Use:

- True defaults are rare
- Enables early warning signals
- Reflects business-specific definitions of risk

### Risks of Using Proxies:

- **Misclassification:** False positives and negatives
- **Model Bias:** Inherited from proxy design
- **Operational Misalignment:** Between model predictions and formal processes
- **Regulatory Scrutiny:** Must justify proxy use
- **Evolving Landscape:** Proxies may degrade over time

Careful validation and monitoring are essential.

## Trade-offs Between Simple and Complex Models

| Feature | Logistic Regression (WoE) | Gradient Boosting |
|--------|----------------------------|-------------------|
| Interpretability | High | Low |
| Performance (Accuracy) | Good to Moderate | High |
| Regulatory Compliance | Easier | Challenging |
| Model Validation | Straightforward | Complex |
| Explainability | Excellent | Difficult |
| Feature Engineering | Crucial | Less critical initially |
| Development Time | Quick | Slower |
| Risk Management | Easy to embed | More challenging |

**Conclusion:**  
In regulated environments, interpretability often outweighs marginal gains in accuracy. Logistic Regression with WoE is often preferred for deployment, while complex models may serve as challenger models or be used internally.

## Exploratory Data Analysis (EDA) Insights

### Key Findings:

- **Data Size:** 95,662 transaction records, 16 columns.
- **Missing Values:** None explicitly present.
- **Currency and Country:** All in UGX (Uganda), single country code (256).
- **Feature Redundancy:** 'Amount' and 'Value' have 0.99 correlation.
- **Outliers:** Significant in 'Amount' and 'Value', handled via winsorization.
- **Categoricals:** ProviderId, ProductId, ProductCategory, ChannelId, PricingStrategy.
- **Temporal Features:** Extracted from 'TransactionStartTime' (e.g., hour, day, month, year).
- **Target Imbalance:** FraudResult used as a proxy, with only 0.2% positive class.

## Feature Engineering

Implemented via a robust `sklearn.pipeline.Pipeline` with custom transformers.

### Key Steps:

**1. Customer-Level Aggregation:**

- `total_transaction_amount`
- `average_transaction_amount`
- `transaction_count`
- `std_transaction_amount`
- `average_pricing_strategy`
- `distinct_product_categories`

Handled by `CustomerAggregator`.

**2. Temporal Feature Extraction:**

From last transaction:
- `last_transaction_hour`
- `last_transaction_day`
- `last_transaction_month`
- `last_transaction_year`

**3. Proxy Target Variable (RFM + Clustering):**

- RFM metrics: Recency, Frequency, Monetary
- StandardScaler applied
- KMeans clustering into 3 groups
- High-risk segment labeled `1`

**4. WoE Transformation:**

- Handles binning, non-linearity, and interpretability.
- Applied to all numerical features using `RobustWoETransformer`.

**5. Feature Scaling:**

- StandardScaler ensures zero mean and unit variance.

**Final Outputs:**

- `data/processed/model_features.csv`
- `data/processed/is_high_risk_target.csv`

## Model Training and Tracking

### Methodology:

- **Split:** 80/20 with `stratify=y`
- **Models:** Logistic Regression, Random Forest, Gradient Boosting
- **Baselines:** Untuned versions for benchmarks
- **Tuning:** GridSearchCV with 5-fold StratifiedKFold
- **Metric:** ROC AUC
- **Tracking:** MLflow

### MLflow Logged:

- Model type, parameters
- Metrics (ROC AUC, Accuracy, Precision, Recall, F1)
- Model artifact

### Model Performance Summary:

| Model | CV ROC AUC | Test ROC AUC | Precision (1) | Recall (1) | F1 (1) |
|-------|------------|--------------|---------------|------------|--------|
| Logistic Regression (Baseline) | N/A | 0.9722 | 0.99 | 0.86 | 0.92 |
| Logistic Regression (Tuned) | 0.9702 | 0.9723 | 0.99 | 0.86 | 0.92 |
| Random Forest (Baseline) | N/A | 0.9856 | 0.96 | 0.92 | 0.94 |
| Random Forest (Tuned) | 0.9909 | 0.9902 | 0.96 | 0.91 | 0.93 |
| Gradient Boosting (Baseline) | N/A | **0.9929** | 0.94 | 0.91 | 0.93 |
| Gradient Boosting (Tuned) | 0.9922 | 0.9908 | 0.93 | 0.91 | 0.92 |

### Conclusion:

Gradient Boosting (Baseline) was the best-performing model.

### Model Registration and Saving:

- Registered as `CreditRiskClassifier` (Version 1) in MLflow
- Saved locally: `models/final_best_credit_risk_model.pkl`
- Pipeline: `models/feature_engineering_pipeline.pkl`

## Model Deployment and CI/CD

### API Deployment (FastAPI)

**Files:**

- `src/api/pydantic_models.py`: Request/response validation models
- `src/api/main.py`:
  - Loads model and pipeline
  - Exposes `/predict` endpoint
  - Handles prediction logic

### Containerization (Docker)

- `Dockerfile`: Defines image and environment
- `docker-compose.yml`: Maps ports, volumes, builds API container

### Continuous Integration (GitHub Actions)

**File:** `.github/workflows/ci.yml`

**Steps:**

- Trigger on push to `main`
- Set up Python 3.9
- Install dependencies
- Run `flake8` on `src/`
- Run `pytest` on `tests/`
- Fail build if linting or tests fail

---

This README provides a complete technical and business overview of the credit risk modeling project, including its motivation, regulatory context, methodology, experimentation, and deployment.
