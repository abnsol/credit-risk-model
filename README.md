# CREDIT RISK MODEL

## Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml        
├── data/                           
│   ├── raw/                        
│   └── processed/                  
├── notebooks/
│   └── 1.0-eda.ipynb               
├── src/
│   ├── __init__.py
│   ├── data_processing.py         
│   ├── train.py                    
│   ├── predict.py                  
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

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes the importance of accurate risk measurement for financial institutions. This has direct implications for credit risk models: regulators require that models be transparent, auditable, and well-documented. Interpretable models ensure that financial institutions can explain their risk assessments to auditors, regulators, and stakeholders. Therefore, using explainable techniques such as Weight of Evidence (WoE) transformations and Logistic Regression becomes not only preferable but often necessary in regulated environments.

### 2. Proxy Variables for Default and Associated Risks

In many real-world datasets, a direct label indicating customer default may not be available. In such cases, a **proxy variable**—such as payment delinquency beyond a certain threshold—is created to approximate default behavior. However, relying on proxy labels introduces **business risks**:
- The proxy may not capture true default behavior, leading to misclassification.
- Poor proxy definitions can bias the model and reduce its generalizability.
- Regulatory bodies may scrutinize or reject models trained on inappropriate or poorly documented proxies.

Hence, careful definition and documentation of the proxy variable are critical to ensure model validity and compliance.

### 3. Model Trade-offs in Regulated Financial Contexts

There is an important trade-off between interpretability and predictive performance in credit scoring models:

- **Simple models (e.g., Logistic Regression with WoE):**
  - Highly interpretable and explainable.
  - Easier to validate and justify to regulators.
  - May underperform compared to complex models.

- **Complex models (e.g., Gradient Boosting Machines):**
  - Often yield higher predictive accuracy.
  - More capable of capturing nonlinear relationships and interactions.
  - Lack transparency, harder to interpret, and may not be accepted by regulatory authorities without added explainability tools (e.g., SHAP values).

In regulated financial contexts, it’s often necessary to strike a balance: favoring simpler models for production use while reserving complex models for internal benchmarking or decision support.
