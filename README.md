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

## Exploratory Data Analysis (EDA) Insights

Following a comprehensive Exploratory Data Analysis (EDA) of the credit risk dataset, several crucial insights have been identified that will inform subsequent data preprocessing and model development:

1.  **Absence of Explicit Missing Values:** A thorough check revealed no explicit missing values (NaNs) across any columns in the dataset. This simplifies the data cleaning process significantly, as direct imputation strategies for NaNs will not be required.

2.  **Perfect Multicollinearity between 'Amount' and 'Value':** The features 'Amount' and 'Value' exhibit a perfect positive linear correlation (1.00). This strong relationship indicates redundancy, where both features convey identical information. To avoid multicollinearity and streamline the feature set, one of these columns will be dropped during feature engineering.

3.  **Significant Outliers in 'Amount' and 'Value':** Both 'Amount' and 'Value' contain a substantial number of extreme outliers, as visualized in their respective box plots. These outliers could heavily skew statistical analyses and disproportionately impact model training. Robust outlier handling techniques, such as capping (winsorization) or data transformations (e.g., log transformation), will be necessary to manage their impact.

4.  **Weak Linear Correlation with 'FraudResult':** The target variable, 'FraudResult' (indicating fraud), shows very weak linear correlations with all numerical features. This suggests that simple linear relationships alone are insufficient for predicting fraudulent transactions effectively. This highlights the importance of exploring non-linear patterns, leveraging categorical features, and extracting new features (e.g., from `TransactionStartTime`) to build a robust fraud detection model.

5.  **Categorical Nature of 'CountryCode' and 'PricingStrategy' & Temporal Feature Potential:** While `CountryCode` and `PricingStrategy` are currently represented as numerical types, their nature as identifier codes or distinct categories suggests they should be treated as categorical features. Converting them will allow models to better capture their influence. Additionally, the `TransactionStartTime` column, currently an object type, will be parsed into a datetime format to extract valuable temporal features (e.g., hour of day, day of week, time since last transaction for a customer) which could be highly indicative of fraudulent activity.