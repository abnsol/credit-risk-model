import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output during demonstration
import warnings
warnings.filterwarnings('ignore')

print("--- Step 0: Initial Setup and Load Data ---")

# Load the capped data
df = pd.read_csv('./data/processed/data_capped.csv')

# Define the target variable (using FraudResult as a temporary proxy for "default")
TARGET_COLUMN = 'FraudResult'
print(f"Using '{TARGET_COLUMN}' as a temporary proxy for the target variable.")

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("-" * 50)

# (Code from Step 0 would be here)

print("--- Step 1: Create Aggregate Features (Revised) ---")

# Convert TransactionStartTime to datetime for feature extraction
X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

# Aggregate features at the CustomerId level
customer_df = X.groupby('CustomerId').agg(
    total_transaction_amount=('Amount', 'sum'),
    average_transaction_amount=('Amount', 'mean'),
    transaction_count=('TransactionId', 'count'),
    std_transaction_amount=('Amount', 'std'),
    distinct_product_categories=('ProductCategory', 'nunique'),
    average_pricing_strategy=('PricingStrategy', 'mean'),
    
    # Extract time-based features for the *latest* transaction of each customer
    last_transaction_hour=('TransactionStartTime', lambda x: x.max().hour),
    last_transaction_day=('TransactionStartTime', lambda x: x.max().day),
    last_transaction_month=('TransactionStartTime', lambda x: x.max().month),
    last_transaction_year=('TransactionStartTime', lambda x: x.max().year)

).reset_index()

# Also, aggregate the target variable (FraudResult) to the CustomerId level.
customer_target = y.groupby(X['CustomerId']).max().reset_index()
customer_df = customer_df.merge(customer_target, on='CustomerId', how='left')

# Fill NaN in std_transaction_amount for customers with only one transaction
customer_df['std_transaction_amount'].fillna(0, inplace=True)

# Now, we have 'customer_df' which is our new feature set 'X_processed' and target 'y_processed'
X_processed = customer_df.drop(columns=[TARGET_COLUMN])
y_processed = customer_df[TARGET_COLUMN]

print(f"Customer-level DataFrame shape: {X_processed.shape}")
print("First 5 rows of customer-level aggregated data with extracted time features:")
print(X_processed.head())
print("-" * 50)

print("--- Step 2: Extract Features from TransactionStartTime ---")

# In a real customer-level model, you'd apply this to the last/first transaction time per customer
X['TransactionHour'] = X['TransactionStartTime'].dt.hour # The hour of the day when the transaction occurred.
X['TransactionDay'] = X['TransactionStartTime'].dt.day # The day of the month when the transaction occurred.
X['TransactionMonth'] = X['TransactionStartTime'].dt.month # The month when the transaction occurred.
X['TransactionYear'] = X['TransactionStartTime'].dt.year # The year when the transaction occurred.

print("Extracted time-based features from 'TransactionStartTime' (at transaction level for demo).")
print(X[['TransactionStartTime', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']].head())
print("-" * 50)

print("--- Step 3: Handle Missing Values (Demonstration) ---")
# For numerical features, typically mean or median imputation
numerical_imputer = SimpleImputer(strategy='median') 

# For categorical features (if you had any with NaNs you wanted to impute)
categorical_imputer = SimpleImputer(strategy='most_frequent')

print("Missing values in 'std_transaction_amount' were handled during aggregation.")
print("If other missing values arise, SimpleImputer can be used within a pipeline.")
print("-" * 50)