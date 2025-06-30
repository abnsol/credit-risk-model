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

print("--- Step 1: Create Aggregate Features ---")

# Convert TransactionStartTime to datetime for feature extraction
X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

# Aggregate features at the CustomerId level
# This creates a new DataFrame where each row is a unique customer
customer_df = X.groupby('CustomerId').agg(
    total_transaction_amount=('Amount', 'sum'), # Sum of all transaction amounts for each customer.
    average_transaction_amount=('Amount', 'mean'), # Average transaction amount per customer.
    transaction_count=('TransactionId', 'count'), # Number of transactions per customer.
    std_transaction_amount=('Amount', 'std'), # Variability of transaction amounts per customer.

    distinct_product_categories=('ProductCategory', 'nunique'),# Number of distinct product categories used by customer
    average_pricing_strategy=('PricingStrategy', 'mean'), # Average pricing strategy used by customer
).reset_index()

customer_target = y.groupby(X['CustomerId']).max().reset_index() # max() will give 1 if any fraud occurred for customer
customer_df = customer_df.merge(customer_target, on='CustomerId', how='left')

# Fill NaN in std_transaction_amount for customers with only one transaction (std dev would be NaN)
customer_df['std_transaction_amount'].fillna(0, inplace=True)

print(f"Customer-level DataFrame shape: {customer_df.shape}")
print("First 5 rows of customer-level aggregated data:")
print(customer_df.head())
print("-" * 50)