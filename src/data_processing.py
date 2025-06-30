import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Suppress warnings for cleaner output during demonstration
import warnings
warnings.filterwarnings('ignore')

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

# --- Custom Transformer: Customer Aggregator & Feature Extractor ---
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', timestamp_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None): # Added y parameter to transform
        X_copy = X.copy()
        X_copy[self.timestamp_col] = pd.to_datetime(X_copy[self.timestamp_col])

        # Store the original CustomerId for joining the target back
        original_customer_ids = X_copy[self.customer_id_col]

        agg_df = X_copy.groupby(self.customer_id_col).agg(**{
            'total_transaction_amount': ('Amount', 'sum'),
            'average_transaction_amount': ('Amount', 'mean'),
            'std_transaction_amount': ('Amount', 'std'),
            'transaction_count': ('TransactionId', 'count'),
            'average_pricing_strategy': ('PricingStrategy', 'mean'),
            'distinct_product_categories': ('ProductCategory', 'nunique'),
            'last_transaction_hour': (self.timestamp_col, lambda x: x.max().hour),
            'last_transaction_day': (self.timestamp_col, lambda x: x.max().day),
            'last_transaction_month': (self.timestamp_col, lambda x: x.max().month),
            'last_transaction_year': (self.timestamp_col, lambda x: x.max().year)
        }).reset_index()

        agg_df['std_transaction_amount'].fillna(0, inplace=True)
        agg_df = agg_df.set_index(self.customer_id_col)
        
        # If y is provided during transform, aggregate it and return it along with X
        if y is not None:
            # Align y with original CustomerIds for aggregation
            y_df = pd.DataFrame({'CustomerId': original_customer_ids, 'Target': y})
            y_aggregated = y_df.groupby('CustomerId')['Target'].max() # Max to get 1 if any fraud occurred
            return agg_df, y_aggregated # Return both X and y
        else:
            return agg_df # Return only X if y is not provided (e.g., during predict)

# --- Custom Transformer: Robust WoE Transformer ---
class RobustWoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols_to_bin, bins=10, fill_value=0.0):
        self.numerical_cols_to_bin = numerical_cols_to_bin
        self.bins = bins
        self.fill_value = fill_value
        self.woe_maps = {}
        self.binner_info = {}

    def fit(self, X, y): # y is expected here from pipeline
        # X here is the aggregated customer-level features (output of CustomerAggregator)
        # y is the aggregated target (y_aggregated from CustomerAggregator.transform)
        
        # Ensure y is a Series for direct alignment with X's index
        y_series = pd.Series(y, index=X.index)

        for col_name in X.columns:
            if col_name in self.numerical_cols_to_bin:
                try:
                    binned_data, bins_edges = pd.qcut(X[col_name], self.bins, duplicates='drop', retbins=True, precision=4)
                    self.binner_info[col_name] = {'type': 'qcut', 'bins_edges': bins_edges}
                    temp_series = binned_data
                except Exception:
                    binned_data, bins_edges = pd.cut(X[col_name], bins=self.bins, duplicates='drop', retbins=True, precision=4)
                    self.binner_info[col_name] = {'type': 'cut', 'bins_edges': bins_edges}
                    temp_series = binned_data
                
                temp_series = temp_series.astype(str)
            else:
                temp_series = X[col_name].astype(str)

            d0 = pd.DataFrame({'x': temp_series, 'y': y_series}) # Use y_series here
            d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
            d.columns = ['Cutoff', 'N', 'Events']
            d['Non-Events'] = d['N'] - d['Events']
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
            d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
            
            self.woe_maps[col_name] = d.set_index('Cutoff')['WoE'].to_dict()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col_name in X.columns:
            if col_name in self.woe_maps:
                if col_name in self.numerical_cols_to_bin:
                    binner_type = self.binner_info[col_name]['type']
                    bins_edges = self.binner_info[col_name]['bins_edges']
                    binned_data = pd.cut(X_transformed[col_name], bins=bins_edges, include_lowest=True, right=True, duplicates='drop', precision=4)
                    X_transformed[col_name] = binned_data.astype(str)
                else:
                    X_transformed[col_name] = X_transformed[col_name].astype(str)

                X_transformed[col_name] = X_transformed[col_name].map(self.woe_maps[col_name]).fillna(self.fill_value)
            else:
                pass
        return X_transformed

# --- Define the numerical features that will be binned and WoE-transformed after aggregation ---
numerical_features_from_agg_to_woe_bin = [
    'total_transaction_amount', 'average_transaction_amount',
    'std_transaction_amount', 'transaction_count',
    'average_pricing_strategy', 'distinct_product_categories',
    'last_transaction_hour', 'last_transaction_day',
    'last_transaction_month', 'last_transaction_year'
]

# --- Build the full preprocessing pipeline ---
# For complex aggregation scenarios where y must be aligned with aggregated X:

print("\nExecuting Feature Engineering in two phases for target alignment:")
print("Phase 1: Transaction-level to Customer-level Aggregation (X and y)")

# Phase 1: Aggregate data (X) and also aggregate the target (y)
customer_aggregator_transformer = CustomerAggregator(customer_id_col='CustomerId', timestamp_col='TransactionStartTime')
X_aggregated, y_aggregated = customer_aggregator_transformer.transform(X, y)

print(f"Aggregated Features (X_aggregated) shape: {X_aggregated.shape}")
print(f"Aggregated Target (y_aggregated) shape: {y_aggregated.shape}")

print("\nPhase 2: WoE Transformation and Scaling on Customer-level data")

# Phase 2: Create a sub-pipeline for WoE and Scaling
preprocessing_sub_pipeline = Pipeline([
    ('woe_transformer', RobustWoETransformer(
        numerical_cols_to_bin=numerical_features_from_agg_to_woe_bin,
        bins=10
    )),
    ('scaler', StandardScaler())
])

# Fit and transform the aggregated data using the sub-pipeline
X_processed_final = preprocessing_sub_pipeline.fit_transform(X_aggregated, y_aggregated)

# Convert back to DataFrame for usability
try:
    final_feature_names = numerical_features_from_agg_to_woe_bin
    X_processed_final_df = pd.DataFrame(X_processed_final, columns=final_feature_names, index=X_aggregated.index)
except Exception as e:
    print(f"Could not reconstruct DataFrame with column names after pipeline: {e}. Showing as NumPy array.")
    X_processed_final_df = pd.DataFrame(X_processed_final)

print(f"\nFinal Model-Ready Features (X_processed_final) shape: {X_processed_final_df.shape}")
print("First 5 rows of the final transformed data:")
print(X_processed_final_df.head())
print("-" * 50)
