import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Custom Transformer: Customer Aggregator & Feature Extractor ---
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', timestamp_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        # --- CRITICAL FIX HERE ---
        # 1. Ensure the column exists to avoid KeyError
        if self.timestamp_col not in X_copy.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in input DataFrame.")
            
        # 2. Robustly convert to datetime, handling errors
        X_copy[self.timestamp_col] = pd.to_datetime(X_copy[self.timestamp_col], errors='coerce')
        
        # 3. Handle NaT values that might result from 'errors=coerce'
        # If a transaction has an invalid date, it might cause issues downstream.
        # For aggregation, typically you'd want to drop such transactions or handle them specifically.
        # For this context, let's drop rows where TransactionStartTime couldn't be parsed.
        # This will prevent errors in .max().hour etc. if a customer has only invalid dates.
        original_rows = X_copy.shape[0]
        X_copy.dropna(subset=[self.timestamp_col], inplace=True)
        if X_copy.shape[0] < original_rows:
            print(f"Warning: Dropped {original_rows - X_copy.shape[0]} rows due to invalid '{self.timestamp_col}' format during aggregation.")
        
        # 4. Handle cases where a CustomerId might disappear entirely after dropping NaT
        # if a customer has *only* invalid dates. This is implicitly handled by groupby.

        original_customer_ids = X_copy[self.customer_id_col]

        agg_df = X_copy.groupby(self.customer_id_col).agg(**{
            'total_transaction_amount': ('Amount', 'sum'),
            'average_transaction_amount': ('Amount', 'mean'),
            'std_transaction_amount': ('Amount', 'std'),
            'transaction_count': ('TransactionId', 'count'),
            'average_pricing_strategy': ('PricingStrategy', 'mean'),
            'distinct_product_categories': ('ProductCategory', 'nunique'),
            # The lambda functions now operate on a Series guaranteed to be datetime
            'last_transaction_hour': (self.timestamp_col, lambda x: x.max().hour if not x.empty else np.nan),
            'last_transaction_day': (self.timestamp_col, lambda x: x.max().day if not x.empty else np.nan),
            'last_transaction_month': (self.timestamp_col, lambda x: x.max().month if not x.empty else np.nan),
            'last_transaction_year': (self.timestamp_col, lambda x: x.max().year if not x.empty else np.nan)
        }).reset_index()

        # Corrected: Avoid inplace=True and use direct assignment for robustness
        agg_df['std_transaction_amount'] = agg_df['std_transaction_amount'].fillna(0)
        # Also fill NaNs for the new datetime extracted features if a customer had no valid dates
        agg_df[['last_transaction_hour', 'last_transaction_day', 'last_transaction_month', 'last_transaction_year']] = \
            agg_df[['last_transaction_hour', 'last_transaction_day', 'last_transaction_month', 'last_transaction_year']].fillna(0) # Or another suitable imputation
            
        agg_df = agg_df.set_index(self.customer_id_col)
        
        if y is not None:
            y_df = pd.DataFrame({'CustomerId': original_customer_ids, 'Target': y})
            y_aggregated = y_df.groupby('CustomerId')['Target'].max() 
            return agg_df, y_aggregated 
        else:
            return agg_df

# --- Custom Transformer: Robust WoE Transformer ---
class RobustWoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols_to_bin, bins=10, fill_value=0.0):
        self.numerical_cols_to_bin = numerical_cols_to_bin
        self.bins = bins
        self.fill_value = fill_value
        self.woe_maps = {}
        self.binner_info = {}

    def fit(self, X, y):
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

            d0 = pd.DataFrame({'x': temp_series, 'y': y_series})
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

# --- Function for Proxy Target Variable Engineering (Task 4) ---
def generate_high_risk_target(df, customer_id_col='CustomerId', timestamp_col='TransactionStartTime', amount_col='Amount', n_clusters=3, random_state=42):
    # This function definition remains the same as previously provided
    # I'll omit its full content here for brevity but it goes into src/transformers.py
    # as a standalone function.
    # Make sure to update the fillna inplace=True calls within this function as well.

    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    snapshot_date = df_copy[timestamp_col].max() + pd.Timedelta(days=1)

    rfm_df = df_copy.groupby(customer_id_col).agg(
        Recency=(timestamp_col, lambda date: (snapshot_date - date.max()).days),
        Frequency=(customer_id_col, 'count'),
        Monetary=(amount_col, 'sum')
    ).reset_index()

    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler_rfm = StandardScaler()
    rfm_scaled = scaler_rfm.fit_transform(rfm_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_features.columns, index=rfm_df[customer_id_col])
    rfm_scaled_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    rfm_df = rfm_df.set_index(customer_id_col)
    rfm_df['Cluster'] = rfm_scaled_df['Cluster']

    cluster_analysis = rfm_df.groupby('Cluster').agg(
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        count=('Cluster', 'count')
    ).sort_values(by=['avg_recency', 'avg_frequency', 'avg_monetary'], ascending=[False, True, True])

    high_risk_cluster_label = cluster_analysis.index[0]
    rfm_df['is_high_risk'] = np.where(rfm_df['Cluster'] == high_risk_cluster_label, 1, 0)

    is_high_risk_series = rfm_df['is_high_risk']
    return is_high_risk_series, rfm_df