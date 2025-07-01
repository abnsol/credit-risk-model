import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import os

# Suppress warnings for cleaner output during demonstration
import warnings
warnings.filterwarnings('ignore')

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


# --- Function for Proxy Target Variable Engineering (Task 4) ---
def generate_high_risk_target(df, customer_id_col='CustomerId', timestamp_col='TransactionStartTime', amount_col='Amount', n_clusters=3, random_state=42):
    """
    Generates the 'is_high_risk' proxy target variable based on RFM metrics and K-Means clustering.

    Args:
        df (pd.DataFrame): The input DataFrame (transaction-level, e.g., df_capped).
        customer_id_col (str): Name of the customer ID column.
        timestamp_col (str): Name of the transaction timestamp column.
        amount_col (str): Name of the transaction amount column.
        n_clusters (int): Number of clusters for K-Means.
        random_state (int): Random state for K-Means for reproducibility.

    Returns:
        pd.Series: A Series with CustomerId as index and 'is_high_risk' (0 or 1) as values.
        pd.DataFrame: A DataFrame with CustomerId as index, containing RFM features and Cluster.
    """
    print("\n--- Task 4: Generating Proxy Target Variable 'is_high_risk' ---")

    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    # 1. Define Snapshot Date: Take the latest transaction date in the entire dataset
    snapshot_date = df_copy[timestamp_col].max() + pd.Timedelta(days=1) # A day after the last transaction
    print(f"Snapshot Date for Recency calculation: {snapshot_date}")

    # 2. Calculate RFM Metrics
    rfm_df = df_copy.groupby(customer_id_col).agg(
        Recency=(timestamp_col, lambda date: (snapshot_date - date.max()).days), # Days since last transaction
        Frequency=(customer_id_col, 'count'), # Number of transactions (using customer_id_col for count is fine)
        Monetary=(amount_col, 'sum') # Sum of transaction amounts
    ).reset_index()

    print("\nRFM Metrics Calculated (first 5 rows):")
    print(rfm_df.head())

    # Handle potential issues with Monetary value (e.g., negative sums if amounts can be negative)
    # RFM Monetary is typically based on positive revenue. If 'Amount' can be negative (refunds)
    # 3. Pre-process (Scale) RFM features
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler_rfm = StandardScaler() # Create a new scaler specific to RFM
    rfm_scaled = scaler_rfm.fit_transform(rfm_features)

    # Convert scaled RFM back to DataFrame with customer IDs as index
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_features.columns, index=rfm_df[customer_id_col])
    
    print("\nRFM Features Scaled (first 5 rows of scaled data):")
    print(rfm_scaled_df.head())

    # 4. Cluster Customers using K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto') # Set random_state
    rfm_scaled_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Merge Cluster back to original RFM df for analysis
    rfm_df = rfm_df.set_index(customer_id_col)
    rfm_df['Cluster'] = rfm_scaled_df['Cluster']

    print(f"\nK-Means Clustering performed with {n_clusters} clusters.")
    print("Clusters assigned to RFM DataFrame (first 5 rows):")
    print(rfm_df.head())

    # 5. Define and Assign the "High-Risk" Label
    # Analyze cluster centroids/averages to identify the high-risk cluster
    cluster_analysis = rfm_df.groupby('Cluster').agg(
        avg_recency=('Recency', 'mean'),
        avg_frequency=('Frequency', 'mean'),
        avg_monetary=('Monetary', 'mean'),
        count=('Cluster', 'count') # Count items in the cluster
    ).sort_values(by=['avg_recency', 'avg_frequency', 'avg_monetary'], ascending=[False, True, True])

    print("\nCluster Analysis (Averages by Cluster - sorted for high-risk identification):")
    print(cluster_analysis)

    # The high-risk cluster is typically characterized by:
    # - High Recency (largest avg_recency - sorted descending)
    # - Low Frequency (smallest avg_frequency - sorted ascending)
    # - Low Monetary (smallest avg_monetary - sorted ascending)
    # So, we pick the first cluster in this sorted list.
    high_risk_cluster_label = cluster_analysis.index[0]
    print(f"\nIdentified 'High-Risk' Cluster Label: {high_risk_cluster_label}")

    # Create 'is_high_risk' column
    rfm_df['is_high_risk'] = np.where(rfm_df['Cluster'] == high_risk_cluster_label, 1, 0)

    print("\n'is_high_risk' label assigned to customers (first 5 rows of RFM data with target):")
    print(rfm_df[['Recency', 'Frequency', 'Monetary', 'is_high_risk']].head())
    print(f"Count of 'High-Risk' (1) customers: {rfm_df['is_high_risk'].sum()}")
    print(f"Count of 'Low-Risk' (0) customers: {len(rfm_df) - rfm_df['is_high_risk'].sum()}")
    
    # Ensure the returned Series has CustomerId as its index
    is_high_risk_series = rfm_df['is_high_risk']

    # Return the 'is_high_risk' Series and the RFM DataFrame (for potential saving or further use)
    return is_high_risk_series, rfm_df # rfm_df now contains CustomerId as index, RFM, Cluster, is_high_risk

# --- Main execution block when the script is run ---
if __name__ == "__main__":
    print("--- Running data_processing.py ---")

    # Load the capped data
    df = pd.read_csv('./data/processed/data_capped.csv')

    # Define the original transaction-level X (features)
    X_initial_transaction_level = df.drop(columns=['FraudResult']) # Original FraudResult will be dropped

    # --- Phase 1: Generate Proxy Target Variable (Task 4) ---
    # This will generate the NEW target variable 'is_high_risk'
    y_is_high_risk, rfm_data_for_analysis = generate_high_risk_target(
        df=df, 
        customer_id_col='CustomerId',
        timestamp_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42
    )
    print(f"\nShape of 'is_high_risk' target: {y_is_high_risk.shape}")
    print(f"Shape of RFM data with cluster and target: {rfm_data_for_analysis.shape}")
    print("-" * 50)

    # --- Phase 2: Main Feature Engineering Pipeline ---
    # We define the columns that will be WoE-transformed after aggregation.
    # These are the *output column names* from CustomerAggregator.
    numerical_features_from_agg_to_woe_bin = [
        'total_transaction_amount', 'average_transaction_amount',
        'std_transaction_amount', 'transaction_count',
        'average_pricing_strategy', 'distinct_product_categories',
        'last_transaction_hour', 'last_transaction_day',
        'last_transaction_month', 'last_transaction_year'
    ]

    feature_engineering_pipeline = Pipeline([
        ('customer_aggregator', CustomerAggregator(customer_id_col='CustomerId', timestamp_col='TransactionStartTime')),
        ('woe_transformer', RobustWoETransformer(
            numerical_cols_to_bin=numerical_features_from_agg_to_woe_bin,
            bins=10
        )),
        ('scaler', StandardScaler())
    ])

    print("\nFeature Engineering Pipeline for Credit Risk Model created successfully.")
    # print(feature_engineering_pipeline)

    # Fit and transform the data using the pipeline
    # X_initial_transaction_level is the transaction-level features.
    # y_is_high_risk is the NEW customer-level.
    # The pipeline's CustomerAggregator will handle the transformation of X_initial_transaction_level
    # to customer-level X, and also aggregate y_is_high_risk to align it.
    X_processed_final = feature_engineering_pipeline.fit_transform(X_initial_transaction_level, y_is_high_risk)

    # Convert back to DataFrame for usability
    try:
        final_feature_names = numerical_features_from_agg_to_woe_bin
        X_processed_final_df = pd.DataFrame(X_processed_final, columns=final_feature_names, index=y_is_high_risk.index)
        # The index for the final X should match the index of the final target (y_is_high_risk)
    except Exception as e:
        print(f"Could not reconstruct DataFrame with column names after pipeline: {e}. Showing as NumPy array.")
        X_processed_final_df = pd.DataFrame(X_processed_final)

    print(f"\nFinal Model-Ready Features (X_processed_final_df) shape: {X_processed_final_df.shape}")
    print("First 5 rows of the final transformed data:")
    print(X_processed_final_df.head())
    print("-" * 50)

    # --- Save Processed Data ---
    output_dir = './data/processed/'
    os.makedirs(output_dir, exist_ok=True)

    # Save features
    output_features_file = output_dir + 'model_features.csv'
    X_processed_final_df.to_csv(output_features_file, index=True)
    print(f"Processed features saved to: {output_features_file}")

    # Save target (y_is_high_risk)
    output_target_file = output_dir + 'is_high_risk_target.csv'
    y_is_high_risk.to_csv(output_target_file, index=True, header=True) 
    print(f"Aggregated 'is_high_risk' target saved to: {output_target_file}")