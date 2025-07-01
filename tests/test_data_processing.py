import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # Used in RFM High Risk Labeler
# Import your custom classes and function from src.data_processing
from src.data_processing import CustomerAggregator, RobustWoETransformer, generate_high_risk_target 
from sklearn.cluster import KMeans # Needed for generate_high_risk_target fixture


# --- Fixtures for Sample Data ---

# A small, controlled raw data sample for testing
@pytest.fixture
def sample_raw_data():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'],
        'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        'AccountId': ['A1', 'A2', 'A1', 'A3', 'A2', 'A4', 'A1'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C2', 'C4', 'C1'], # C1 has 3 trans, C2 has 2, C3, C4 have 1
        'CurrencyCode': ['UGX']*7,
        'CountryCode': [256]*7,
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P2', 'P4', 'P1'],
        'ProductId': ['PR1', 'PR2', 'PR1', 'PR3', 'PR2', 'PR4', 'PR1'],
        'ProductCategory': ['airtime', 'data', 'airtime', 'financial_services', 'data', 'utility_bill', 'airtime'],
        'ChannelId': ['CH1', 'CH2', 'CH1', 'CH3', 'CH2', 'CH4', 'CH1'],
        'Amount': [100.0, 50.0, 150.0, 200.0, -10.0, 500.0, 75.0], # Includes a negative amount
        'Value': [100, 50, 150, 200, 10, 500, 75],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07']),
        'PricingStrategy': [1, 2, 1, 3, 2, 4, 1],
        'FraudResult': [0, 0, 1, 0, 0, 1, 0] # Example Fraud results for WoE/target
    }
    return pd.DataFrame(data)

# A fixture to provide the capped data, as your actual pipeline starts from df_capped
@pytest.fixture
def sample_capped_data(sample_raw_data):
    # Apply your capping logic here (simplified for fixture, typically done in data_processing.py)
    df_capped_fixture = sample_raw_data.copy()
    
    # Identify numerical columns for basic capping in fixture
    numerical_cols_fixture = df_capped_fixture.select_dtypes(include=np.number).columns.tolist()
    numerical_features_for_outliers_fixture = [col for col in numerical_cols_fixture if col not in ['FraudResult', 'CountryCode']]
    
    for col in numerical_features_for_outliers_fixture:
        Q1 = df_capped_fixture[col].quantile(0.25)
        Q3 = df_capped_fixture[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Corrected: Avoid inplace=True and use direct assignment for robustness
        df_capped_fixture[col] = np.where(df_capped_fixture[col] < lower_bound, lower_bound, df_capped_fixture[col])
        df_capped_fixture[col] = np.where(df_capped_fixture[col] > upper_bound, upper_bound, df_capped_fixture[col])
    return df_capped_fixture


# --- Unit Tests for generate_high_risk_target (Task 4 Proxy Target) ---
def test_generate_high_risk_target_output_shape(sample_capped_data):
    is_high_risk_series, rfm_df = generate_high_risk_target(sample_capped_data)
    # Assuming 4 unique customers in sample_capped_data (C1, C2, C3, C4)
    assert is_high_risk_series.shape == (4,)
    assert rfm_df.shape[0] == 4
    # RFM df has Recency, Frequency, Monetary, Cluster, is_high_risk = 5 columns
    assert rfm_df.shape[1] == 5 
    assert isinstance(is_high_risk_series, pd.Series)
    assert is_high_risk_series.name == 'is_high_risk'

def test_generate_high_risk_target_logic(sample_capped_data):
    # Corrected: Removed 'snapshot_date' as it's calculated internally
    is_high_risk_series, rfm_df = generate_high_risk_target(sample_capped_data)
    
    # Verify properties of the generated target
    assert is_high_risk_series.nunique() == 2 # Should be binary (0 and 1)
    assert is_high_risk_series.sum() > 0 # At least one high-risk customer (assuming some disengagement in small sample)
    assert (is_high_risk_series == 1).sum() < len(is_high_risk_series) # Not all customers are high-risk

    # --- Deeper check for specific cluster logic (optional, but good if behavior is very predictable) ---
    # To rigorously test the high-risk cluster identification, you'd calculate RFM, scale,
    # and then predict clusters using the *same random_state* as the function.
    # Then compare the characteristics of the *actual* identified high_risk_cluster_label
    # with what you expect for a disengaged group.
    # For this small sample, K-Means results can be very sensitive.
    # A more robust test here might be to mock KMeans or pre-define cluster assignments.
    # For now, asserting it's binary and not all 0s or 1s is a basic sanity check.


# --- Unit Tests for CustomerAggregator (Task 3 Feature Creation) ---
def test_customer_aggregator_output(sample_capped_data):
    aggregator = CustomerAggregator(customer_id_col='CustomerId', timestamp_col='TransactionStartTime')
    
    # We need to pass y to transform, even if its specific values aren't asserted in this test,
    # because the CustomerAggregator.transform expects it to align indexes.
    X_agg, y_agg_placeholder = aggregator.transform(sample_capped_data.drop(columns=['FraudResult']), sample_capped_data['FraudResult'])

    assert X_agg.shape == (4, 10) # 4 unique customers (C1, C2, C3, C4), 10 aggregate features
    assert list(X_agg.index) == ['C1', 'C2', 'C3', 'C4'] # Check CustomerId as index
    assert y_agg_placeholder.shape == (4,) # Target also aggregated and aligned

    # Check a few aggregate values for customer C1 from sample_raw_data
    # C1 transactions: Amount: [100.0, 150.0, 75.0]
    # Total: 325.0, Avg: 325/3 = 108.333..., Count: 3
    # last_transaction_hour for C1: 2023-01-07 00:00:00 (hour 0)
    
    # Using np.isclose for float comparisons
    assert np.isclose(X_agg.loc['C1', 'total_transaction_amount'], 325.0)
    assert np.isclose(X_agg.loc['C1', 'average_transaction_amount'], 325.0 / 3)
    assert X_agg.loc['C1', 'transaction_count'] == 3
    assert X_agg.loc['C1', 'last_transaction_hour'] == 0 
    assert X_agg.loc['C1', 'last_transaction_month'] == 1 # January
    assert X_agg.loc['C1', 'last_transaction_year'] == 2023 # 2023

def test_customer_aggregator_nan_handling(sample_raw_data):
    # Create a sample where some customers have only one transaction
    # (e.g., C1, C2, C3, C4 all become single-transaction customers here)
    single_trans_data = sample_raw_data.copy()
    # To guarantee single transactions per customer for testing NaN handling of STD:
    single_trans_data = single_trans_data.groupby('CustomerId').first().reset_index() # Take first trans for each unique CustomerId
    
    aggregator = CustomerAggregator(customer_id_col='CustomerId', timestamp_col='TransactionStartTime')
    X_agg, _ = aggregator.transform(single_trans_data.drop(columns=['FraudResult']), single_trans_data['FraudResult'])
    
    # Check that std_transaction_amount is 0 for these single-transaction customers
    assert (X_agg['std_transaction_amount'] == 0).all()
