import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Corrected: Import custom transformers and function from the new 'src.transformers' module
from src.transformers import CustomerAggregator, RobustWoETransformer, generate_high_risk_target 

warnings.filterwarnings('ignore')

# --- Main execution block when the script is run ---
if __name__ == "__main__":
    print("--- Running data_processing.py ---")

    # Load the capped data
    df = pd.read_csv('./data/processed/data_capped.csv')

    # Define the original transaction-level X (features)
    X_initial_transaction_level = df.drop(columns=['FraudResult']) # Original FraudResult will be dropped

    # --- Phase 1: Generate Proxy Target Variable (Task 4) ---
    y_is_high_risk, rfm_data_for_analysis = generate_high_risk_target(
        df=df, # df should be df_capped from your previous work
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
    # print(feature_engineering_pipeline) # Uncomment for pipeline structure output

    X_processed_final = feature_engineering_pipeline.fit_transform(X_initial_transaction_level, y_is_high_risk)

    try:
        final_feature_names = numerical_features_from_agg_to_woe_bin
        X_processed_final_df = pd.DataFrame(X_processed_final, columns=final_feature_names, index=y_is_high_risk.index)
    except Exception as e:
        print(f"Could not reconstruct DataFrame with column names after pipeline: {e}. Showing as NumPy array.")
        X_processed_final_df = pd.DataFrame(X_processed_final)

    print(f"\nFinal Model-Ready Features (X_processed_final_df) shape: {X_processed_final_df.shape}")
    print("First 5 rows of the final transformed data:")
    print(X_processed_final_df.head())
    print("-" * 50)

    # --- Save Processed Data ---
    output_dir = './data/processed/'
    models_dir = './models/' 
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save features
    output_features_file = os.path.join(output_dir, 'model_features.csv')
    X_processed_final_df.to_csv(output_features_file, index=True)
    print(f"Processed features saved to: {output_features_file}")

    # Save the fitted feature engineering pipeline
    joblib.dump(feature_engineering_pipeline, os.path.join(models_dir, 'feature_engineering_pipeline.pkl'))
    print(f"Fitted feature engineering pipeline saved to: {os.path.join(models_dir, 'feature_engineering_pipeline.pkl')}")

    # Save target (y_is_high_risk)
    output_target_file = os.path.join(output_dir, 'is_high_risk_target.csv')
    y_is_high_risk.to_csv(output_target_file, index=True, header=True) 
    print(f"Aggregated 'is_high_risk' target saved to: {output_target_file}")

    print("\nAll data processing (Task 2, Task 4, and remaining Task 3) completed and data saved.")
    print("You can now proceed with model training (Task 5, likely in train.py).")