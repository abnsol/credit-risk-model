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

print("--- Step 4: Encode Categorical Variables (using WoE/IV) ---")

def iv_woe(data, target, bins=10, show_woe=False):
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns #Extract Column Names
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        # Handle continuous numerical features by binning using qcut (which is used if there are more than `bins` (default 10) unique values)
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > bins):
            try:
                binned_x = pd.qcut(data[ivars], bins,  duplicates='drop', precision=4) # Use qcut for quantile-based binning for numerical features with many unique values
                d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                print(f"Binning '{ivars}' using pd.qcut with {bins} bins.")
            except Exception as e:
                print(f"Warning: pd.qcut failed for '{ivars}'. Attempting cut with fixed width or skipping. Error: {e}")
                d0 = pd.DataFrame({'x': pd.cut(data[ivars], bins=bins, duplicates='drop', precision=4), 'y': data[target]})
        else:
            # For categorical or numerical with few unique values, treat as is
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
            print(f"Processing '{ivars}' as categorical/few unique values.")

        d0 = d0.astype({"x": str}) # Convert bins/categories to string for grouping
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        
        # Add small constant to avoid division by zero or log of zero
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events']) # WoE formula
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events']) # IV formula 
        d.insert(loc=0, column='Variable', value=ivars)
        
        total_iv = d['IV'].sum()
        print("Information value of " + ivars + " is " + str(round(total_iv,6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [total_iv]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

# Before applying WoE/IV, let's identify categorical and numerical features suitable for it
# Remember X_processed is now customer-level data
categorical_cols_for_woe = X_processed.select_dtypes(include='object').columns.tolist()
# Exclude 'CustomerId' as it's an ID, not a predictive feature to transform directly
if 'CustomerId' in categorical_cols_for_woe:
    categorical_cols_for_woe.remove('CustomerId')

# Numerical columns to be binned for WoE calculation
numerical_cols_for_woe = X_processed.select_dtypes(include=np.number).columns.tolist()

# Combine all features that will undergo WoE transformation
features_for_woe = categorical_cols_for_woe + numerical_cols_for_woe

print(f"\nFeatures selected for WoE/IV calculation: {features_for_woe}")

# Create a DataFrame for WoE/IV calculation (including features and target)
df_for_woe_iv = X_processed[features_for_woe].copy()
df_for_woe_iv[TARGET_COLUMN] = y_processed # Add the target column

# Calculate IV and WoE values for each feature
iv_table, woe_table = iv_woe(df_for_woe_iv, TARGET_COLUMN, bins=10, show_woe=False)

print("\n--- Information Value (IV) Table ---")
print(iv_table.sort_values(by='IV', ascending=False))

# --- Feature Selection based on IV ---
selected_features_iv = iv_table[iv_table['IV'] >= 0.02]['Variable'].tolist() # Keep features with at least weak predictive power
print(f"\nFeatures selected based on IV (>= 0.02): {selected_features_iv}")

# Create a WoE Transformer Class for use in scikit-learn Pipeline
class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, woe_map=None, fill_value=0):
        self.woe_map = woe_map # Stores WoE mapping for each feature
        self.fill_value = fill_value # Value to use for categories not seen during fit

    def fit(self, X, y):
        self.woe_map = {}
        for col in X.columns:
            # Need to re-bin continuous numerical columns if they were part of WoE calculation
            if X[col].dtype.kind in 'bifc' and col in numerical_cols_for_woe: # Check if it's a numerical col that was binned
                try:
                    # Use pd.qcut for numerical, same bins as in iv_woe if possible
                    binned_x = pd.qcut(X[col], 10,  duplicates='drop', precision=4)
                    temp_df = pd.DataFrame({'x': binned_x, 'y': y})
                except Exception:
                    temp_df = pd.DataFrame({'x': pd.cut(X[col], bins=10, duplicates='drop', precision=4), 'y': y})
            else:
                # For categorical or numerical with few unique values
                temp_df = pd.DataFrame({'x': X[col], 'y': y})
            
            temp_df['x'] = temp_df['x'].astype(str) # Ensure string for grouping

            d = temp_df.groupby("x", as_index=False, dropna=False).agg(
                {"y": ["count", "sum"]}
            )
            d.columns = ['Cutoff', 'N', 'Events']
            d['Non-Events'] = d['N'] - d['Events']
            
            # Avoid division by zero for proportions
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
            
            d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
            
            # Store the mapping from category/bin to WoE
            self.woe_map[col] = d.set_index('Cutoff')['WoE'].to_dict()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.woe_map: # Only transform columns for which we have a WoE map
                # Need to re-bin continuous numerical columns if they were part of WoE calculation
                if X_transformed[col].dtype.kind in 'bifc' and col in numerical_cols_for_woe:
                    try:
                         # Use pd.qcut for numerical, same bins as in iv_woe if possible
                        binned_x = pd.qcut(X_transformed[col], 10,  duplicates='drop', precision=4)
                        X_transformed[col] = binned_x.astype(str)
                    except Exception:
                        X_transformed[col] = pd.cut(X_transformed[col], bins=10, duplicates='drop', precision=4).astype(str)
                else:
                    X_transformed[col] = X_transformed[col].astype(str)

                # Map categories/bins to WoE values
                X_transformed[col] = X_transformed[col].map(self.woe_map[col]).fillna(self.fill_value)

        return X_transformed

# Prepare data for WoE transformation: select features identified by IV
# Note: WoE Transformer will handle binning/categorization internally based on its fit method
X_woe_transformed = X_processed[selected_features_iv].copy()

# Initialize and fit the WoE Transformer
woe_transformer = WoETransformer()
woe_transformer.fit(X_woe_transformed, y_processed)

# Transform the selected features
X_woe_transformed = woe_transformer.transform(X_woe_transformed)

print("\n--- Features after WoE Transformation ---")
print(X_woe_transformed.head())
print("-" * 50)

print("--- Step 5: Normalize/Standardize Numerical Features ---")

# Apply StandardScaler to these
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_woe_transformed)

# Convert back to DataFrame with original column names
X_scaled = pd.DataFrame(X_scaled, columns=X_woe_transformed.columns, index=X_woe_transformed.index)

print("\n--- Features after Standardization ---")
print(X_scaled.head())
print("-" * 50)