# --- import libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split,RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, classification_report
import mlflow 
import mlflow.sklearn 
import joblib
import os

# --- Load Data --- 
X = pd.read_csv('./data/processed/model_features.csv', index_col='CustomerId')
y = pd.read_csv('./data/processed/is_high_risk_target.csv', index_col='CustomerId', squeeze=True) # Use squeeze=True for Series
X, y = X.align(y, join='inner', axis=0)  # perfect alignment based on CustomerId

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}\n")

# Use StratifiedKFold for cross-validation on imbalanced data
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Set random_state for reproducibility

# Dictionary to store all models for evaluation
all_models = {}

# --- 1. Logistic Regression ---
print("--- Training Logistic Regression Models ---")

# Training a baseline Logistic Regression model
with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
    print("Training Baseline Logistic Regression Model...")
    log_reg_baseline = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    log_reg_baseline.fit(X_train, y_train) 
    print("Baseline Logistic Regression Model Trained.")

    # Log parameters and metrics for baseline
    mlflow.log_param("model_type", "Logistic Regression Baseline")
    mlflow.log_param("solver", log_reg_baseline.solver)
    mlflow.log_param("C", log_reg_baseline.C) # Default C
    mlflow.log_param("penalty", log_reg_baseline.penalty) # Default penalty
    mlflow.log_param("class_weight", "balanced")
    
    y_pred_baseline = log_reg_baseline.predict(X_test)
    y_pred_proba_baseline = log_reg_baseline.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_baseline))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_baseline))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_baseline))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_baseline))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_baseline))
    mlflow.sklearn.log_model(log_reg_baseline, "baseline_logistic_regression_model")
    all_models["Logistic Regression (Baseline)"] = log_reg_baseline
    print(f"MLflow Run ID for Baseline LR: {mlflow.active_run().info.run_id}\n")


# Training a tuned Logistic Regression model using GridSearchCV
with mlflow.start_run(run_name="Logistic_Regression_Tuned"):
    print("Starting Hyperparameter Tuning for Logistic Regression...")
    log_reg_tuned = LogisticRegression(random_state=42, class_weight='balanced') # Instantiate with class_weight
    log_reg_param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],  # Expanded grid for better tuning
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'] # 'liblinear' supports both l1 and l2 with 'C'
    }

    grid_search_log_reg = GridSearchCV(
        estimator=log_reg_tuned,
        param_grid=log_reg_param_grid,
        cv=cv_folds,
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1 
    )

    grid_search_log_reg.fit(X_train, y_train) 

    best_log_reg_model = grid_search_log_reg.best_estimator_
    print("Hyperparameter Tuning for Logistic Regression Complete.")
    print(f"Best Logistic Regression Hyperparameters: {grid_search_log_reg.best_params_}")
    print(f"Best Cross-Validation ROC AUC: {grid_search_log_reg.best_score_:.4f}\n")

    # Log parameters and metrics for tuned model
    mlflow.log_params(grid_search_log_reg.best_params_)
    mlflow.log_param("model_type", "Logistic Regression Tuned")
    mlflow.log_param("class_weight", "balanced") # Ensure this is logged if set in tuned model

    y_pred_tuned = best_log_reg_model.predict(X_test)
    y_pred_proba_tuned = best_log_reg_model.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_tuned))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_tuned))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_tuned))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_tuned))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_tuned))
    mlflow.sklearn.log_model(best_log_reg_model, "best_logistic_regression_model")
    all_models["Logistic Regression (Best Tuned)"] = best_log_reg_model
    print(f"MLflow Run ID for Tuned LR: {mlflow.active_run().info.run_id}\n")