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


# --- 2. Random Forest ---
print("--- Training Random Forest Models ---")

# Training a baseline Random Forest model
with mlflow.start_run(run_name="Random_Forest_Baseline"):
    print("Training Baseline Random Forest Model...")
    rf_baseline = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_baseline.fit(X_train, y_train) 
    print("Baseline Random Forest Model Trained.\n")

    mlflow.log_param("model_type", "Random Forest Baseline")
    mlflow.log_param("n_estimators", rf_baseline.n_estimators) # Default n_estimators
    mlflow.log_param("max_depth", rf_baseline.max_depth) # Default max_depth
    mlflow.log_param("class_weight", "balanced")

    y_pred_baseline = rf_baseline.predict(X_test)
    y_pred_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_baseline))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_baseline))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_baseline))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_baseline))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_baseline))
    mlflow.sklearn.log_model(rf_baseline, "baseline_random_forest_model")
    all_models["Random Forest (Baseline)"] = rf_baseline
    print(f"MLflow Run ID for Baseline RF: {mlflow.active_run().info.run_id}\n")


# Training a tuned Random Forest model
with mlflow.start_run(run_name="Random_Forest_Tuned"):
    print("Starting Hyperparameter Tuning for Random Forest...")
    rf_tuned = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_param_grid = {
        'n_estimators': [50, 100, 200], # Added 200 for more thorough tuning
        'max_depth': [5, 10, 20, None], # Added 20 for more depth options
        'min_samples_split': [2, 5, 10], # Added 10
        'min_samples_leaf': [1, 2, 4] # Added 4
    }

    grid_search_rf = GridSearchCV(
        estimator=rf_tuned,
        param_grid=rf_param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search_rf.fit(X_train, y_train)

    best_rf_model = grid_search_rf.best_estimator_
    print("Hyperparameter Tuning for Random Forest Complete.")
    print(f"Best Random Forest Hyperparameters: {grid_search_rf.best_params_}")
    print(f"Best Cross-Validation ROC AUC: {grid_search_rf.best_score_:.4f}\n")

    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_param("model_type", "Random Forest Tuned")
    mlflow.log_param("class_weight", "balanced")

    y_pred_tuned = best_rf_model.predict(X_test)
    y_pred_proba_tuned = best_rf_model.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_tuned))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_tuned))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_tuned))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_tuned))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_tuned))
    mlflow.sklearn.log_model(best_rf_model, "best_random_forest_model")
    all_models["Random Forest (Best Tuned)"] = best_rf_model
    print(f"MLflow Run ID for Tuned RF: {mlflow.active_run().info.run_id}\n")

# --- 3. Gradient Boosting Machines (GBM) ---
print("--- Training Gradient Boosting Models ---")

# Training a baseline Gradient Boosting model
with mlflow.start_run(run_name="GBM_Baseline"):
    print("Training Baseline Gradient Boosting model...")
    # GBM does not have a class_weight parameter directly,
    # but you can use `sample_weight` or adjust `scale_pos_weight` in libraries like XGBoost/LightGBM.
    # For scikit-learn GBM, you might rely on `stratify=y` in train_test_split.
    gbm_baseline = GradientBoostingClassifier(random_state=42)
    gbm_baseline.fit(X_train, y_train)
    print("Baseline Gradient Boosting model Trained.\n")

    mlflow.log_param("model_type", "GBM Baseline")
    mlflow.log_param("n_estimators", gbm_baseline.n_estimators)
    mlflow.log_param("learning_rate", gbm_baseline.learning_rate)
    mlflow.log_param("max_depth", gbm_baseline.max_depth)

    y_pred_baseline = gbm_baseline.predict(X_test)
    y_pred_proba_baseline = gbm_baseline.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_baseline))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_baseline))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_baseline))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_baseline))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_baseline))
    mlflow.sklearn.log_model(gbm_baseline, "baseline_gbm_model")
    all_models["Gradient Boosting (Baseline)"] = gbm_baseline
    print(f"MLflow Run ID for Baseline GBM: {mlflow.active_run().info.run_id}\n")


# Training a tuned Gradient Boosting model
with mlflow.start_run(run_name="GBM_Tuned"):
    print("Starting Hyperparameter Tuning for Gradient Boosting Classifier...")
    gbm_tuned = GradientBoostingClassifier(random_state=42) # Instantiate for tuning

    gbm_param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages
        'learning_rate': [0.01, 0.1, 0.2], # Contribution of each tree
        'max_depth': [3, 5, 7],           # Max depth of each tree
        'subsample': [0.8, 1.0],          # Fraction of samples for each tree
    }

    grid_search_gbm = GridSearchCV(
        estimator=gbm_tuned,
        param_grid=gbm_param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2 # Set to 1 or 2 for detailed progress output
    )

    grid_search_gbm.fit(X_train, y_train)

    best_gbm_model = grid_search_gbm.best_estimator_
    print("Hyperparameter Tuning for Gradient Boosting Classifier Complete.")
    print(f"Best Gradient Boosting Classifier Hyperparameters: {grid_search_gbm.best_params_}")
    print(f"Best Cross-Validation ROC AUC: {grid_search_gbm.best_score_:.4f}\n")

    mlflow.log_params(grid_search_gbm.best_params_)
    mlflow.log_param("model_type", "GBM Tuned")

    y_pred_tuned = best_gbm_model.predict(X_test)
    y_pred_proba_tuned = best_gbm_model.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba_tuned))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_tuned))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_tuned))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_tuned))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_tuned))
    mlflow.sklearn.log_model(best_gbm_model, "best_gbm_model")
    all_models["Gradient Boosting (Best Tuned)"] = best_gbm_model
    print(f"MLflow Run ID for Tuned GBM: {mlflow.active_run().info.run_id}\n")

# --- Evaluation of All Models ---
print("--- Overall Model Evaluation Summary ---")
for name, model in all_models.items():
    print(f"\nEvaluating: {name}")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# --- Identify and Register Best Model ---
# Based on your previous output, the Gradient Boosting (Baseline) model had the highest ROC AUC (0.9929).
best_model_run_id = "f7ecee4b8feb418cb4923481817344b5" # From MLflow output for GBM Baseline
best_model_artifact_path = "baseline_gbm_model"      # Artifact name used when logging GBM Baseline
best_model_name = "CreditRiskClassifier"             # Name for the model in MLflow Model Registry

if best_model_run_id == "REPLACE_WITH_BEST_RUN_ID" or best_model_artifact_path == "REPLACE_WITH_BEST_MODEL_ARTIFACT_PATH":
    print("\nSkipping MLflow model registration and final model saving.")
    print("Please update 'best_model_run_id' and 'best_model_artifact_path' after reviewing MLflow UI to register your best model.")
else:
    model_uri = f"runs:/{best_model_run_id}/{best_model_artifact_path}"
    
    try:
        # Register the best model in MLflow Model Registry
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=best_model_name,
            tags={"purpose": "Credit Risk PD", "version": "1.0", "data_version": "v1_rfm_proxy"}
            # description="Best performing model for high-risk customer identification, tuned with GridSearchCV."
        )
        print(f"\nModel registered as: {registered_model.name} (Version {registered_model.version})")
        # MLflow UI URL (adjust port if different, default is 5000)
        print(f"MLflow Model Registry URL: http://localhost:5000/#/models/{registered_model.name}/versions/{registered_model.version}")

        # Final saving of the best model to disk (alternative/supplement to MLflow artifacts)
        models_dir = os.path.join('./models/')
        os.makedirs(models_dir, exist_ok=True)
        
        # Corrected: Assign the correct best model instance for local saving
        if best_model_artifact_path == "best_logistic_regression_model":
            final_best_model_instance = best_log_reg_model
        elif best_model_artifact_path == "best_random_forest_model":
            final_best_model_instance = best_rf_model
        elif best_model_artifact_path == "best_gbm_model":
            final_best_model_instance = best_gbm_model
        elif best_model_artifact_path == "baseline_logistic_regression_model": 
            final_best_model_instance = log_reg_baseline
        elif best_model_artifact_path == "baseline_random_forest_model":
            final_best_model_instance = rf_baseline
        elif best_model_artifact_path == "baseline_gbm_model": 
            final_best_model_instance = gbm_baseline
        else:
            print(f"Warning: Could not identify specific best model instance from artifact path: {best_model_artifact_path}. Skipping local save of best model.")
            final_best_model_instance = None
        
        if final_best_model_instance:
            joblib.dump(final_best_model_instance, os.path.join(models_dir, 'final_best_credit_risk_model.pkl'))
            print(f"\nFinal best credit risk model saved to: {os.path.join(models_dir, 'final_best_credit_risk_model.pkl')}")


    except Exception as e:
        print(f"\nError during MLflow model registration or final saving: {e}")
        print("Please ensure MLflow Tracking Server is running (`mlflow ui`) and the 'best_model_run_id' and 'best_model_artifact_path' are correctly updated.")
        print("Also check that the model artifact exists at the specified URI.")