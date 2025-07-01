from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.pyfunc
from typing import List, Dict, Any
import sys # Import sys

# Corrected: Import custom transformers explicitly.
# This makes them available in the main.py namespace.
from src.transformers import CustomerAggregator, RobustWoETransformer, generate_high_risk_target 

# Import your Pydantic models
from .pydantic_models import PredictionRequest, PredictionResponse


# --- FIX FOR PICKLE/JOBLIB AttributeError ---
# When joblib.load() is called in main.py, it might expect CustomAggregator to be in __main__.
# This class temporarily re-routes that expectation to where the class *actually* lives.
# This MUST be present in the file that loads the pickled object.
class _Restorer(object):
    """
    A helper class for joblib.load to correctly restore custom classes
    that might have been saved from a __main__ context or a different module path.
    """
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name

    def __reduce__(self):
        # This method is called by pickle to know how to reconstruct the object.
        # It tells pickle to look for 'class_name' in 'module_name'.
        return (getattr, (sys.modules[self.module_name], self.class_name))

# Temporarily patch sys.modules to allow joblib/pickle to find custom classes
# if they were pickled from a __main__ context.
# This mapping needs to be present *before* joblib.load() is called.
# You typically only need to patch the classes that are causing the AttributeError.
sys.modules['__main__'] = sys.modules['src.transformers']
sys.modules['src.data_processing'] = sys.modules['src.transformers'] # In case it was pickled with src.data_processing path


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting customer credit risk (high-risk proxy) based on transaction data.",
    version="1.0.0"
)

# --- Model and Pipeline Loading ---
# IMPORTANT: Adjust this path if you're NOT saving to the local models folder,
# or if your docker-compose.yml volume mount is different.
# For local Docker, this means `/app/credit-risk-model/models/`
# For Google Colab, this would be `/content/drive/My Drive/credit-risk-model/models/`
MODELS_BASE_PATH_IN_CONTAINER = "/app/credit-risk-model/models/" 

MLFLOW_MODEL_NAME = "CreditRiskClassifier"
MLFLOW_MODEL_VERSION = 1 # Update this to the version you want to serve

FE_PIPELINE_PATH = os.path.join(MODELS_BASE_PATH_IN_CONTAINER, 'feature_engineering_pipeline.pkl')
BEST_MODEL_PATH = os.path.join(MODELS_BASE_PATH_IN_CONTAINER, 'final_best_credit_risk_model.pkl')

ml_model = None
feature_pipeline = None

@app.on_event("startup")
async def load_model_and_pipeline():
    global ml_model, feature_pipeline
    
    print("Loading feature engineering pipeline...")
    try:
        # joblib.load now should find the classes because of the sys.modules patch
        feature_pipeline = joblib.load(FE_PIPELINE_PATH)
        print(f"Feature engineering pipeline loaded from {FE_PIPELINE_PATH}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Feature engineering pipeline not found at {FE_PIPELINE_PATH}. Ensure it's saved correctly.")
    except Exception as e:
        # Re-raising the error with more context
        raise HTTPException(status_code=500, detail=f"Error loading feature engineering pipeline: {e}. Check if custom transformers are defined/imported correctly in src/transformers.py and paths are valid.")

    print(f"Loading ML model '{MLFLOW_MODEL_NAME}' version {MLFLOW_MODEL_VERSION} from MLflow Model Registry...")
    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}"
        ml_model = mlflow.pyfunc.load_model(model_uri)
        print(f"ML model loaded from MLflow Registry: {model_uri}")
    except Exception as e:
        print(f"Error loading model from MLflow Registry: {e}. Attempting to load from local .pkl as fallback.")
        try:
            ml_model = joblib.load(BEST_MODEL_PATH)
            print(f"ML model loaded from local file: {BEST_MODEL_PATH}")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"ML model not found at {BEST_MODEL_PATH} and failed to load from MLflow. Ensure it's saved correctly.")
        except Exception as e_local:
            raise HTTPException(status_code=500, detail=f"Error loading model from local .pkl: {e_local}")

    if ml_model is None or feature_pipeline is None:
        raise HTTPException(status_code=500, detail="Failed to load model or feature pipeline during startup.")
    print("Model and pipeline loaded successfully. API is ready.")


# --- Helper function for preprocessing new data ---
def preprocess_new_data(raw_transactions: List[Dict[str, Any]]):
    if feature_pipeline is None:
        raise HTTPException(status_code=500, detail="Feature engineering pipeline not loaded.")

    raw_df = pd.DataFrame(raw_transactions)

    try:
        # Crucial Fix: Ensure TransactionStartTime is datetime BEFORE passing to pipeline
        # The pipeline's CustomerAggregator will also convert it, but explicitly doing it
        # here ensures consistency and handles potential issues if the pipeline expected it.
        # This explicitly applies to the DataFrame passed to `transform`.
        raw_df['TransactionStartTime'] = pd.to_datetime(raw_df['TransactionStartTime'], errors='coerce')
        
        # You might also want to handle any NaNs that result from 'coerce' if dates are malformed.
        # For example, drop rows with NaT if they are critical, or impute.
        # raw_df.dropna(subset=['TransactionStartTime'], inplace=True) # Example

        # The pipeline expects transaction-level data and outputs customer-level transformed features.
        processed_features_array = feature_pipeline.transform(raw_df)
        
        customer_id = raw_transactions[0]['CustomerId'] 

        if processed_features_array.ndim == 1:
            processed_features_array = processed_features_array.reshape(1, -1)

        return customer_id, processed_features_array

    except Exception as e:
        # Updated error message to include the specific column in question.
        raise HTTPException(status_code=500, detail=f"Error during feature preprocessing: {e}. Check input data format, especially 'TransactionStartTime'.")


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    """
    Accepts new customer transaction data and returns the predicted credit risk probability.
    """
    if ml_model is None or feature_pipeline is None:
        raise HTTPException(status_code=500, detail="Model or pipeline not loaded. API not ready.")

    customer_id, processed_features = preprocess_new_data(request.customer_transactions)

    risk_probability = ml_model.predict_proba(processed_features)[:, 1][0]
    is_high_risk = int(ml_model.predict(processed_features)[0])

    return PredictionResponse(
        customer_id=customer_id,
        risk_probability=risk_probability,
        is_high_risk=is_high_risk
    )