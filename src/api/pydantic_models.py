from pydantic import BaseModel, Field
from typing import List, Optional

# Define the schema for the incoming customer data for prediction
# These fields should match the features your model expects *after* aggregation and transformation.
# For simplicity, we'll define fields for the raw transaction data that the API *might* receive,
# and then the API's internal logic will handle the aggregation and transformation using the pipeline.

# A single transaction's data point
class TransactionData(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str # Will be parsed as datetime internally
    PricingStrategy: int
    # FraudResult is the target, not an input feature for prediction

# The request body for the /predict endpoint will be a list of these transactions
# This allows sending multiple transactions for a customer to be aggregated.
class PredictionRequest(BaseModel):
    # For simplicity, let's assume the API receives a list of raw transactions
    # that belong to *one* customer, or it can handle multiple customers
    # if you modify the aggregation logic.
    # For this setup, we'll assume a list of transactions for a *single* customer.
    customer_transactions: List[TransactionData] = Field(
        ..., description="List of raw transaction data for a customer."
    )

# The response body for the /predict endpoint
class PredictionResponse(BaseModel):
    customer_id: str = Field(..., description="The ID of the customer.")
    risk_probability: float = Field(..., description="The predicted probability of the customer being high-risk.")
    is_high_risk: int = Field(..., description="Binary prediction: 1 for high-risk, 0 for low-risk.")