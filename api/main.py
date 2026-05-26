from fastapi import FastAPI
from api.schemas import PredictionInput, PredictionOutput

# Initialize the FastAPI application
app = FastAPI(
    title="NLP Toxicity Classifier API",
    description="API for detecting toxicity, racism, and hate speech in text comments.",
    version="1.0.0"
)

# 1. Health check endpoint to verify the API status
@app.get("/health", tags=["Diagnostic"])
def health_check():
    """
    Returns the current health status of the API.
    """
    return {"status": "healthy", "message": "API is up and running successfully."}

# 2. Main prediction endpoint with mock response for simulation
@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_toxicity(payload: PredictionInput):
    """
    Analyzes the input text and returns simulated classification metrics.
    """
    # TODO: Integrate the actual NLP model trained by Iris once it is ready
    # For now, we return a hardcoded mock response matching the Pydantic schema
    mock_response = {
        "is_toxic": False,
        "is_racist": False,
        "is_hate_speech": False
    }
    
    return mock_response