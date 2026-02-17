from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import logging

from app.inference import load_bundle, predict_from_dict

app = FastAPI(title="Manufacturing Prediction API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load bundle
BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "model" / "model_bundle_perfect.pkl"

try:
    model, scaler_X, scaler_Y, feature_columns = load_bundle(str(BUNDLE_PATH))
    logger.info(f" Model loaded successfully from {BUNDLE_PATH}")
    logger.info(f" Features: {len(feature_columns)}")
except Exception as e:
    logger.error(f" Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

class PredictRequest(BaseModel):
    # accept any key-values
    data: dict


@app.get("/")
def home():
    return {"message": "API is running!", "status": "ok"}


@app.get("/features")
def features():
    return {"feature_columns": feature_columns}


@app.post("/predict")
def predict(req: PredictRequest):
    prediction = predict_from_dict(model, scaler_X, scaler_Y, feature_columns, req.data)
    return {"prediction": prediction}

@app.get("/model-info")
def get_model_info():
    return {
        "r2": 0.905,  # Your model's R² score
        "rmse": 3.52,  # Root Mean Square Error
        "mae": 2.68,   # Mean Absolute Error  
        "mape": 11.6,  # Mean Absolute Percentage Error
        "version": "2.0",
        "features_count": len(feature_columns),
        "training_samples": 800,
        "test_samples": 200
    }

@app.get("/test-predictions")
def get_test_predictions():
    # You'll need to save test data and predictions during model training
    # For now, here's a mock structure:
    
    # In your training notebook, you should save:
    # test_results = {
    #     "actual": y_test_original.flatten().tolist(),
    #     "predicted": test_preds_original.flatten().tolist()
    # }
    
    # Mock data for demonstration:
    import random
    np.random.seed(42)
    
    actual_values = np.random.normal(35, 12, 50).clip(10, 65).tolist()
    predicted_values = [val + np.random.normal(0, 2.5) for val in actual_values]
    
    results = []
    for actual, pred in zip(actual_values, predicted_values):
        error = pred - actual
        error_pct = (error / actual) * 100
        results.append({
            "actual": round(actual, 2),
            "predicted": round(pred, 2), 
            "error": round(error, 2),
            "error_percentage": round(error_pct, 2)
        })
    
    return results

@app.get("/feature-importance")
def get_feature_importance():
    # Extract actual model weights
    weights = model.linear.weight.data.numpy().flatten()
    importance_dict = {}
    
    for i, feature in enumerate(feature_columns):
        importance_dict[feature] = float(weights[i])
    
    return importance_dict
