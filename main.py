from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
model = joblib.load("forest_fire_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")
print(f" Loaded model with features: {feature_order}")
print("Backend scaler n_features_in_:", scaler.n_features_in_)
app = FastAPI(
    title="Algerian Forest Fire FWI Predictor API",
    version="1.0.0"
)
model = joblib.load("forest_fire_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")
print(f"Loaded model with features: {feature_order}")
class FireFeatures(BaseModel):
    Temperature: float
    RH: float
    Ws: float
    Rain: float
    FFMC: float
    DMC: float
    DC: float
    ISI: float
    BUI: float
@app.get("/")
def root():
    return {
        "message": "FWI Predictor API running!",
        "features_required": feature_order
    }
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}
@app.post("/predict")
def predict_fwi(features: FireFeatures):
    try:
        data_dict = features.dict()
        print("Incoming data:", data_dict)
        print("Feature order:", feature_order)
        x = np.array([[data_dict[col] for col in feature_order]])
        print("x shape:", x.shape)
        x_scaled = scaler.transform(x)
        print("x_scaled shape:", x_scaled.shape)
        fwi_prediction = model.predict(x_scaled)[0]
        print("prediction:", fwi_prediction)
        return {
            "FWI_prediction": float(fwi_prediction),
            "status": "success"
        }
    except Exception as e:
        print("ERROR in /predict:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
