from pydantic import BaseModel
from joblib import load
from fastapi import FastAPI
app=FastAPI()
class PredictionInput(BaseModel):
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    month: float
    weekday: float
    hour: float
    minute_oftheday: float
    distance: float
    direction: float
model_path = "models/model.joblib"
model = load(model_path)
@app.get('/')
def home():
    return "working fine"
@app.post('/predict')
def predict(input_data:PredictionInput):
    features=[input_data.passenger_count,
    input_data.pickup_longitude,
    input_data.pickup_latitude,
    input_data.dropoff_longitude,
    input_data.dropoff_latitude,
    input_data.month,
    input_data.weekday,
    input_data.hour,
    input_data.minute_oftheday,
    input_data.distance,
    input_data.direction]

    prediction = model.predict([features])[0].item()
    # Return the prediction
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)