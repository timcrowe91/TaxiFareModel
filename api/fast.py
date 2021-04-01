from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict_fare(key, pickup_datetime, pickup_longitude, pickup_latitude,
                    dropoff_longitude, dropoff_latitude, passenger_count):

    pickup_longitude = float(pickup_longitude)
    pickup_latitude = float(pickup_latitude)
    dropoff_longitude = float(dropoff_longitude)
    dropoff_latitude = float(dropoff_latitude)
    passenger_count = int(passenger_count)
    
    X_list = [key, pickup_datetime, pickup_longitude, pickup_latitude, \
                dropoff_longitude, dropoff_latitude, passenger_count]
    X_cols = ["key", "pickup_datetime", "pickup_longitude", "pickup_latitude", \
                    "dropoff_longitude", "dropoff_latitude", "passenger_count"]
    X_pred = pd.DataFrame(np.array(X_list).reshape(-1,len(X_list)), columns=X_cols)

    pipeline = joblib.load('model.joblib')

    prediction = pipeline.predict(X_pred)

    return {"pickup_datetime": pickup_datetime,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "passenger_count": passenger_count,
            "prediction": prediction[0]}


