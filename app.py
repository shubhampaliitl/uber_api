import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import os
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from geopy.geocoders import Nominatim
import random
from sklearn.metrics import r2_score

app = FastAPI()
pickle_in = open("uber.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get("/")
def index():
    return {"message": "Hello, stranger"}


# @app.get("/{pickup location}")
# def get_locn(p_location: str):
#     return {"PickUp Location is :": f"{p_location}"}


# @app.get("/{Dropoff location}")
# def get_locn(d_location: str):
#     return {"Dropoff Location is ": f"{d_location}"}


def harversine(long1, long2, lat1, lat2):
    long1, long2, lat1, lat2 = map(np.radians, [long1, long2, lat1, lat2])
    diff_long = long2 - long1
    diff_lat = lat2 - lat1
    km = (
        2
        * 6371
        * np.arcsin(
            np.sqrt(
                np.sin(diff_lat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(diff_long / 2.0) ** 2
            )
        )
    )
    return km


# @app.post("/predict")
def predict_fare(pickup_location, dropoff_loctaion):
    geolocator = Nominatim(user_agent="MyApp")
    location_pickup = geolocator.geocode(pickup_location)
    location_dropoff = geolocator.geocode(dropoff_loctaion)

    location_pickup_lat = location_pickup.latitude
    location_pickup_long = location_pickup.longitude
    location_dropoff_long = location_dropoff.longitude
    location_dropoff_lat = location_dropoff.latitude
    prediction = classifier.predict(
        [
            [
                harversine(
                    location_pickup_long,
                    location_dropoff_long,
                    location_pickup_lat,
                    location_dropoff_lat,
                )
            ]
        ]
    )
    # prediction = random.randint(int((prediction * 83) / 4), 3000)
    prediction = (int(prediction * 83.12)) / 4
    return {"fare": prediction}


@app.get("/price/")
async def read_item(pickup_location, dropoff_location):
    return predict_fare(pickup_location, dropoff_location)


if __name__ == "__main__":
     # Use the port provided by Render
    port = int(os.environ.get("PORT", 8000))
    # Listen on all network interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)
