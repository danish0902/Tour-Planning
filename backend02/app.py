from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Import your model training and itinerary logic
from model import (
    train_ml_model,
    train_route_optimizer_model,
    train_location_ordering_model,
    generate_multi_day_itinerary
)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests (needed for React frontend)

# Load datasets
restaurants = pd.read_csv("goa_restaurants.csv")
tourist_places = pd.read_csv("goa_tourist_places.csv")

# Train models once on startup
bf_model, bf_scaler_X, bf_scaler_y = train_ml_model(restaurants, tourist_places)
route_model, route_scaler = train_route_optimizer_model(tourist_places)
loc_order_model, loc_order_scaler = train_location_ordering_model(tourist_places)

@app.route("/api/generate_itinerary", methods=["POST"])
def generate_itinerary():
    data = request.get_json()
    num_days = int(data.get("numDays", 3))
    must_visit = data.get("locations", [])

    itinerary = generate_multi_day_itinerary(
        num_days,
        must_visit,
        bf_model, bf_scaler_X, bf_scaler_y,
        route_model, route_scaler,
        loc_order_model, loc_order_scaler
    )

    return jsonify(itinerary)

if __name__ == "__main__":
    app.run(debug=True)
