import pandas as pd
import numpy as np
from geopy.distance import geodesic
from difflib import get_close_matches
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and clean data
restaurants = pd.read_csv("goa_restaurants.csv")
tourist_places = pd.read_csv("goa_tourist_places.csv")
tourist_places = tourist_places[~tourist_places['Name'].str.contains('- Alt')]

class RouteOptimizer:
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def train_route_model(self, tourist_places, epochs=10):
        """Train neural network to predict optimal routes between places"""
        print(f"\nTraining route optimization model for {epochs} epochs...")
        
        # Create training data from all place pairs
        places = tourist_places[['Latitude', 'Longitude', 'Star Rating']].values
        X = []
        y = []
        
        for i in range(len(places)):
            for j in range(len(places)):
                if i != j:
                    # Input: both places' coordinates and ratings
                    X.append(np.concatenate([places[i], places[j]]))
                    # Output: 1 if i->j is better, 0 if j->i is better
                    # (We'll use distance + rating as criteria)
                    dist_i_j = geodesic(places[i][:2], places[j][:2]).km
                    dist_j_i = geodesic(places[j][:2], places[i][:2]).km
                    score_i_j = (places[j][2] / dist_i_j)  # Higher rating, shorter distance -> better
                    score_j_i = (places[i][2] / dist_j_i)
                    y.append(1 if score_i_j > score_j_i else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(6,)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            verbose=1
        )
        print("Model training complete!\n")
    
    def get_optimal_order(self, place1, place2):
        """Determine optimal visiting order using trained model"""
        # Prepare input features
        input_features = np.array([
            place1['Latitude'], place1['Longitude'], place1['Star Rating'],
            place2['Latitude'], place2['Longitude'], place2['Star Rating']
        ]).reshape(1, -1)
        
        # Predict which order is better
        prediction = self.model.predict(self.scaler.transform(input_features))[0][0]
        return (place1, place2) if prediction > 0.5 else (place2, place1)

def find_nearest_meal(location_coords, meal_type, restaurants, radius=0.2):
    """Find nearest restaurant serving specified meal"""
    # Filter by meal type and location
    filtered = restaurants[
        (restaurants[meal_type] == 'Yes') &
        (restaurants['Latitude'].between(location_coords[0]-radius, location_coords[0]+radius)) &
        (restaurants['Longitude'].between(location_coords[1]-radius, location_coords[1]+radius))
    ]
    
    if not filtered.empty:
        # Find the closest by distance
        filtered['distance'] = filtered.apply(
            lambda row: geodesic(location_coords, (row['Latitude'], row['Longitude'])).km, axis=1)
        return filtered.loc[filtered['distance'].idxmin()]
    return None

def generate_itinerary(place1, place2, restaurants, tourist_places, route_optimizer):
    """Generate complete detailed itinerary"""
    itinerary = []
    total_distance = 0
    prev_location = None
    
    # Determine optimal order using ML
    first_place, second_place = route_optimizer.get_optimal_order(place1, place2)
    print(f"\nOptimized route order: {first_place['Name']} → {second_place['Name']}")
    
# 1. Breakfast near first place (starting point - no distance)
breakfast = find_nearest_meal(
    (first_place['Latitude'], first_place['Longitude']), 
    'Breakfast', restaurants)

if breakfast is not None:
    itinerary.append({
        'Type': 'Breakfast Restaurant',
        'Name': breakfast['Name'],
        'Cuisine': breakfast['Cuisine Type'],
        'Rating': breakfast['Rating'],
        'Coordinates': (breakfast['Latitude'], breakfast['Longitude'])
        # No distance calculation or field
    })
    prev_coords = (breakfast['Latitude'], breakfast['Longitude'])
else:
    prev_coords = (first_place['Latitude'], first_place['Longitude'])

# 2. First tourist place (with distance from breakfast)
dist_to_first = geodesic(prev_coords, (first_place['Latitude'], first_place['Longitude'])).km

itinerary.append({
    'Type': 'Tourist Attraction',
    'Name': first_place['Name'],
    'Category': first_place['Type'],
    'Rating': first_place['Star Rating'],
    'Coordinates': (first_place['Latitude'], first_place['Longitude']),
    'Distance': f"{dist_to_first:.2f} km"  # First distance shown
})
total_distance += dist_to_first
    
    # 3. Lunch at midpoint
    midpoint = (
        (first_place['Latitude'] + second_place['Latitude']) / 2,
        (first_place['Longitude'] + second_place['Longitude']) / 2
    )
    lunch = find_nearest_meal(midpoint, 'Lunch', restaurants)
    if lunch is not None:
        lunch_dist = geodesic(prev_location, (lunch['Latitude'], lunch['Longitude'])).km
        itinerary.append({
            'Type': 'Lunch Restaurant',
            'Name': lunch['Name'],
            'Cuisine': lunch['Cuisine Type'],
            'Rating': lunch['Rating'],
            'Coordinates': (lunch['Latitude'], lunch['Longitude']),
            'Distance': lunch_dist
        })
        total_distance += lunch_dist
        prev_location = (lunch['Latitude'], lunch['Longitude'])
    
    # 4. Second tourist place
    dist_to_second = geodesic(prev_location, (second_place['Latitude'], second_place['Longitude'])).km
    itinerary.append({
        'Type': 'Tourist Attraction',
        'Name': second_place['Name'],
        'Category': second_place['Type'],
        'Rating': second_place['Star Rating'],
        'Coordinates': (second_place['Latitude'], second_place['Longitude']),
        'Distance': dist_to_second
    })
    total_distance += dist_to_second
    prev_location = (second_place['Latitude'], second_place['Longitude'])
    
    # 5. Dinner near second place
    dinner = find_nearest_meal(
        (second_place['Latitude'], second_place['Longitude']), 
        'Dinner', restaurants)
    
    if dinner is not None:
        dinner_dist = geodesic(prev_location, (dinner['Latitude'], dinner['Longitude'])).km
        itinerary.append({
            'Type': 'Dinner Restaurant',
            'Name': dinner['Name'],
            'Cuisine': dinner['Cuisine Type'],
            'Rating': dinner['Rating'],
            'Coordinates': (dinner['Latitude'], dinner['Longitude']),
            'Distance': dinner_dist
        })
        total_distance += dinner_dist
    
    return itinerary, total_distance

def main():
    # Initialize and train route optimizer
    route_optimizer = RouteOptimizer()
    route_optimizer.train_route_model(tourist_places, epochs=10)
    
    # Get user input for two places
    print("\nPlease enter two places you'd like to visit in Goa:")
    place1_name = input("First place to visit: ")
    place2_name = input("Second place to visit: ")
    
    # Find matching places
    def find_place(name):
        matches = get_close_matches(name, tourist_places['Name'].tolist(), n=1, cutoff=0.6)
        if not matches:
            raise ValueError(f"Could not find place matching '{name}'")
        return tourist_places[tourist_places['Name'] == matches[0]].iloc[0]
    
    place1 = find_place(place1_name)
    place2 = find_place(place2_name)
    
    # Generate itinerary
    itinerary, total_distance = generate_itinerary(place1, place2, restaurants, tourist_places, route_optimizer)
    
    # Print detailed itinerary
    print("\nOPTIMIZED GOA TOUR ITINERARY")
    print("="*60)
    for i, stop in enumerate(itinerary):
        print(f"\n{i+1}. {stop['Type'].upper()}")
        print(f"   Name: {stop['Name']}")
        if 'Cuisine' in stop:
            print(f"   Cuisine: {stop['Cuisine']}")
        else:
            print(f"   Category: {stop['Category']}")
        print(f"   Rating: {stop['Rating']}/5")
        print(f"   Coordinates: ({stop['Coordinates'][0]:.6f}, {stop['Coordinates'][1]:.6f})")
        if i > 0:
            print(f"   Distance from previous: {stop['Distance']:.2f} km")
    print("\n" + "="*60)
    print(f"TOTAL TRAVEL DISTANCE: {total_distance:.2f} km")
    print("="*60)

if __name__ == "__main__":
    main()