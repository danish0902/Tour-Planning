import pandas as pd
import numpy as np
from geopy.distance import geodesic
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load datasets
restaurants = pd.read_csv("goa_restaurants.csv")
tourist_places = pd.read_csv("goa_tourist_places.csv")

# Clean data
tourist_places = tourist_places[~tourist_places['Name'].str.contains('- Alt')]
restaurants = restaurants.drop_duplicates(subset=['Name'])

def find_closest_match(user_input, options):
    """Fuzzy match user input with options"""
    matches = get_close_matches(user_input, options, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_nearest_location(current_coords, candidates_df, visited, time_filter=None):
    """Find nearest unvisited location with optional time filter"""
    candidates = []
    for _, row in candidates_df.iterrows():
        if row['Name'] not in visited:
            if time_filter is None or row[time_filter[0]] == time_filter[1]:
                candidates.append((row['Name'], (row['Latitude'], row['Longitude']), row))
    
    if not candidates:
        return None, None, float('inf')
    
    # Calculate distances
    if current_coords is None:
        # For first location, return a random candidate (we'll optimize this with ML)
        import random
        random_idx = random.randint(0, len(candidates)-1)
        return candidates[random_idx][0], candidates[random_idx][2], 0.0
    
    distances = [geodesic(current_coords, coords).km for _, coords, _ in candidates]
    min_idx = np.argmin(distances)
    return candidates[min_idx][0], candidates[min_idx][2], distances[min_idx]

def train_route_optimizer_model(tourist_places):
    """Train ML model to optimize route between two tourist locations"""
    # Create training data: pairs of tourist places with their distances
    X = []
    y = []
    
    tourist_places_list = tourist_places.to_dict('records')
    
    # Generate all possible pairs of tourist places
    for i, place1 in enumerate(tourist_places_list):
        for j, place2 in enumerate(tourist_places_list):
            if i != j:
                # Features: coordinates and ratings of both places
                features = [
                    place1['Latitude'], place1['Longitude'], place1['Star Rating'],
                    place2['Latitude'], place2['Longitude'], place2['Star Rating']
                ]
                
                # Target: optimal order (0 if place1 should be visited first, 1 if place2 should be first)
                # For training, we'll use simple heuristic: higher rated place first
                target = 0 if place1['Star Rating'] >= place2['Star Rating'] else 1
                
                X.append(features)
                y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Build model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training ML model for route optimization...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=10,
        verbose=1
    )
    
    return model, scaler_X

def train_ml_model(restaurants, tourist_places):
    """Train ML model to predict optimal breakfast locations"""
    # Create training data: (tourist_lat, tourist_lon) -> (restaurant_lat, restaurant_lon)
    X = []
    y = []
    
    # We'll train on the assumption that good breakfast places are near popular tourist spots
    for _, place in tourist_places.iterrows():
        for _, rest in restaurants[restaurants['Breakfast'] == 'Yes'].iterrows():
            X.append([place['Latitude'], place['Longitude'], place['Star Rating']])
            y.append([rest['Latitude'], rest['Longitude'], rest['Rating']])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(64, activation='relu'),
        Dense(3)
    ])
    
    model.compile(optimizer=Adam(0.001), loss='mse')
    
    print("Training ML model for breakfast location optimization...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, scaler_y.transform(y_test)),
        epochs=10,  # Changed to 10 epochs as requested
        verbose=1
    )
    
    return model, scaler_X, scaler_y

def predict_breakfast_location(model, scaler_X, scaler_y, tourist_place):
    """Use trained model to predict optimal breakfast location"""
    # Prepare input
    X = np.array([[tourist_place['Latitude'], tourist_place['Longitude'], tourist_place['Star Rating']]])
    X_scaled = scaler_X.transform(X)
    
    # Predict
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Find closest actual breakfast place to prediction
    breakfast_options = restaurants[restaurants['Breakfast'] == 'Yes']
    min_dist = float('inf')
    best_rest = None
    
    for _, rest in breakfast_options.iterrows():
        dist = geodesic((y_pred[0][0], y_pred[0][1]), (rest['Latitude'], rest['Longitude'])).km
        if dist < min_dist:
            min_dist = dist
            best_rest = rest
    
    return best_rest['Name'], best_rest, 0.0

def optimize_two_locations_order(route_model, route_scaler, loc1_data, loc2_data):
    """Use ML model to decide the optimal order of two locations"""
    # Prepare input
    features = [
        loc1_data['Latitude'], loc1_data['Longitude'], loc1_data['Star Rating'],
        loc2_data['Latitude'], loc2_data['Longitude'], loc2_data['Star Rating']
    ]
    X = np.array([features])
    X_scaled = route_scaler.transform(X)
    
    # Predict optimal order
    prediction = route_model.predict(X_scaled)[0][0]
    
    # Return ordered locations
    if prediction < 0.5:
        return loc1_data, loc2_data
    else:
        return loc2_data, loc1_data

def generate_itinerary(location1, location2, bf_model, bf_scaler_X, bf_scaler_y, 
                      route_model, route_scaler):
    """Generate optimized itinerary using ML for breakfast selection and route optimization"""
    itinerary = []
    distances = []
    visited = set()
    
    # Get location data
    loc1_data = tourist_places[tourist_places['Name'] == location1].iloc[0]
    loc2_data = tourist_places[tourist_places['Name'] == location2].iloc[0]
    
    # Determine optimal order of the two locations using ML
    first_loc, second_loc = optimize_two_locations_order(route_model, route_scaler, loc1_data, loc2_data)
    
    # 1. Find breakfast restaurant using ML prediction based on first location
    breakfast_name, breakfast_data, breakfast_dist = predict_breakfast_location(
        bf_model, bf_scaler_X, bf_scaler_y, first_loc)
    
    itinerary.append({
        'Type': 'Breakfast Restaurant',
        'Name': breakfast_name,
        'Cuisine': breakfast_data['Cuisine Type'],
        'Rating': breakfast_data['Rating'],
        'Latitude': breakfast_data['Latitude'],
        'Longitude': breakfast_data['Longitude']
    })
    distances.append(None)  # No distance for starting location as requested
    visited.add(breakfast_name)
    current_coords = (breakfast_data['Latitude'], breakfast_data['Longitude'])
    
    # 2. Go to first selected tourist place
    dist_to_first = geodesic(current_coords, (first_loc['Latitude'], first_loc['Longitude'])).km
    
    itinerary.append({
        'Type': 'Tourist Attraction (First Selected)',
        'Name': first_loc['Name'],
        'Category': first_loc['Type'],
        'Rating': first_loc['Star Rating'],
        'Latitude': first_loc['Latitude'],
        'Longitude': first_loc['Longitude']
    })
    distances.append(dist_to_first)
    visited.add(first_loc['Name'])
    current_coords = (first_loc['Latitude'], first_loc['Longitude'])
    
    # 3. Find lunch restaurant (must have Lunch=Yes)
    lunch_name, lunch_data, lunch_dist = get_nearest_location(
        current_coords,
        restaurants,
        visited,
        time_filter=('Lunch', 'Yes')
    )
    
    if not lunch_name:
        print("No lunch restaurants found!")
        return None, None
    
    itinerary.append({
        'Type': 'Lunch Restaurant',
        'Name': lunch_name,
        'Cuisine': lunch_data['Cuisine Type'],
        'Rating': lunch_data['Rating'],
        'Latitude': lunch_data['Latitude'],
        'Longitude': lunch_data['Longitude']
    })
    distances.append(lunch_dist)
    visited.add(lunch_name)
    current_coords = (lunch_data['Latitude'], lunch_data['Longitude'])
    
    # 4. Go to second selected tourist place
    dist_to_second = geodesic(current_coords, (second_loc['Latitude'], second_loc['Longitude'])).km
    
    itinerary.append({
        'Type': 'Tourist Attraction (Second Selected)',
        'Name': second_loc['Name'],
        'Category': second_loc['Type'],
        'Rating': second_loc['Star Rating'],
        'Latitude': second_loc['Latitude'],
        'Longitude': second_loc['Longitude']
    })
    distances.append(dist_to_second)
    visited.add(second_loc['Name'])
    current_coords = (second_loc['Latitude'], second_loc['Longitude'])
    
    # 5. Find another tourist attraction
    tourist_name, tourist_data, tourist_dist = get_nearest_location(
        current_coords,
        tourist_places[~tourist_places['Name'].isin(visited)],
        visited
    )
    
    if tourist_name:  # This one is optional
        itinerary.append({
            'Type': 'Tourist Attraction',
            'Name': tourist_name,
            'Category': tourist_data['Type'],
            'Rating': tourist_data['Star Rating'],
            'Latitude': tourist_data['Latitude'],
            'Longitude': tourist_data['Longitude']
        })
        distances.append(tourist_dist)
        visited.add(tourist_name)
        current_coords = (tourist_data['Latitude'], tourist_data['Longitude'])
    
    # 6. Find dinner restaurant (must have Dinner=Yes)
    dinner_name, dinner_data, dinner_dist = get_nearest_location(
        current_coords,
        restaurants,
        visited,
        time_filter=('Dinner', 'Yes')
    )
    
    if dinner_name:
        itinerary.append({
            'Type': 'Dinner Restaurant',
            'Name': dinner_name,
            'Cuisine': dinner_data['Cuisine Type'],
            'Rating': dinner_data['Rating'],
            'Latitude': dinner_data['Latitude'],
            'Longitude': dinner_data['Longitude']
        })
        distances.append(dinner_dist)
    
    return itinerary, distances

def print_itinerary(itinerary, distances):
    """Print formatted itinerary"""
    print("\nOPTIMIZED GOA TOUR ITINERARY")
    print("="*50)
    total_distance = 0
    
    for i, stop in enumerate(itinerary):
        print(f"\n{i+1}. {stop['Type'].upper()}")
        print(f"   Name: {stop['Name']}")
        
        if 'Cuisine' in stop:
            print(f"   Cuisine: {stop['Cuisine']}")
        else:
            print(f"   Category: {stop['Category']}")
            
        print(f"   Rating: {stop['Rating']}/5")
        print(f"   Coordinates: ({stop['Latitude']:.6f}, {stop['Longitude']:.6f})")
        
        # Only show distance if not None (skipping for first location)
        if distances[i] is not None:
            print(f"   Distance from previous: {distances[i]:.2f} km")
            total_distance += distances[i]
    
    print("\n" + "="*50)
    print(f"TOTAL TRAVEL DISTANCE: {total_distance:.2f} km")
    print("="*50)

# Main program
if __name__ == "__main__":
    print("Goa Tour Itinerary Planner")
    
    # Train ML models
    bf_model, bf_scaler_X, bf_scaler_y = train_ml_model(restaurants, tourist_places)
    route_model, route_scaler = train_route_optimizer_model(tourist_places)
    
    print("\nAvailable Tourist Attractions:")
    print(tourist_places['Name'].head(10).to_string(index=False))  # Show sample
    
    while True:
        print("\nEnter two tourist attractions to visit (or 'quit' to exit):")
        user_input1 = input("First location: ")
        if user_input1.lower() == 'quit':
            break
            
        user_input2 = input("Second location: ")
        if user_input2.lower() == 'quit':
            break
            
        location1 = find_closest_match(user_input1, tourist_places['Name'].tolist())
        location2 = find_closest_match(user_input2, tourist_places['Name'].tolist())
        
        if not location1:
            print(f"First location '{user_input1}' not found. Please try again.")
            continue
            
        if not location2:
            print(f"Second location '{user_input2}' not found. Please try again.")
            continue
            
        if location1 == location2:
            print("Please select two different locations.")
            continue
            
        print(f"\nGenerating optimized itinerary including {location1} and {location2}...")
        itinerary, distances = generate_itinerary(
            location1, location2, 
            bf_model, bf_scaler_X, bf_scaler_y,
            route_model, route_scaler
        )
        
        if itinerary:
            print_itinerary(itinerary, distances)
        else:
            print("Could not generate itinerary. Please try different locations.")