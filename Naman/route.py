import pandas as pd
import numpy as np
from geopy.distance import geodesic
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import itertools
from datetime import datetime, timedelta
import random

# Load datasets
restaurants = pd.read_csv("goa_restaurants.csv")
tourist_places = pd.read_csv("goa_tourist_places.csv")

# Clean data
tourist_places = tourist_places[~tourist_places['Name'].str.contains('- Alt')]
restaurants = restaurants.drop_duplicates(subset=['Name'])

# Define time slots for activities
time_slots = {
    'breakfast': ('08:00', '09:00'),
    'location1': ('09:00', '12:00'),
    'lunch': ('12:00', '14:00'),
    'location2': ('14:00', '17:00'),
    'location3': ('17:00', '20:00'),
    'dinner': ('20:00', '22:00')
}

def time_to_minutes(time_str):
    """Convert time string to minutes from midnight"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except:
        return 9 * 60  # Default to 9:00 AM if invalid format

def is_open_during_slot(place, slot_start, slot_end):
    """Check if a place is open during a given time slot"""
    # Convert time strings to minutes for easier comparison
    slot_start_min = time_to_minutes(slot_start)
    slot_end_min = time_to_minutes(slot_end)
    
    # Check if place has opening and closing times
    if 'Opening Time' not in place or 'Closing Time' not in place:
        return True  # If no time data, assume always open
    
    if pd.isna(place['Opening Time']) or pd.isna(place['Closing Time']):
        return True  # If missing time data, assume always open
    
    # Get opening and closing times in minutes
    try:
        open_time_min = time_to_minutes(place['Opening Time'])
        close_time_min = time_to_minutes(place['Closing Time'])
    except:
        return True  # If time format is invalid, assume always open
    
    # Check if place is open during slot
    # If closing time is less than opening time, it means it closes after midnight
    if close_time_min < open_time_min:
        close_time_min += 24 * 60
    
    # Place is open if slot start is after opening and slot end is before closing
    return (open_time_min <= slot_start_min and slot_end_min <= close_time_min) or \
           (open_time_min <= slot_start_min < close_time_min) or \
           (open_time_min < slot_end_min <= close_time_min)

def find_closest_match(user_input, options):
    """Fuzzy match user input with options"""
    matches = get_close_matches(user_input, options, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_nearby_restaurants(coords, rest_df, visited, max_distance=2.5, time_slot=None, time_filter=None, max_results=3):
    """Find nearby restaurants within specified distance"""
    nearby = []
    
    for _, row in rest_df.iterrows():
        if row['Name'] not in visited:
            # Check time filter (breakfast, lunch, dinner)
            time_filter_ok = True
            if time_filter is not None:
                time_filter_ok = row[time_filter[0]] == time_filter[1]
            
            # Check if place is open during the time slot
            time_slot_ok = True
            if time_slot is not None:
                slot_start, slot_end = time_slots[time_slot]
                time_slot_ok = is_open_during_slot(row, slot_start, slot_end)
            
            if time_filter_ok and time_slot_ok:
                rest_coords = (row['Latitude'], row['Longitude'])
                distance = geodesic(coords, rest_coords).km
                
                if distance <= max_distance:
                    nearby.append((row['Name'], row['Cuisine Type'], distance))
    
    # Sort by distance and return at most max_results
    nearby.sort(key=lambda x: x[2])
    return nearby[:max_results]

def get_nearest_location(current_coords, candidates_df, visited, time_slot=None, time_filter=None):
    """Find nearest unvisited location with optional time filter and time slot check"""
    candidates = []
    
    for _, row in candidates_df.iterrows():
        if row['Name'] not in visited:
            # Check time filter (for restaurants: breakfast, lunch, dinner)
            time_filter_ok = True
            if time_filter is not None:
                time_filter_ok = row[time_filter[0]] == time_filter[1]
            
            # Check if place is open during the time slot
            time_slot_ok = True
            if time_slot is not None:
                slot_start, slot_end = time_slots[time_slot]
                time_slot_ok = is_open_during_slot(row, slot_start, slot_end)
            
            if time_filter_ok and time_slot_ok:
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

def greedy_route_optimizer(places, starting_coords):
    """Implement a simple greedy algorithm for route optimization as fallback"""
    remaining = places.copy()
    route = []
    current = starting_coords
    
    while remaining:
        # Find closest unvisited place
        best_dist = float('inf')
        best_place = None
        best_idx = -1
        
        for i, place in enumerate(remaining):
            place_coords = (place['Latitude'], place['Longitude'])
            dist = geodesic(current, place_coords).km
            
            if dist < best_dist:
                best_dist = dist
                best_place = place
                best_idx = i
        
        if best_place is not None:
            route.append(best_place)
            current = (best_place['Latitude'], best_place['Longitude'])
            remaining.pop(best_idx)
        else:
            break
    
    return route

def train_route_optimizer_model(tourist_places):
    """Train ML model to optimize route between tourist locations"""
    # Create training data: pairs of tourist places with their distances
    X = []
    y = []
    
    tourist_places_list = tourist_places.to_dict('records')
    
    # Generate all possible pairs of tourist places
    for i, place1 in enumerate(tourist_places_list):
        for j, place2 in enumerate(tourist_places_list):
            if i != j:
                # Enhanced features: add distance-based features
                dist = geodesic((place1['Latitude'], place1['Longitude']), 
                               (place2['Latitude'], place2['Longitude'])).km
                
                features = [
                    place1['Latitude'], place1['Longitude'], place1['Star Rating'],
                    place2['Latitude'], place2['Longitude'], place2['Star Rating'],
                    dist,  # Add calculated distance between points
                    # Add time-based factors if available
                    time_to_minutes(place1.get('Opening Time', '09:00')),
                    time_to_minutes(place2.get('Opening Time', '09:00'))
                ]
                
                # Better target heuristic: combine distance and rating
                rating_diff = place2['Star Rating'] - place1['Star Rating']
                
                # Prefer shorter distances and higher ratings
                # If distance is small OR rating is much better, choose second place first
                target = 1 if (dist < 15 and rating_diff > 0.5) or (rating_diff > 1.5) else 0
                
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
    
    # Build enhanced model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Add dropout for regularization
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training ML model for route optimization...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=25,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler_X

def train_ml_model(restaurants, tourist_places):
    """Train ML model to predict optimal breakfast locations"""
    # Create training data: (tourist_lat, tourist_lon) -> (restaurant_lat, restaurant_lon)
    X = []
    y = []
    
    # Enhanced training data with more features
    for _, place in tourist_places.iterrows():
        for _, rest in restaurants[restaurants['Breakfast'] == 'Yes'].iterrows():
            # Check if restaurant is open during breakfast time
            if is_open_during_slot(rest, time_slots['breakfast'][0], time_slots['breakfast'][1]):
                # Calculate distance between tourist place and restaurant
                dist = geodesic((place['Latitude'], place['Longitude']), 
                               (rest['Latitude'], rest['Longitude'])).km
                
                # Add more features including distance and ratings
                X.append([
                    place['Latitude'], place['Longitude'], place['Star Rating'],
                    dist,  # Distance between restaurant and tourist place
                    rest['Rating']  # Restaurant rating
                ])
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
    
    # Build model with regularization
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(3)
    ])
    
    model.compile(optimizer=Adam(0.0005), loss='mse')
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training ML model for breakfast location optimization...")
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, scaler_y.transform(y_test)),
        epochs=25,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler_X, scaler_y

def train_location_ordering_model(tourist_places):
    """Train ML model to order 3 locations optimally"""
    # Create training data from all possible triplets of locations
    X = []
    y = []
    
    tourist_places_list = tourist_places.to_dict('records')
    
    # Generate all possible triplets (limited to a reasonable subset for training)
    max_places = min(50, len(tourist_places_list))  # Limit to 50 places for practicality
    triplet_indices = list(itertools.combinations(range(max_places), 3))
    
    for idx_a, idx_b, idx_c in triplet_indices:
        place_a = tourist_places_list[idx_a]
        place_b = tourist_places_list[idx_b]
        place_c = tourist_places_list[idx_c]
        
        # Calculate distances between each place
        dist_ab = geodesic((place_a['Latitude'], place_a['Longitude']), 
                         (place_b['Latitude'], place_b['Longitude'])).km
        dist_bc = geodesic((place_b['Latitude'], place_b['Longitude']), 
                         (place_c['Latitude'], place_c['Longitude'])).km
        dist_ac = geodesic((place_a['Latitude'], place_a['Longitude']), 
                         (place_c['Latitude'], place_c['Longitude'])).km
        
        all_permutations = list(itertools.permutations([place_a, place_b, place_c]))
        
        # For each permutation, calculate total travel distance with more realistic factors
        min_distance = float('inf')
        best_permutation_idx = 0
        
        for i, (p1, p2, p3) in enumerate(all_permutations):
            # Calculate total distance for this permutation
            d1 = geodesic((0, 0), (p1['Latitude'], p1['Longitude'])).km  # From origin
            d2 = geodesic((p1['Latitude'], p1['Longitude']), (p2['Latitude'], p2['Longitude'])).km
            d3 = geodesic((p2['Latitude'], p2['Longitude']), (p3['Latitude'], p3['Longitude'])).km
            
            # Add weighting factors: favor higher ratings for earlier visits
            # and penalize longer distances more significantly
            rating_factor = 0.1 * (p1['Star Rating'] + 0.8 * p2['Star Rating'] + 0.6 * p3['Star Rating'])
            distance_penalty = d1 + 1.2 * d2 + d3  # Middle journey weighted higher
            
            total_score = distance_penalty - rating_factor
            
            if total_score < min_distance:
                min_distance = total_score
                best_permutation_idx = i
        
        # Enhanced features: include distances between places and ratings
        features = [
            place_a['Latitude'], place_a['Longitude'], place_a['Star Rating'],
            place_b['Latitude'], place_b['Longitude'], place_b['Star Rating'],
            place_c['Latitude'], place_c['Longitude'], place_c['Star Rating'],
            dist_ab, dist_bc, dist_ac,  # Add pairwise distances
            # Add opening hours if available
            time_to_minutes(place_a.get('Opening Time', '09:00')),
            time_to_minutes(place_b.get('Opening Time', '09:00')),
            time_to_minutes(place_c.get('Opening Time', '09:00'))
        ]
        
        # Target: index of best permutation (0-5)
        X.append(features)
        y.append(best_permutation_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode the permutation index
    from tensorflow.keras.utils import to_categorical
    y_one_hot = to_categorical(y, num_classes=6)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    # Normalize
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Build enhanced model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    print("Training ML model for optimal location ordering...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler_X

def predict_breakfast_location(model, scaler_X, scaler_y, tourist_place):
    """Use trained model to predict optimal breakfast location"""
    # Find highest rated breakfast places first
    breakfast_options = restaurants[restaurants['Breakfast'] == 'Yes'].sort_values('Rating', ascending=False)
    
    # Filter to only keep places open during breakfast time
    open_breakfast = []
    for _, rest in breakfast_options.iterrows():
        if is_open_during_slot(rest, time_slots['breakfast'][0], time_slots['breakfast'][1]):
            # Calculate distance to tourist place
            dist = geodesic((tourist_place['Latitude'], tourist_place['Longitude']), 
                          (rest['Latitude'], rest['Longitude'])).km
            open_breakfast.append((rest, dist))
    
    # Sort by a combination of distance and rating
    # Prioritize places with good ratings that aren't too far
    scored_options = [(rest, dist, rest['Rating'] - 0.15 * dist) for rest, dist in open_breakfast]
    sorted_options = sorted(scored_options, key=lambda x: x[2], reverse=True)
    
    if sorted_options:
        best_rest = sorted_options[0][0]
        return best_rest['Name'], best_rest, sorted_options[0][1]
    
    # Fallback to model prediction if no suitable places found
    # Prepare input with enhanced features
    dist = 0  # placeholder, will be calculated in the loop
    X = np.array([[
        tourist_place['Latitude'], 
        tourist_place['Longitude'], 
        tourist_place['Star Rating'],
        dist,  # Distance placeholder
        4.0  # Default rating placeholder
    ]])
    X_scaled = scaler_X.transform(X)
    
    # Predict
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Find closest actual breakfast place to prediction
    min_dist = float('inf')
    best_rest = None
    
    for _, rest in breakfast_options.iterrows():
        dist = geodesic((y_pred[0][0], y_pred[0][1]), (rest['Latitude'], rest['Longitude'])).km
        if dist < min_dist:
            min_dist = dist
            best_rest = rest
    
    if best_rest is None:
        # Absolute fallback - just pick the highest rated breakfast place
        best_rest = breakfast_options.iloc[0]
        dist = geodesic((tourist_place['Latitude'], tourist_place['Longitude']), 
                      (best_rest['Latitude'], best_rest['Longitude'])).km
        return best_rest['Name'], best_rest, dist
    
    return best_rest['Name'], best_rest, min_dist

def optimize_three_locations_order(loc_model, loc_scaler, loc1_data, loc2_data, loc3_data, starting_coords):
    """Use ML model to decide the optimal order of three locations"""
    # Calculate direct distances first for a more explicit comparison
    direct_distances = {
        'loc1': geodesic(starting_coords, (loc1_data['Latitude'], loc1_data['Longitude'])).km,
        'loc2': geodesic(starting_coords, (loc2_data['Latitude'], loc2_data['Longitude'])).km,
        'loc3': geodesic(starting_coords, (loc3_data['Latitude'], loc3_data['Longitude'])).km
    }
    
    # Calculate pairwise distances
    dist_12 = geodesic((loc1_data['Latitude'], loc1_data['Longitude']), 
                     (loc2_data['Latitude'], loc2_data['Longitude'])).km
    dist_23 = geodesic((loc2_data['Latitude'], loc2_data['Longitude']), 
                     (loc3_data['Latitude'], loc3_data['Longitude'])).km
    dist_13 = geodesic((loc1_data['Latitude'], loc1_data['Longitude']), 
                     (loc3_data['Latitude'], loc3_data['Longitude'])).km
    
    # Enhanced features for prediction
    features = [
        loc1_data['Latitude'], loc1_data['Longitude'], loc1_data['Star Rating'],
        loc2_data['Latitude'], loc2_data['Longitude'], loc2_data['Star Rating'],
        loc3_data['Latitude'], loc3_data['Longitude'], loc3_data['Star Rating'],
        dist_12, dist_23, dist_13,  # Add pairwise distances
        direct_distances['loc1'], direct_distances['loc2'], direct_distances['loc3'],  # Add starting distances
        # Include opening time info
        time_to_minutes(loc1_data.get('Opening Time', '09:00')),
        time_to_minutes(loc2_data.get('Opening Time', '09:00')),
        time_to_minutes(loc3_data.get('Opening Time', '09:00'))
    ]
    
    # Ensure we have exactly 15 features to match the model
    features = features[:15]
    
    X = np.array([features])
    
    # Check if features match the expected model input dimensions
    if X.shape[1] != loc_scaler.n_features_in_:
        print(f"Warning: Feature mismatch. Model expects {loc_scaler.n_features_in_} features, got {X.shape[1]}")
        # Fallback to greedy approach if dimensions don't match
        return greedy_route_optimizer([loc1_data, loc2_data, loc3_data], starting_coords)
    
    X_scaled = loc_scaler.transform(X)
    
    # Predict optimal permutation
    try:
        perm_probs = loc_model.predict(X_scaled)[0]
        best_perm_idx = np.argmax(perm_probs)
        
        # Define all permutations in the same order as during training
        permutations = list(itertools.permutations([loc1_data, loc2_data, loc3_data]))
        best_permutation = permutations[best_perm_idx]
    except:
        print("ML prediction failed, using greedy fallback...")
        return greedy_route_optimizer([loc1_data, loc2_data, loc3_data], starting_coords)
    
    # Check time constraints for each permutation
    valid_permutations = []
    
    for perm in permutations:
        valid = True
        for i, place in enumerate(perm):
            slot_name = f'location{i+1}'
            if not is_open_during_slot(place, time_slots[slot_name][0], time_slots[slot_name][1]):
                valid = False
                break
        
        if valid:
            valid_permutations.append(perm)
    
    if valid_permutations:
        # Calculate total distance and weighted score for each valid permutation
        min_score = float('inf')
        best_valid_perm = valid_permutations[0]
        
        for perm in valid_permutations:
            current_coords = starting_coords
            total_dist = 0
            
            for place in perm:
                place_coords = (place['Latitude'], place['Longitude'])
                dist = geodesic(current_coords, place_coords).km
                total_dist += dist
                current_coords = place_coords
            
            # Create a score that considers both distance and ratings
            # Weighted to favor higher rated places earlier in the day
            rating_factor = (1.0 * perm[0]['Star Rating'] + 
                             0.8 * perm[1]['Star Rating'] + 
                             0.6 * perm[2]['Star Rating'])
            
            score = total_dist - (rating_factor * 2)  # Balance distance vs. ratings
            
            if score < min_score:
                min_score = score
                best_valid_perm = perm
        
        return best_valid_perm
    
    # If no valid permutations based on time slots, use pure distance optimization
    return greedy_route_optimizer([loc1_data, loc2_data, loc3_data], starting_coords)

def generate_daily_itinerary(must_visit_locations, other_locations, bf_model, bf_scaler_X, bf_scaler_y, 
                           route_model, route_scaler, loc_order_model, loc_order_scaler):
    """Generate optimized daily itinerary using ML for route optimization"""
    itinerary = []
    distances = []
    nearby_options = []
    visited = set()
    
    # Select must-visit locations for this day (minimum 1, maximum 2)
    day_must_visit = must_visit_locations[:min(2, len(must_visit_locations))]
    remaining_must_visit = must_visit_locations[min(2, len(must_visit_locations)):]
    
    # Get location data for first must-visit location (or fallback if none)
    if len(day_must_visit) > 0:
        loc1_data = tourist_places[tourist_places['Name'] == day_must_visit[0]].iloc[0]
    else:
        # If no must-visit locations left, pick a high-rated location
        loc1_data = tourist_places.sort_values('Star Rating', ascending=False).iloc[0]
        day_must_visit.append(loc1_data['Name'])
    
    # Get location data for second must-visit location (or find a nearby one)
    if len(day_must_visit) > 1:
        loc2_data = tourist_places[tourist_places['Name'] == day_must_visit[1]].iloc[0]
    else:
        # Find a nearby high-rated location
        nearby = get_nearest_location(
            (loc1_data['Latitude'], loc1_data['Longitude']),
            tourist_places[~tourist_places['Name'].isin(must_visit_locations + day_must_visit)],
            visited
        )
        if nearby[0]:
            loc2_data = nearby[1]
            day_must_visit.append(nearby[0])
        else:
            # Fallback if no nearby locations found
            loc2_data = tourist_places.sort_values('Star Rating', ascending=False).iloc[1]
            day_must_visit.append(loc2_data['Name'])
    
    # 1. Find breakfast restaurant using ML prediction
    breakfast_name, breakfast_data, breakfast_dist = predict_breakfast_location(
        bf_model, bf_scaler_X, bf_scaler_y, loc1_data)
    
    itinerary.append({
        'Type': 'Breakfast Restaurant',
        'Name': breakfast_name,
        'Cuisine': breakfast_data['Cuisine Type'],
        'Rating': breakfast_data['Rating'],
        'Latitude': breakfast_data['Latitude'],
        'Longitude': breakfast_data['Longitude'],
        'Time': time_slots['breakfast']
    })
    distances.append(None)  # No distance for starting location
    visited.add(breakfast_name)
    current_coords = (breakfast_data['Latitude'], breakfast_data['Longitude'])
    
    # Find nearby breakfast options
    breakfast_coords = (breakfast_data['Latitude'], breakfast_data['Longitude'])
    nearby_breakfast = get_nearby_restaurants(
        breakfast_coords, 
        restaurants[restaurants['Breakfast'] == 'Yes'], 
        visited,
        time_slot='breakfast'
    )
    nearby_options.append(nearby_breakfast)
    
    # 2. Find an additional tourist place (either from remaining must-visit or other locations)
    if remaining_must_visit:
        candidate_places = tourist_places[
            ~tourist_places['Name'].isin(day_must_visit) & 
            tourist_places['Name'].isin(remaining_must_visit)
        ]
    else:
        candidate_places = tourist_places[
            ~tourist_places['Name'].isin(day_must_visit)
        ]
    
    best_additional = None
    best_additional_data = None
    min_total_distance = float('inf')
    
    for _, candidate in candidate_places.iterrows():
        if candidate['Name'] not in visited and is_open_during_slot(candidate, 
                                                                   time_slots['location3'][0], 
                                                                   time_slots['location3'][1]):
            # Try this candidate and measure the total route distance
            temp_order = optimize_three_locations_order(
                loc_order_model, loc_order_scaler,
                loc1_data, loc2_data, candidate,
                current_coords
            )
            
            # Calculate total distance for this route
            total_dist = 0
            temp_coords = current_coords
            
            for place in temp_order:
                place_coords = (place['Latitude'], place['Longitude'])
                dist = geodesic(temp_coords, place_coords).km
                total_dist += dist
                temp_coords = place_coords
            
            # Also consider the place's rating
            adjusted_score = total_dist - (candidate['Star Rating'] * 3)
            
            if adjusted_score < min_total_distance:
                min_total_distance = adjusted_score
                best_additional = candidate['Name']
                best_additional_data = candidate
    
    if not best_additional:
        print("No suitable additional tourist place found!")
        additional_name, additional_data, additional_dist = get_nearest_location(
            current_coords,
            tourist_places[~tourist_places['Name'].isin(day_must_visit)],
            visited,
            time_slot='location3'
        )
        if not additional_name:
            return None, None, None, remaining_must_visit
    else:
        additional_name = best_additional
        additional_data = best_additional_data
    
    # 3. Optimize the order of the three tourist places
    all_tourist_places = [loc1_data, loc2_data, additional_data]
    optimal_order = optimize_three_locations_order(
        loc_order_model, loc_order_scaler,
        loc1_data, loc2_data, additional_data,
        current_coords
    )
    
    # 4. Visit first tourist location
    first_location = optimal_order[0]
    first_location_coords = (first_location['Latitude'], first_location['Longitude'])
    dist_to_first = geodesic(current_coords, first_location_coords).km
    
    itinerary.append({
        'Type': 'Tourist Attraction (Location 1)',
        'Name': first_location['Name'],
        'Category': first_location['Type'],
        'Rating': first_location['Star Rating'],
        'Latitude': first_location['Latitude'],
        'Longitude': first_location['Longitude'],
        'Time': time_slots['location1']
    })
    distances.append(dist_to_first)
    visited.add(first_location['Name'])
    current_coords = first_location_coords
    nearby_options.append([])
    
    # 5. Find lunch restaurant
    lunch_candidates = restaurants[restaurants['Lunch'] == 'Yes']
    
    # Find nearby lunch options that are open during lunch time
    nearby_lunch = get_nearby_restaurants(
        current_coords,
        lunch_candidates,
        visited,
        time_slot='lunch',
        max_distance=3.0
    )
    
    if not nearby_lunch:
        # Fallback - find any lunch place within larger radius
        nearby_lunch = get_nearby_restaurants(
            current_coords,
            lunch_candidates,
            visited,
            time_slot='lunch',
            max_distance=5.0
        )
    
    if nearby_lunch:
        # Select the best lunch option based on rating and distance
        scored_lunch = []
        for name, cuisine, dist in nearby_lunch:
            rest_data = restaurants[restaurants['Name'] == name].iloc[0]
            score = rest_data['Rating'] - 0.1 * dist
            scored_lunch.append((name, cuisine, dist, score))
        
        scored_lunch.sort(key=lambda x: x[3], reverse=True)
        best_lunch_name, best_lunch_cuisine, lunch_dist, _ = scored_lunch[0]
        
        # Get full lunch data
        lunch_data = restaurants[restaurants['Name'] == best_lunch_name].iloc[0]
        
        itinerary.append({
            'Type': 'Lunch Restaurant',
            'Name': best_lunch_name,
            'Cuisine': best_lunch_cuisine,
            'Rating': lunch_data['Rating'],
            'Latitude': lunch_data['Latitude'],
            'Longitude': lunch_data['Longitude'],
            'Time': time_slots['lunch']
        })
        distances.append(lunch_dist)
        visited.add(best_lunch_name)
        current_coords = (lunch_data['Latitude'], lunch_data['Longitude'])
        nearby_options.append(nearby_lunch)
    else:
        itinerary.append({
            'Type': 'Note',
            'Note': 'No suitable lunch location found nearby',
            'Time': time_slots['lunch']
        })
        distances.append(None)
        nearby_options.append([])
    
    # 6. Visit second tourist location
    second_location = optimal_order[1]
    dist_to_second = geodesic(current_coords, (second_location['Latitude'], second_location['Longitude'])).km
    
    itinerary.append({
        'Type': 'Tourist Attraction (Location 2)',
        'Name': second_location['Name'],
        'Category': second_location['Type'],
        'Rating': second_location['Star Rating'],
        'Latitude': second_location['Latitude'],
        'Longitude': second_location['Longitude'],
        'Time': time_slots['location2']
    })
    distances.append(dist_to_second)
    visited.add(second_location['Name'])
    current_coords = (second_location['Latitude'], second_location['Longitude'])
    nearby_options.append([])
    
    # 7. Visit third tourist location (additional place)
    third_location = optimal_order[2]
    dist_to_third = geodesic(current_coords, (third_location['Latitude'], third_location['Longitude'])).km
    
    itinerary.append({
        'Type': 'Tourist Attraction (Location 3)',
        'Name': third_location['Name'],
        'Category': third_location['Type'],
        'Rating': third_location['Star Rating'],
        'Latitude': third_location['Latitude'],
        'Longitude': third_location['Longitude'],
        'Time': time_slots['location3']
    })
    distances.append(dist_to_third)
    visited.add(third_location['Name'])
    current_coords = (third_location['Latitude'], third_location['Longitude'])
    nearby_options.append([])
    
    # 8. Find dinner restaurant
    dinner_candidates = restaurants[restaurants['Dinner'] == 'Yes']
    
    # Find nearby dinner options that are open during dinner time
    nearby_dinner = get_nearby_restaurants(
        current_coords,
        dinner_candidates,
        visited,
        time_slot='dinner',
        max_distance=3.0
    )
    
    if not nearby_dinner:
        # Fallback - find any dinner place within larger radius
        nearby_dinner = get_nearby_restaurants(
            current_coords,
            dinner_candidates,
            visited,
            time_slot='dinner',
            max_distance=5.0
        )
    
    if nearby_dinner:
        # Select the best dinner option based on rating and distance
        scored_dinner = []
        for name, cuisine, dist in nearby_dinner:
            rest_data = restaurants[restaurants['Name'] == name].iloc[0]
            score = rest_data['Rating'] - 0.1 * dist
            scored_dinner.append((name, cuisine, dist, score))
        
        scored_dinner.sort(key=lambda x: x[3], reverse=True)
        best_dinner_name, best_dinner_cuisine, dinner_dist, _ = scored_dinner[0]
        
        # Get full dinner data
        dinner_data = restaurants[restaurants['Name'] == best_dinner_name].iloc[0]
        
        itinerary.append({
            'Type': 'Dinner Restaurant',
            'Name': best_dinner_name,
            'Cuisine': best_dinner_cuisine,
            'Rating': dinner_data['Rating'],
            'Latitude': dinner_data['Latitude'],
            'Longitude': dinner_data['Longitude'],
            'Time': time_slots['dinner']
        })
        distances.append(dinner_dist)
        visited.add(best_dinner_name)
        nearby_options.append(nearby_dinner)
    else:
        itinerary.append({
            'Type': 'Note',
            'Note': 'No suitable dinner location found nearby',
            'Time': time_slots['dinner']
        })
        distances.append(None)
        nearby_options.append([])
    
    # Update remaining must-visit locations
    if additional_name in remaining_must_visit:
        remaining_must_visit.remove(additional_name)
    if day_must_visit[0] in remaining_must_visit:
        remaining_must_visit.remove(day_must_visit[0])
    if day_must_visit[1] in remaining_must_visit:
        remaining_must_visit.remove(day_must_visit[1])
    
    return itinerary, distances, nearby_options, remaining_must_visit

def generate_multi_day_itinerary(num_days, must_visit_locations, bf_model, bf_scaler_X, bf_scaler_y,
                               route_model, route_scaler, loc_order_model, loc_order_scaler):
    """Generate multi-day itinerary with optimized daily plans"""
    all_itineraries = []
    remaining_must_visit = must_visit_locations.copy()
    other_locations = tourist_places[~tourist_places['Name'].isin(must_visit_locations)]
    
    for day in range(1, num_days + 1):
        print(f"\nGenerating itinerary for Day {day}...")
        
        # Generate daily itinerary
        itinerary, distances, nearby_options, remaining_must_visit = generate_daily_itinerary(
            remaining_must_visit if remaining_must_visit else must_visit_locations,
            other_locations,
            bf_model, bf_scaler_X, bf_scaler_y,
            route_model, route_scaler,
            loc_order_model, loc_order_scaler
        )
        
        if not itinerary:
            print(f"Could not generate itinerary for Day {day}")
            continue
        
        # Store the daily itinerary
        all_itineraries.append({
            'day': day,
            'itinerary': itinerary,
            'distances': distances,
            'nearby_options': nearby_options
        })
        
        # If we've visited all must-see locations, fill remaining days with other attractions
        if not remaining_must_visit:
            must_visit_locations = []  # Switch to using only other locations
    
    return all_itineraries

def print_multi_day_itinerary(all_itineraries):
    """Print the complete multi-day itinerary in the requested format"""
    if not all_itineraries:
        print("No itineraries were generated.")
        return
    
    total_trip_distance = 0
    
    for day_plan in all_itineraries:
        day = day_plan['day']
        itinerary = day_plan['itinerary']
        distances = day_plan['distances']
        nearby_options = day_plan['nearby_options']
        
        day_distance = sum(d for d in distances if d is not None)
        total_trip_distance += day_distance
        
        print("\n" + "=" * 60)
        print(f" " * 20 + f"DAY {day} ITINERARY")
        print("=" * 60)
        print("\n")
        
        for i, item in enumerate(itinerary):
            # Print section number
            print(f"{i+1}. {item['Type'].upper()}")
            
            # Print details with consistent formatting
            if 'Name' in item:
                print(f"   Name: {item['Name']}")
            if 'Cuisine' in item:
                print(f"   Cuisine: {item['Cuisine']}")
            if 'Category' in item:
                print(f"   Category: {item['Category']}")
            if 'Rating' in item:
                print(f"   Rating: {item['Rating']:.1f}/5")
            if 'Latitude' in item and 'Longitude' in item:
                print(f"   Coordinates: ({item['Latitude']:.6f}, {item['Longitude']:.6f})")
            if 'Note' in item:  # Add this condition to print the note content
                print(f"   Note: {item['Note']}")
            if distances[i] is not None:
                print(f"   Distance from previous: {distances[i]:.2f} km")
            
            # Show nearby alternatives if available and it's a restaurant
            if nearby_options[i] and ('Restaurant' in item['Type'] or 'Breakfast' in item['Type']):
                print("\n   Nearby alternatives:")
                for alt in nearby_options[i]:
                    print(f"   - {alt[0]} ({alt[1]}, {alt[2]:.1f} km away)")
            
            print()  # Add empty line between items
        
        print("=" * 60)
        print(f"DAY {day} TRAVEL DISTANCE: {day_distance:.2f} km")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print(f" " * 15 + "COMPLETE TRIP SUMMARY")
    print("=" * 60)
    print(f"\nTotal trip duration: {len(all_itineraries)} days")
    print(f"Total travel distance: {total_trip_distance:.2f} km")
    print("=" * 60)

# Train all models
print("Training models...")
bf_model, bf_scaler_X, bf_scaler_y = train_ml_model(restaurants, tourist_places)
route_model, route_scaler = train_route_optimizer_model(tourist_places)
loc_order_model, loc_order_scaler = train_location_ordering_model(tourist_places)

# Example usage
if __name__ == "__main__":
    # Get user input for trip duration
    while True:
        try:
            num_days = int(input("Enter number of days for your Goa trip (3-7 days): "))
            if 3 <= num_days <= 7:
                break
            else:
                print("Please enter a number between 3 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get user input for must-visit locations
    available_locations = tourist_places['Name'].tolist()
    print("\nAvailable tourist locations in Goa:")
    for i, loc in enumerate(available_locations[:20]):  # Show first 20 for brevity
        print(f"{i+1}. {loc}")
    
    must_visit_locations = []
    print("\nPlease enter 6 must-visit locations (one at a time):")
    while len(must_visit_locations) < 6:
        loc = input(f"Enter location {len(must_visit_locations)+1}: ")
        matched_loc = find_closest_match(loc, available_locations)
        if matched_loc:
            if matched_loc not in must_visit_locations:
                must_visit_locations.append(matched_loc)
                print(f"Added: {matched_loc}")
            else:
                print("You've already added this location.")
        else:
            print("Location not found. Please try again.")
    
    print("\nYour must-visit locations:")
    for i, loc in enumerate(must_visit_locations):
        print(f"{i+1}. {loc}")
    
    # Generate multi-day itinerary
    print("\nGenerating your optimized Goa itinerary...")
    all_itineraries = generate_multi_day_itinerary(
        num_days,
        must_visit_locations,
        bf_model, bf_scaler_X, bf_scaler_y,
        route_model, route_scaler,
        loc_order_model, loc_order_scaler
    )
    
    # Print the complete itinerary
    print_multi_day_itinerary(all_itineraries)