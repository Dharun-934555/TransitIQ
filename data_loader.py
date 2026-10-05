import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=600)
def fetch_live_data():
    """Fetches live public transit data. We use MBTA V3 API as an example of an open API without mandatory keys."""
    status = "🔴 API OFFLINE"
    data = None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Fetch routes
        routes_response = requests.get("https://api-v3.mbta.com/routes", timeout=5)
        routes_response.raise_for_status()
        routes_data = routes_response.json().get("data", [])
        
        # Fetch vehicles
        vehicles_response = requests.get("https://api-v3.mbta.com/vehicles", timeout=5)
        vehicles_response.raise_for_status()
        vehicles_data = vehicles_response.json().get("data", [])
        
        if not routes_data and not vehicles_data:
            return None, status, timestamp
            
        routes_df = pd.DataFrame([{
            'route_id': r.get('id'),
            'type': r.get('attributes', {}).get('type'), # 0,1=subway, 2=rail, 3=bus, 4=ferry
            'name': r.get('attributes', {}).get('long_name')
        } for r in routes_data])
        
        type_mapping = {0: 'Subway', 1: 'Subway', 2: 'Commuter Rail', 3: 'Bus', 4: 'Ferry'}
        routes_df['transport_mode'] = routes_df['type'].map(type_mapping).fillna('Other')
        
        vehicles_df = pd.DataFrame([{
            'vehicle_id': v.get('id'),
            'route_id': v.get('relationships', {}).get('route', {}).get('data', {}).get('id'),
            'status': v.get('attributes', {}).get('current_status'),
            'lat': v.get('attributes', {}).get('latitude'),
            'lon': v.get('attributes', {}).get('longitude'),
            'speed': v.get('attributes', {}).get('speed')
        } for v in vehicles_data])
        
        # Merge
        if not vehicles_df.empty and not routes_df.empty:
            merged = pd.merge(vehicles_df, routes_df, on='route_id', how='left')
        else:
            merged = pd.DataFrame()
            
        status = "🟢 LIVE DATA"
        return merged, status, timestamp
    except Exception as e:
        return None, status, timestamp

def get_demo_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        df = pd.read_csv("sample_data/demo_transport_data.csv")
        return df, "🔵 DEMO DATA", timestamp
    except:
        # Generate some dummy if file missing
        data = {
            'vehicle_id': [f"V{i}" for i in range(100)],
            'route_id': np.random.choice(['Red', 'Blue', 'Green', 'Orange', 'Bus-1', 'Bus-2'], 100),
            'transport_mode': np.random.choice(['Subway', 'Bus'], 100),
            'status': ['IN_TRANSIT_TO'] * 100,
            'lat': np.random.uniform(42.3, 42.4, 100),
            'lon': np.random.uniform(-71.1, -71.0, 100),
            'speed': np.random.uniform(0, 40, 100)
        }
        return pd.DataFrame(data), "🔵 DEMO DATA", timestamp

def generate_behavior_data(transport_data, n_passengers=500):
    """Generates anonymous travel-behavior records based on available transportation data."""
    np.random.seed(42)
    
    # If we have real routes, use them. Else generic.
    if transport_data is not None and not transport_data.empty and 'transport_mode' in transport_data.columns:
        modes = transport_data['transport_mode'].dropna().unique()
        if len(modes) == 0:
            modes = ['Bus', 'Subway']
        routes = transport_data['route_id'].dropna().unique()
        if len(routes) == 0:
            routes = ['Route A', 'Route B']
    else:
        modes = ['Bus', 'Subway']
        routes = ['Route A', 'Route B']
        
    data = []
    for i in range(n_passengers):
        # Create some patterns
        profile_type = np.random.choice(['commuter', 'occasional', 'weekend', 'heavy'])
        
        if profile_type == 'commuter':
            trips = int(np.random.normal(10, 2))
            dist = np.random.normal(15, 5)
            time_spent = np.random.normal(45, 10)
            peak = np.random.uniform(0.7, 1.0)
            weekend = np.random.uniform(0, 0.2)
            transfers = int(np.random.choice([0, 1, 2], p=[0.4, 0.5, 0.1]))
        elif profile_type == 'occasional':
            trips = int(np.random.normal(3, 1))
            dist = np.random.normal(8, 4)
            time_spent = np.random.normal(20, 10)
            peak = np.random.uniform(0.1, 0.4)
            weekend = np.random.uniform(0.2, 0.5)
            transfers = int(np.random.choice([0, 1], p=[0.8, 0.2]))
        elif profile_type == 'weekend':
            trips = int(np.random.normal(4, 1))
            dist = np.random.normal(20, 8)
            time_spent = np.random.normal(60, 20)
            peak = np.random.uniform(0.0, 0.2)
            weekend = np.random.uniform(0.8, 1.0)
            transfers = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
        else:
            trips = int(np.random.normal(20, 5))
            dist = np.random.normal(10, 5)
            time_spent = np.random.normal(30, 15)
            peak = np.random.uniform(0.4, 0.8)
            weekend = np.random.uniform(0.4, 0.8)
            transfers = int(np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
            
        trips = max(1, trips)
        dist = max(1.0, dist)
        time_spent = max(5.0, time_spent)
        
        data.append({
            'anonymous_id': f"PAX-{np.random.randint(10000, 99999)}",
            'trips_per_week': trips,
            'avg_travel_distance_km': round(dist, 1),
            'avg_travel_duration_min': round(time_spent, 1),
            'peak_hour_usage_pct': round(peak, 2),
            'weekend_usage_pct': round(weekend, 2),
            'avg_transfers': transfers,
            'preferred_mode': np.random.choice(modes),
            'preferred_route': np.random.choice(routes)
        })
        
    return pd.DataFrame(data)
