import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

def get_preprocessor():
    numeric_features = ['trips_per_week', 'avg_travel_distance_km', 'avg_travel_duration_min', 
                        'peak_hour_usage_pct', 'weekend_usage_pct', 'avg_transfers']
    categorical_features = ['preferred_mode']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    return preprocessor

def train_kmeans(df, n_clusters=4):
    features = df.drop(columns=['anonymous_id', 'preferred_route'], errors='ignore')
    preprocessor = get_preprocessor()
    
    X = preprocessor.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    score = silhouette_score(X, clusters)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Generate dynamic names based on cluster stats
    cluster_names = {}
    for c in range(n_clusters):
        c_data = df_clustered[df_clustered['cluster'] == c]
        avg_trips = c_data['trips_per_week'].mean()
        avg_peak = c_data['peak_hour_usage_pct'].mean()
        avg_weekend = c_data['weekend_usage_pct'].mean()
        avg_dist = c_data['avg_travel_distance_km'].mean()
        
        name_parts = []
        if avg_trips > 15:
            name_parts.append("Heavy")
        elif avg_trips > 8:
            name_parts.append("Frequent")
        else:
            name_parts.append("Occasional")
            
        if avg_weekend > 0.6:
            name_parts.append("Weekend")
        elif avg_peak > 0.6:
            name_parts.append("Peak-Hour")
            
        if avg_dist > 18:
            name_parts.append("Long-Distance Travelers")
        else:
            name_parts.append("Commuters" if "Peak-Hour" in name_parts else "Travelers")
            
        cluster_names[c] = " ".join(name_parts)
        
    df_clustered['behavior_group'] = df_clustered['cluster'].map(cluster_names)
    
    # Create Pipeline for new predictions
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', kmeans)
    ])
    
    return model, df_clustered, score, cluster_names

def calculate_elbow(df, max_k=10):
    features = df.drop(columns=['anonymous_id', 'preferred_route'], errors='ignore')
    preprocessor = get_preprocessor()
    X = preprocessor.fit_transform(features)
    
    inertias = []
    K = range(2, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeanModel.fit(X)
        inertias.append(kmeanModel.inertia_)
        
    fig = go.Figure(data=go.Scatter(x=list(K), y=inertias, mode='lines+markers'))
    fig.update_layout(title='Elbow Method for Optimal K',
                      xaxis_title='Number of Clusters (K)',
                      yaxis_title='Inertia')
    return fig

def predict_new_passenger(model, cluster_names, passenger_data):
    df = pd.DataFrame([passenger_data])
    cluster = model.predict(df)[0]
    return cluster, cluster_names[cluster]
