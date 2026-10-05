import streamlit as st
import pandas as pd
import plotly.express as px
from utils import set_page_config, render_header, render_footer
from data_loader import fetch_live_data, get_demo_data, generate_behavior_data
from ml_model import train_kmeans, calculate_elbow, predict_new_passenger

set_page_config()

# Sidebar Navigation
st.sidebar.title("TransitIQ Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard", 
    "📡 Live Transport Data", 
    "👥 Passenger Behavior", 
    "🤖 ML Segmentation", 
    "📊 Visual Analytics", 
    "🔎 Cluster Insights", 
    "🧪 New Passenger Analysis", 
    "ℹ️ Data Source"
])

# Initialize session state for data
if 'transport_data' not in st.session_state:
    st.session_state.transport_data = None
    st.session_state.data_status = None
    st.session_state.last_updated = None
    st.session_state.behavior_data = None
    st.session_state.model = None
    st.session_state.clustered_data = None
    st.session_state.silhouette = None
    st.session_state.cluster_names = None

def load_data(force_refresh=False):
    if force_refresh or st.session_state.transport_data is None:
        with st.spinner("Fetching transport data..."):
            df, status, ts = fetch_live_data()
            if df is None:
                if st.session_state.transport_data is not None:
                    # Use cached
                    st.session_state.data_status = "🟡 CACHED DATA"
                else:
                    # Use demo
                    df, status, ts = get_demo_data()
                    st.session_state.transport_data = df
                    st.session_state.data_status = status
                    st.session_state.last_updated = ts
            else:
                st.session_state.transport_data = df
                st.session_state.data_status = status
                st.session_state.last_updated = ts
                
            # Regenerate behavior data on new transport data
            st.session_state.behavior_data = generate_behavior_data(st.session_state.transport_data)
            # Reset ML model
            st.session_state.model = None

load_data()

render_header()
st.sidebar.markdown("---")
st.sidebar.write(f"**Status:** {st.session_state.data_status}")
st.sidebar.write(f"**Updated:** {st.session_state.last_updated}")
if st.sidebar.button("🔄 Refresh Live Data"):
    load_data(force_refresh=True)

if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    st.write(f"Current Data Status: **{st.session_state.data_status}** (Last Updated: {st.session_state.last_updated})")
    
    if st.session_state.data_status == "🟡 CACHED DATA":
        st.warning(f"⚠️ Live API is currently unavailable. Showing cached data from {st.session_state.last_updated}.")
    elif st.session_state.data_status == "🔴 API OFFLINE":
        st.error("Unable to retrieve live transport data. Please try again later.")
        
    df = st.session_state.transport_data
    b_df = st.session_state.behavior_data
    
    if df is not None and not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Active Vehicles", len(df))
        col2.metric("Active Routes", df['route_id'].nunique() if 'route_id' in df.columns else 0)
        col3.metric("Transport Modes", df['transport_mode'].nunique() if 'transport_mode' in df.columns else 0)
        col4.metric("Avg Trip Distance", f"{b_df['avg_travel_distance_km'].mean():.1f} km" if b_df is not None else "N/A")
        
        st.subheader("Transport Mode Distribution")
        if 'transport_mode' in df.columns:
            fig = px.pie(df, names='transport_mode', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Route Activity")
        if 'name' in df.columns:
            route_counts = df['name'].value_counts().reset_index()
            route_counts.columns = ['Route Name', 'Count']
            fig2 = px.bar(route_counts.head(10), x='Route Name', y='Count')
            st.plotly_chart(fig2, use_container_width=True)
            
elif page == "📡 Live Transport Data":
    st.header("📡 Live Transport Data")
    st.write("### Connection Status")
    st.info(f"{st.session_state.data_status} - Last Updated: {st.session_state.last_updated}")
    
    df = st.session_state.transport_data
    if df is not None and not df.empty:
        st.write("### Data Explorer")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download CSV", csv, "live_transport_data.csv", "text/csv")
        
        if 'lat' in df.columns and 'lon' in df.columns:
            st.write("### Live Map")
            map_df = df.dropna(subset=['lat', 'lon'])
            if not map_df.empty:
                fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", hover_name="route_id", 
                                        color="transport_mode" if 'transport_mode' in map_df.columns else None,
                                        zoom=10, mapbox_style="carto-positron")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Location information is not available from the selected data source.")
    else:
        st.write("No transport data available.")

elif page == "👥 Passenger Behavior":
    st.header("👥 Passenger Behavior")
    st.write("Anonymous travel behavior indicators derived from transport data.")
    
    b_df = st.session_state.behavior_data
    if b_df is not None:
        modes = ["All"] + list(b_df['preferred_mode'].unique())
        sel_mode = st.selectbox("Filter by Transport Mode", modes)
        
        filtered_df = b_df if sel_mode == "All" else b_df[b_df['preferred_mode'] == sel_mode]
        st.dataframe(filtered_df)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Behavior Data", csv, "behavior_data.csv", "text/csv")

elif page == "🤖 ML Segmentation":
    st.header("🤖 ML Segmentation")
    st.write("Group travel patterns using K-Means clustering.")
    
    b_df = st.session_state.behavior_data
    if b_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 6, 4)
        
        if st.button("🤖 Analyze Travel Behavior"):
            with st.spinner("Training K-Means..."):
                model, clustered, sil, names = train_kmeans(b_df, n_clusters)
                st.session_state.model = model
                st.session_state.clustered_data = clustered
                st.session_state.silhouette = sil
                st.session_state.cluster_names = names
                
        if st.session_state.clustered_data is not None:
            st.metric("Silhouette Score", f"{st.session_state.silhouette:.3f}")
            
            c_df = st.session_state.clustered_data
            st.subheader("Cluster Distribution")
            dist = c_df['behavior_group'].value_counts().reset_index()
            dist.columns = ['Behavior Group', 'Count']
            st.dataframe(dist)
            
            st.subheader("Travel Distance vs Frequency")
            fig = px.scatter(c_df, x='avg_travel_distance_km', y='trips_per_week', color='behavior_group',
                             hover_data=['avg_travel_duration_min'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Elbow Method")
            fig_elbow = calculate_elbow(b_df)
            st.plotly_chart(fig_elbow, use_container_width=True)

elif page == "📊 Visual Analytics":
    st.header("📊 Visual Analytics")
    b_df = st.session_state.behavior_data
    if b_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(b_df, x='trips_per_week', title="Trips Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.histogram(b_df, x='avg_travel_distance_km', title="Travel Distance Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.pie(b_df, names='preferred_mode', title="Trips by Transport Mode")
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = px.histogram(b_df, x='avg_transfers', title="Transfer Count Distribution")
            st.plotly_chart(fig4, use_container_width=True)

elif page == "🔎 Cluster Insights":
    st.header("🔎 Cluster Insights")
    c_df = st.session_state.clustered_data
    if c_df is not None:
        for c_idx, name in st.session_state.cluster_names.items():
            cluster_data = c_df[c_df['cluster'] == c_idx]
            with st.expander(f"### {name} (N={len(cluster_data)})", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Trips/Week", f"{cluster_data['trips_per_week'].mean():.1f}")
                col2.metric("Avg Distance", f"{cluster_data['avg_travel_distance_km'].mean():.1f} km")
                col3.metric("Peak-Hour %", f"{cluster_data['peak_hour_usage_pct'].mean()*100:.1f}%")
                
                st.write(f"**Explanation**: This group shows travel behavior characteristic of {name.lower()}, based on their average frequency of {cluster_data['trips_per_week'].mean():.1f} trips per week and average distance of {cluster_data['avg_travel_distance_km'].mean():.1f} km.")
    else:
        st.warning("Please run ML Segmentation first.")

elif page == "🧪 New Passenger Analysis":
    st.header("🧪 New Passenger Analysis")
    st.write("Analyze a new anonymous travel profile.")
    
    if st.session_state.model is not None:
        with st.form("new_passenger_form"):
            trips = st.number_input("Trips per week", 1, 50, 10)
            dist = st.number_input("Average travel distance (km)", 0.5, 100.0, 15.0)
            time = st.number_input("Average travel time (min)", 5, 240, 45)
            peak = st.slider("Peak-hour usage", 0.0, 1.0, 0.8)
            weekend = st.slider("Weekend travel", 0.0, 1.0, 0.2)
            transfers = st.number_input("Transfer count", 0, 10, 1)
            
            modes = st.session_state.behavior_data['preferred_mode'].unique()
            mode = st.selectbox("Transport mode", modes)
            
            submitted = st.form_submit_button("Predict Travel Behavior")
            if submitted:
                new_data = {
                    'trips_per_week': trips,
                    'avg_travel_distance_km': dist,
                    'avg_travel_duration_min': time,
                    'peak_hour_usage_pct': peak,
                    'weekend_usage_pct': weekend,
                    'avg_transfers': transfers,
                    'preferred_mode': mode
                }
                c_idx, name = predict_new_passenger(st.session_state.model, st.session_state.cluster_names, new_data)
                st.success(f"### Predicted Behavior: {name}")
                st.info("This classification describes travel behavior only. It does not identify or track an individual person.")
    else:
        st.warning("Please run ML Segmentation first to train the model.")

elif page == "ℹ️ Data Source":
    st.header("ℹ️ Data Source")
    st.write("### Source")
    st.write("MBTA V3 API (Massachusetts Bay Transportation Authority)")
    st.write("### Data Type")
    st.write("Live JSON API / Open Data")
    st.write("### Fields Used")
    st.write("- Route ID, Route Type, Long Name")
    st.write("- Vehicle ID, Status, Latitude, Longitude, Speed")
    st.write("### Current Status")
    st.write(f"{st.session_state.data_status} (Last Successful Refresh: {st.session_state.last_updated})")
    
    st.info("TransitIQ does not collect personally identifiable passenger information. It analyzes anonymous travel patterns derived from public transportation data.")
    st.markdown("[Visit Data Source](https://api-v3.mbta.com/)")

render_footer()
