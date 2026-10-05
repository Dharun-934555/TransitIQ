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
    # 2. Top Header
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown("<h1 style='margin-bottom:0; padding-bottom:0;'>TransitIQ</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:gray; font-size:18px; margin-top:0;'>Real-Time Public Transport Intelligence</p>", unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown("<div style='text-align:right; margin-top:20px; font-size:20px;'>🔔 👤 <b>Admin User</b></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. Welcome Section
    welcome_cols = st.columns([3, 1])
    with welcome_cols[0]:
        st.markdown("### Welcome back, Admin 👋")
        st.markdown("**Monitor transport activity, understand travel behavior, and discover intelligent mobility insights.**")
    with welcome_cols[1]:
        status_color = "#28a745" if "LIVE" in st.session_state.data_status else "#fd7e14" if "CACHED" in st.session_state.data_status else "#007bff" if "DEMO" in st.session_state.data_status else "#dc3545"
        st.markdown(f"<div style='text-align:right; padding: 15px; border-radius: 10px; background-color: {status_color}; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'><b>{st.session_state.data_status}</b><br><small>Last updated: {st.session_state.last_updated}</small></div>", unsafe_allow_html=True)
        
    st.write("")
    
    df = st.session_state.transport_data
    b_df = st.session_state.behavior_data
    c_df = st.session_state.clustered_data
    
    # 4. KPI CARDS
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    trips_val = len(df) if df is not None else 0
    routes_val = df['route_id'].nunique() if df is not None and 'route_id' in df.columns else 0
    stops_val = len(df) if df is not None else 0 
    time_val = f"{b_df['avg_travel_duration_min'].mean():.1f}m" if b_df is not None else "0m"
    groups_val = len(st.session_state.cluster_names) if st.session_state.cluster_names else 0
    
    kpi1.metric("🚌 Total Trips", trips_val)
    kpi2.metric("🛣️ Active Routes", routes_val)
    kpi3.metric("📍 Active Stops", stops_val)
    kpi4.metric("⏱️ Avg Travel Time", time_val)
    kpi5.metric("👥 Behavior Groups", groups_val)
    
    st.write("")
    
    # 5. MAIN ANALYTICS SECTION
    col_main1, col_main2 = st.columns([2, 1])
    with col_main1:
        st.markdown("### 📈 Transport Activity")
        if df is not None and not df.empty:
            if 'status' in df.columns:
                status_counts = df['status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig_act = px.area(status_counts, x='Status', y='Count', color='Status', markers=True)
                fig_act.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_act, use_container_width=True)
            else:
                st.info("No activity data available.")
        else:
            st.info("No data available.")
            
    with col_main2:
        st.markdown("### 🍩 Transport Mode")
        if df is not None and not df.empty and 'transport_mode' in df.columns:
            fig_mode = px.pie(df, names='transport_mode', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_mode.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
            fig_mode.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_mode, use_container_width=True)
        else:
            st.info("No mode data available.")
            
    st.write("")
    
    # 6. SECOND ANALYTICS ROW
    col_sec1, col_sec2 = st.columns(2)
    with col_sec1:
        st.markdown("### 📊 Route Activity")
        if df is not None and not df.empty and 'name' in df.columns:
            route_counts = df['name'].value_counts().reset_index().head(6)
            route_counts.columns = ['Route Name', 'Activity']
            fig_route = px.bar(route_counts, y='Route Name', x='Activity', orientation='h', color='Activity', color_continuous_scale='Purples')
            fig_route.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_route, use_container_width=True)
        else:
            st.info("No route data available.")
            
    with col_sec2:
        st.markdown("### 📊 Peak Hour Analysis")
        if b_df is not None:
            peak_bins = pd.cut(b_df['peak_hour_usage_pct'], bins=[-0.1, 0.4, 0.7, 1.1], labels=['Off-Peak', 'Mixed', 'Peak-Hour'])
            peak_counts = peak_bins.value_counts().reset_index()
            peak_counts.columns = ['Period', 'Users']
            fig_peak = px.bar(peak_counts, x='Period', y='Users', color='Period', color_discrete_map={'Off-Peak': '#AEC7E8', 'Mixed': '#FFBB78', 'Peak-Hour': '#FF7F0E'})
            fig_peak.update_layout(margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_peak, use_container_width=True)
        else:
            st.info("No behavior data available.")
            
    st.write("---")
    
    # 7. PASSENGER BEHAVIOR OVERVIEW
    st.markdown("## 👥 Passenger Behavior Insights")
    if c_df is not None and st.session_state.cluster_names:
        cluster_cols = st.columns(len(st.session_state.cluster_names))
        total_pax = len(c_df)
        
        for i, (c_idx, name) in enumerate(st.session_state.cluster_names.items()):
            count = len(c_df[c_df['cluster'] == c_idx])
            pct = (count / total_pax) * 100
            
            border_colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3']
            b_color = border_colors[i % len(border_colors)]
            
            with cluster_cols[i]:
                st.markdown(f"""
                <div style='padding: 15px; border-radius: 10px; background-color: var(--background-color); border-left: 5px solid {b_color}; box-shadow: 0 2px 10px rgba(0,0,0,0.1); height: 100%;'>
                    <h4 style='margin-top:0;'>{name}</h4>
                    <h2 style='color:{b_color}; margin:0;'>{pct:.0f}%</h2>
                    <p style='color:gray; font-size:14px; margin-bottom:0;'>{count} travelers</p>
                </div>
                """, unsafe_allow_html=True)
                
        # 8. BEHAVIOR DISTRIBUTION
        st.write("")
        st.markdown("### Travel Behavior Distribution")
        dist = c_df['behavior_group'].value_counts().reset_index()
        dist.columns = ['Behavior Group', 'Count']
        fig_dist = px.bar(dist, x='Behavior Group', y='Count', color='Behavior Group', text='Count')
        fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("**Run ML Segmentation** from the sidebar to discover travel behavior groups.")
        
    st.write("---")
    
    # 9 & 10. SMART INSIGHTS & LIVE STATUS
    bot_col1, bot_col2 = st.columns([2, 1])
    with bot_col1:
        st.markdown("## 💡 Smart Mobility Insights")
        if b_df is not None and not b_df.empty:
            top_mode = b_df['preferred_mode'].mode()[0]
            avg_peak = b_df['peak_hour_usage_pct'].mean()
            peak_str = "higher" if avg_peak > 0.5 else "moderate"
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, rgba(99, 110, 250, 0.1) 0%, rgba(171, 99, 250, 0.1) 100%); border: 1px solid rgba(99, 110, 250, 0.2);'>
                <ul style='font-size: 16px; line-height: 1.8; margin-bottom:0;'>
                    <li><b>{top_mode}</b> routes account for the highest transport preference among tracked travelers.</li>
                    <li>Overall peak-hour usage is <b>{peak_str}</b> across the network ({avg_peak*100:.1f}% average).</li>
                    <li>The average traveler takes <b>{b_df['trips_per_week'].mean():.1f}</b> trips per week covering <b>{b_df['avg_travel_distance_km'].mean():.1f}</b> km.</li>
                    <li>Long-distance travel patterns are concentrated based on route availability.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough data to generate insights.")
            
    with bot_col2:
        st.markdown("### 🟢 Live Transport Status")
        status_bg = "#d4edda" if "LIVE" in st.session_state.data_status else "#fff3cd" if "CACHED" in st.session_state.data_status else "#cce5ff" if "DEMO" in st.session_state.data_status else "#f8d7da"
        status_color = "#155724" if "LIVE" in st.session_state.data_status else "#856404" if "CACHED" in st.session_state.data_status else "#004085" if "DEMO" in st.session_state.data_status else "#721c24"
        
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {status_bg}; color: {status_color}; border: 1px solid {status_color};'>
            <h4 style='margin-top:0;'>{st.session_state.data_status}</h4>
            <b>Data Source:</b> MBTA V3 API<br>
            <b>Records:</b> {trips_val}<br>
            <br>
            <small>Last updated: {st.session_state.last_updated}</small>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", key="refresh_dash"):
            load_data(force_refresh=True)
            st.rerun()
            
    st.write("---")
    
    # 11. QUICK ACTIONS
    st.markdown("## ⚡ Quick Actions")
    qa_cols = st.columns(5)
    qa_actions = [
        ("📡 Live Transport", "View Live Data"),
        ("🤖 Run ML Analysis", "Train Model"),
        ("📊 View Analytics", "Show Charts"),
        ("👥 Behavior Insights", "Cluster Info"),
        ("🧪 Analyze Profile", "New Passenger")
    ]
    
    for i, (label, desc) in enumerate(qa_actions):
        with qa_cols[i]:
            st.markdown(f"""
            <div style='text-align:center; padding:15px; background-color:rgba(128,128,128,0.05); border-radius:10px; border:1px solid rgba(128,128,128,0.2);'>
                <b>{label}</b><br>
                <small style='color:gray;'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
            

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
