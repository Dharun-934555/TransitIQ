# TransitIQ

**Real-Time Public Transport Intelligence & Passenger Behavior Analytics**

## Overview
TransitIQ is a real-world public transportation data analysis and machine learning application. It retrieves live public transport data, derives anonymous travel-behavior indicators, and uses K-Means clustering to automatically identify meaningful travel patterns and groups.

## Problem Statement
Understanding public transit usage patterns can be difficult due to privacy concerns and the massive amount of disconnected vehicle data. 

## Objectives
- Fetch live transportation data without tracking real identities.
- Create anonymous travel behavior profiles.
- Cluster and identify travel patterns dynamically.
- Provide visual analytics for transport intelligence.

## Features
- **Live Data Architecture**: Connects to the MBTA V3 open data API.
- **Privacy Considerations**: Only uses anonymous indicators, never PII.
- **Data Preprocessing**: Handles missing values, scaling, and encoding automatically.
- **K-Means Methodology**: Implements scalable clustering to group behavior.
- **Silhouette Score & Elbow Method**: Allows evaluation of optimal clusters.
- **Dashboard Features**: Interactive maps, KPIs, and Plotly charts.

## Technologies
- Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Requests

## Project Structure
```text
TransitIQ/
├── app.py
├── data_loader.py
├── ml_model.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_data/
    └── demo_transport_data.csv
```

## Local Installation
```bash
git clone <repository_url>
cd TransitIQ
pip install -r requirements.txt
```

## Running Instructions
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment
1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Click "New app" and select your repository, branch, and `app.py` as the main file.
4. Click Deploy.

## API Configuration
TransitIQ currently uses the free MBTA V3 API. It does not strictly require an API key for basic rate limits.
However, if you want to configure an API key for higher limits (e.g., using another provider), create `.streamlit/secrets.toml`:
```toml
[api]
transport_api_key = "YOUR_KEY_HERE"
```

## Limitations & Future Enhancements
- Currently limited to MBTA data structure for live fetch.
- Future enhancements could include real-time prediction and anomaly detection.
