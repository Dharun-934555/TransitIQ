import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="TransitIQ | Public Transport Analytics",
        page_icon="🚆",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_header():
    st.title("TransitIQ")
    st.subheader("Real-Time Public Transport Intelligence")
    
def render_footer():
    st.markdown("---")
    st.markdown("**TransitIQ — Public Transport Intelligence & Behavioral Analytics**")
