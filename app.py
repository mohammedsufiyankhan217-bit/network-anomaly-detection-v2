"""
Network Anomaly Detection V2
Main Streamlit Application with Official CIC-IDS2017 Dataset Integration.
"""
import streamlit as st
import pandas as pd

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    DATASET_NAME,
    DATASET_SOURCE_URL,
    TOTAL_DATASET_ROWS,
)
from src.data_loader import get_dataset_status, load_stratified_sample
from views.overview import render_overview_view
from views.attack_distribution import render_attack_distribution_view
from views.dataset_explorer import render_dataset_explorer_view
from views.ml_evaluation import render_ml_evaluation_view
from views.detection import render_detection_view
from views.feature_importance import render_feature_importance_view
from views.reports_view import render_reports_view

# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber Dark CSS Styling
st.markdown(
    """
    <style>
    /* Dark Cybersecurity Backgrounds */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1526;
        border-right: 1px solid #1e293b;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* Status Badge Card */
    .status-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .status-badge-active {
        display: inline-block;
        background-color: #064e3b;
        color: #34d399;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 9999px;
        border: 1px solid #059669;
        margin-bottom: 6px;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Clean Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_cached_status():
    return get_dataset_status()


@st.cache_data(show_spinner=False)
def get_cached_sample(sample_size: int):
    return load_stratified_sample(sample_size=sample_size)


def main():
    # Load dataset status
    status = get_cached_status()

    # Sidebar Navigation & Telemetry
    with st.sidebar:
        st.markdown(f"### 🛡️ {APP_TITLE}")
        st.caption(f"v{APP_VERSION} | Enterprise Anomaly Detection")

        # Dataset Status Indicator Card
        st.markdown(
            f"""
            <div class="status-card">
                <span class="status-badge-active">● REAL DATASET ACTIVE</span>
                <div style="font-size: 0.82rem; color: #e2e8f0; font-weight: 600; margin-top: 4px;">
                    {DATASET_NAME}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                    Verified: <b>{status['found_count']}/{status['total_expected']} daily CSVs</b> ({status['total_size_mb']:.1f} MB)<br>
                    Total Records: <b>{status['total_rows']:,} network flows</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📁 Verified Daily CSV Files", expanded=False):
            for f in status.get("found_files", []):
                st.markdown(f"- ✅ `{f}`")

        st.markdown("---")
        st.markdown("#### 🧭 Navigation")
        nav_options = [
            "Executive Overview",
            "Attack Distribution",
            "Dataset Explorer",
            "ML Training & Evaluation",
            "Live Anomaly Detection",
            "Feature Importance",
            "Security Reports & Export",
        ]
        selected_nav = st.radio(
            "Select Section",
            nav_options,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### ⚙️ Sampling Configuration")
        active_sample_size = st.select_slider(
            "Active Sample Size (Rows)",
            options=[25000, 50000, 100000, 200000],
            value=100000,
            help="Stratified sample used for high-speed dashboard telemetry and initial modeling.",
        )
        st.caption(
            f"Active: **{active_sample_size:,}** stratified flows with 100% preservation of rare attack classes."
        )

        st.markdown("---")
        st.caption(
            f"[CIC-IDS2017 Dataset Source]({DATASET_SOURCE_URL})<br>"
            "Canadian Institute for Cybersecurity",
            unsafe_allow_html=True,
        )

    # Load active sample
    with st.spinner("Loading stratified CIC-IDS2017 network flows..."):
        df_sample = get_cached_sample(active_sample_size)

    # View Routing
    if selected_nav == "Executive Overview":
        render_overview_view(df_sample, status)
    elif selected_nav == "Attack Distribution":
        render_attack_distribution_view(df_sample, status)
    elif selected_nav == "Dataset Explorer":
        render_dataset_explorer_view(df_sample, status)
    elif selected_nav == "ML Training & Evaluation":
        render_ml_evaluation_view(df_sample, status)
    elif selected_nav == "Live Anomaly Detection":
        render_detection_view(df_sample, status)
    elif selected_nav == "Feature Importance":
        render_feature_importance_view(df_sample, status)
    elif selected_nav == "Security Reports & Export":
        render_reports_view(df_sample, status)


if __name__ == "__main__":
    main()
