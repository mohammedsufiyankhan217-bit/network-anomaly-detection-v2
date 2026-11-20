"""
Reusable UI Components and Cyber-Themed Chart Utilities
for Network Anomaly Detection V2.
"""
from typing import List, Optional
import streamlit as st
import plotly.graph_objects as go

# Unified Cyber Dark Theme Palettes
CYBER_PALETTE = [
    "#38bdf8",  # Cyber Light Blue / Cyan
    "#10b981",  # Emerald Green (Benign)
    "#ef4444",  # Crimson Red (Attack)
    "#f59e0b",  # Amber / Warning
    "#8b5cf6",  # Violet
    "#ec4899",  # Neon Pink
    "#06b6d4",  # Electric Cyan
    "#f97316",  # Intense Orange
    "#64748b",  # Slate
    "#a855f7",  # Purple
    "#14b8a6",  # Teal
    "#e11d48",  # Rose
]

TRAFFIC_COLOR_MAP = {
    "BENIGN": "#10b981",
    "ATTACK": "#ef4444",
    "Benign Traffic": "#10b981",
    "DoS / DDoS": "#ef4444",
    "Reconnaissance / Probe": "#f59e0b",
    "Brute Force": "#f97316",
    "Web Attacks": "#8b5cf6",
    "Botnet": "#ec4899",
    "Infiltration / Compromise": "#06b6d4",
    "Vulnerability Exploit": "#e11d48",
}


def render_chart_header(
    title: str,
    chart_types: List[str],
    key: str,
    default: Optional[str] = None,
    help_text: Optional[str] = None,
) -> str:
    """
    Renders a compact cyber-styled header row with the chart title on the left
    and a chart-type selector on the right. Immediately reactive without reloading dataset.
    """
    widget_key = f"{key}_sel"
    
    # Determine initial value safely
    if key not in st.session_state:
        st.session_state[key] = default if default and default in chart_types else chart_types[0]
    elif st.session_state[key] not in chart_types:
        st.session_state[key] = default if default and default in chart_types else chart_types[0]

    current_val = st.session_state[key]
    idx = chart_types.index(current_val) if current_val in chart_types else 0

    c_title, c_sel = st.columns([2.2, 1.4])
    with c_title:
        st.markdown(f"#### {title}")
        if help_text:
            st.caption(help_text)
    with c_sel:
        selected_type = st.selectbox(
            f"Select visualization for {title}",
            options=chart_types,
            index=idx,
            key=widget_key,
            label_visibility="collapsed",
            help="Switch visualization format",
        )
        st.session_state[key] = selected_type

    return selected_type


def apply_cyber_theme(
    fig: go.Figure,
    height: int = 420,
    margin: Optional[dict] = None,
    show_legend: bool = True,
    legend_orientation: str = "h",
) -> go.Figure:
    """
    Applies the unified cybersecurity dark aesthetic to Plotly figures.
    """
    if margin is None:
        margin = dict(l=40, r=20, t=30, b=40)

    layout_update = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#f1f5f9",
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        ),
        margin=margin,
        height=height,
    )

    if show_legend:
        if legend_orientation == "h":
            layout_update["legend"] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#94a3b8", size=11),
            )
        else:
            layout_update["legend"] = dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(color="#94a3b8", size=11),
            )
    else:
        layout_update["showlegend"] = False

    fig.update_layout(**layout_update)
    fig.update_xaxes(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"))
    fig.update_yaxes(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"))
    return fig
