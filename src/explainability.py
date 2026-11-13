"""
Explainability, feature importance, and cybersecurity domain interpretation engine
for Network Anomaly Detection V2.
"""
from typing import Dict, List, Tuple, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Detailed Cybersecurity Domain Knowledge Base for CIC-IDS2017 Features
FEATURE_DOMAIN_KNOWLEDGE = {
    "Destination Port": (
        "Target service port. Essential for differentiating normal services (80/443 HTTP/S, 53 DNS) "
        "from targeted exploit ports (21 FTP, 22 SSH) and random high ports scanned during PortScans."
    ),
    "Flow Duration": (
        "Total elapsed time of bidirectional communication in microseconds. Volumetric attacks (DDoS) "
        "and PortScans often produce microscopic flow durations, while slowloris attacks maintain long lingering flows."
    ),
    "Init_Win_bytes_forward": (
        "TCP initial window size in forward packets. Distinctive OS fingerprinting artifact. "
        "Automated attack tools (such as nmap, scapy, hping) often forge or use default non-standard window sizes."
    ),
    "Init_Win_bytes_backward": (
        "TCP initial window size in backward responses. Reveals target server resource state and "
        "exhaustion under SYN flood or connection starvation attacks."
    ),
    "Total Length of Fwd Packets": (
        "Cumulative byte volume transmitted from client to server. Zero or near-zero in reconnaissance "
        "probes; massive in payload injection, exfiltration, or HTTP floods."
    ),
    "Fwd Packet Length Max": (
        "Maximum packet size observed in forward direction. Detects oversized exploit payloads, "
        "fragmentation attacks, and buffer overflow attempts."
    ),
    "Packet Length Mean": (
        "Average packet size in the entire flow. Distinguishes small interactive command traffic "
        "from bulk data transfer and fixed-size flood packets."
    ),
    "Average Packet Size": (
        "Mean size across all transmitted packets in flow. Provides key statistical signature "
        "for identifying synthetic bot traffic."
    ),
    "Bwd Packet Length Mean": (
        "Average response packet size returned by the server. Low or zero in unacknowledged denial-of-service "
        "attacks where the victim server is unable to respond."
    ),
    "Flow Packets/s": (
        "Packet transmission velocity per second. Sudden surges indicate aggressive volumetric DDoS "
        "or high-rate port scanning activity."
    ),
    "Flow Bytes/s": (
        "Network bandwidth consumption rate. Critical metric for detecting pipe-saturating denial-of-service attacks."
    ),
    "Flow IAT Mean": (
        "Mean inter-arrival time between packets. Microsecond spacing indicates automated scripted floods; "
        "human browsing generates bursty, irregular inter-arrival times."
    ),
    "ACK Flag Count": (
        "Number of TCP packets with ACK flag set. Anomalous ACK-only floods or absence of ACKs "
        "signals TCP state machine manipulation."
    ),
    "SYN Flag Count": (
        "Number of TCP packets initiating connections. Outlier counts highlight classic SYN Flood DoS attacks."
    ),
    "FIN Flag Count": (
        "Count of connection termination packets. Stealth scans (FIN scan) abuse FIN packets without prior handshakes."
    ),
    "PSH Flag Count": (
        "Count of push flags forcing immediate buffer flushing. High PSH counts indicate active interactive exploit execution."
    ),
    "Down/Up Ratio": (
        "Ratio of received to transmitted packets. Pronounced asymmetry indicates DNS/NTP amplification "
        "or one-way spoofed attack traffic."
    ),
    "min_seg_size_forward": (
        "Minimum segment size in forward direction. Used to identify anomalous TCP MSS options forged by exploit tools."
    ),
}


def plot_feature_importance(
    feature_importances: List[Tuple[str, float]],
    top_n: int = 15,
) -> go.Figure:
    """
    Render an interactive Plotly horizontal bar chart of top discriminative features.
    """
    top_feats = feature_importances[:top_n]
    names = [f[0] for f in top_feats][::-1]
    scores = [f[1] for f in top_feats][::-1]

    fig = go.Figure(
        data=go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[
                    [0.0, "#0284c7"],
                    [0.5, "#38bdf8"],
                    [1.0, "#67e8f9"],
                ],
            ),
            text=[f"{s:.4f}" for s in scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Discriminative Network Flow Features (Gini Importance)",
            font=dict(color="#f1f5f9", size=15),
        ),
        xaxis=dict(
            title=dict(text="Feature Importance Score", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            tickfont=dict(color="#f1f5f9", size=11),
            gridcolor="#1e293b",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=180, r=30, t=50, b=40),
        height=max(400, top_n * 25),
    )
    return fig


def get_feature_explanation(feature_name: str) -> str:
    """
    Retrieve security analyst explanation for a specific feature.
    """
    return FEATURE_DOMAIN_KNOWLEDGE.get(
        feature_name,
        "Statistical network flow metric computed over bidirectional packet headers and inter-arrival intervals.",
    )
