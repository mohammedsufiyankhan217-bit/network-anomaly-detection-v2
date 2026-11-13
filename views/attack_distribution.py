"""
Attack Distribution & Threat Taxonomy View.
Detailed analytics on the 15 real attack classes captured in CIC-IDS2017.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import BENIGN_LABEL, ATTACK_CATEGORIES, CATEGORY_COLORS


def render_attack_distribution_view(df_sample: pd.DataFrame, status: dict):
    st.markdown("### 🎯 Attack Distribution & Threat Taxonomy")
    st.caption("Comprehensive analysis of all 15 network traffic classifications.")

    label_dist = status.get("label_distribution", {})
    total_records = status.get("total_rows", len(df_sample))

    # Top level metrics
    total_attacks = sum(v for k, v in label_dist.items() if k != BENIGN_LABEL)
    top_attack = max((k, v) for k, v in label_dist.items() if k != BENIGN_LABEL)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Total Malicious Intrusion Flows",
            f"{total_attacks:,}",
            help="Total attack packets across all 8 days",
        )
    with c2:
        st.metric(
            "Primary Attack Vector",
            top_attack[0],
            f"{top_attack[1]:,} occurrences",
        )
    with c3:
        st.metric(
            "Attack Classification Ratio",
            f"{(total_attacks / total_records) * 100:.2f}%",
            "Severe class imbalance",
        )

    st.markdown("---")

    # Threat Taxonomy Chart & Distribution
    col_chart1, col_chart2 = st.columns([1.2, 1])

    with col_chart1:
        st.markdown("#### 🛡️ Attack Category Breakdown (Excluding Benign)")
        # Filter out benign to see the attack proportions clearly
        attack_only_dist = {k: v for k, v in label_dist.items() if k != BENIGN_LABEL}
        df_attack_bars = pd.DataFrame([
            {"Attack Type": k, "Flows": v, "Percentage": (v / total_attacks) * 100}
            for k, v in attack_only_dist.items()
        ]).sort_values("Flows", ascending=True)

        fig_bar = px.bar(
            df_attack_bars,
            x="Flows",
            y="Attack Type",
            orientation="h",
            text="Flows",
            color="Flows",
            color_continuous_scale=[[0.0, "#0284c7"], [0.5, "#f97316"], [1.0, "#ef4444"]],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", title="Flow Count (Log Scale)", type="log"),
            yaxis=dict(gridcolor="#1e293b", title=""),
            coloraxis_showscale=False,
            margin=dict(l=150, r=20, t=20, b=40),
            height=480,
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🧬 Threat Category Taxonomy")
        # Aggregate by Threat Category
        category_sums = {}
        for attack, count in label_dist.items():
            cat = ATTACK_CATEGORIES.get(attack, "Other Attacks")
            category_sums[cat] = category_sums.get(cat, 0) + count

        df_cats = pd.DataFrame([
            {"Category": k, "Count": v} for k, v in category_sums.items()
        ])
        fig_donut = px.pie(
            df_cats,
            names="Category",
            values="Count",
            hole=0.6,
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
            margin=dict(l=10, r=10, t=20, b=20),
            height=480,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # Complete Taxonomy Detail Table
    st.markdown("#### 📋 Full Attack Taxonomy & Threat Level Matrix")
    matrix_rows = []
    for attack, count in label_dist.items():
        cat = ATTACK_CATEGORIES.get(attack, "Other Attacks")
        pct_all = (count / total_records) * 100
        if attack == BENIGN_LABEL:
            severity = "BASELINE"
            action = "Authorized production traffic"
        elif "DoS" in attack or "DDoS" in attack or "Heartbleed" in attack:
            severity = "CRITICAL"
            action = "Rate limiting, scrubbing center, patch CVE-2014-0160"
        elif "Patator" in attack or "Infiltration" in attack:
            severity = "HIGH"
            action = "Account lockout, fail2ban, host isolation"
        elif "Web Attack" in attack:
            severity = "HIGH"
            action = "WAF inspection, input sanitization, CSP rules"
        elif "Bot" in attack:
            severity = "HIGH"
            action = "C2 domain block, endpoint forensics"
        else:
            severity = "MEDIUM"
            action = "Firewall port blocking, drop stealth probes"

        matrix_rows.append({
            "Classification": attack,
            "Threat Category": cat,
            "Total Occurrences": f"{count:,}",
            "Share of Total (%)": f"{pct_all:.4f}%",
            "Threat Severity": severity,
            "Mitigation Protocol": action,
        })

    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
