import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

MODEL_DIR = "models"
SCORED_PATH = os.path.join(MODEL_DIR, "scored_test_logs.parquet")
EXPLAINED_PATH = os.path.join(MODEL_DIR, "explained_alerts.parquet")

st.set_page_config(
    page_title="Behavioral Threat Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# MITRE ATT&CK mapping — verified against attack.mitre.org (IT + ICS/OT)
# ---------------------------------------------------------------------------
MITRE_ATTACK_MAP = {
    "brute_force": "T1110 / T0812",
    "lateral_movement": "T1021 / T0866",
    "impossible_travel": "T1078.004 / T0859",
    "device_spoof": "T1563 / T0830",
    "normal": "—",
}
MITRE_ATTACK_NAMES = {
    "brute_force": "Brute Force · Default Credentials",
    "lateral_movement": "Remote Services · Exploitation of Remote Services",
    "impossible_travel": "Cloud Accounts · Valid Accounts",
    "device_spoof": "Session Hijacking · Adversary-in-the-Middle",
    "normal": "Baseline traffic",
}
SEVERITY_MAP = {
    "brute_force": ("high", "High"),
    "lateral_movement": ("critical", "Critical"),
    "impossible_travel": ("high", "High"),
    "device_spoof": ("medium", "Medium-High"),
    "normal": ("benign", "Benign"),
}

# ---------------------------------------------------------------------------
# Design tokens + global styling (including fixes for multiselect tags)
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0A0D12; --panel:#12161D; --panel-alt:#161B24;
  --border:rgba(255,255,255,0.08); --border-soft:rgba(255,255,255,0.05);
  --text:#E6E9EF; --text-muted:#7C8595; --text-dim:#4E5566;
  --accent:#3FD3C6; --accent-dim:rgba(63,211,198,0.12);
  --crit:#FF4D6D; --crit-dim:rgba(255,77,109,0.14);
  --high:#FF8A3D; --high-dim:rgba(255,138,61,0.14);
  --med:#FFCB47; --med-dim:rgba(255,203,71,0.14);
  --benign:#3ADD8C; --benign-dim:rgba(58,221,140,0.14);
}
.stApp{ background:var(--bg); }
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--text); }
.mono{ font-family:'IBM Plex Mono', monospace; }
.display{ font-family:'Space Grotesk', sans-serif; }

/* Header */
.console-header{ display:flex; align-items:center; gap:14px; margin-bottom:2px; }
.console-header h1{
  font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.65rem;
  margin:0; letter-spacing:-0.01em;
}
.console-sub{ color:var(--text-muted); font-size:0.92rem; margin:6px 0 22px 0; max-width:760px; }

/* KPI instrument cards */
.kpi-row{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:8px; }
.kpi-card{
  background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:16px 18px; position:relative; overflow:hidden;
}
.kpi-card .kpi-icon{ position:absolute; top:14px; right:14px; opacity:0.35; }
.kpi-label{
  font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--text-muted); font-weight:500; margin-bottom:10px;
}
.kpi-value{ font-family:'IBM Plex Mono', monospace; font-size:1.9rem; font-weight:600; line-height:1; }
.kpi-delta{ font-size:0.78rem; margin-top:8px; color:var(--text-dim); font-family:'IBM Plex Mono', monospace; }
.kpi-delta.pos{ color:var(--accent); }

/* Advisory panel */
.advisory{
  border:1px solid var(--border); background:var(--panel-alt); border-left:3px solid var(--accent);
  border-radius:8px; padding:12px 14px; font-size:0.85rem; color:var(--text-muted);
  margin:14px 0 20px 0; line-height:1.5;
}
.advisory.warn{ border-left-color:var(--med); }
.advisory b{ color:var(--text); font-weight:600; }
.advisory code{ background:rgba(255,255,255,0.06); padding:1px 5px; border-radius:4px; font-size:0.82rem; }

/* Badges */
.badge{
  display:inline-block; padding:2px 9px; border-radius:20px; font-size:0.72rem;
  font-weight:600; letter-spacing:0.01em; white-space:nowrap;
}
.badge.critical{ background:var(--crit-dim); color:var(--crit); }
.badge.high{ background:var(--high-dim); color:var(--high); }
.badge.medium{ background:var(--med-dim); color:var(--med); }
.badge.benign{ background:var(--benign-dim); color:var(--benign); }
.tag{
  display:inline-block; background:rgba(255,255,255,0.06); color:var(--text-muted);
  font-family:'IBM Plex Mono', monospace; font-size:0.7rem; padding:2px 7px; border-radius:5px;
}

/* Masonry alert card grid */
.masonry{ column-count:3; column-gap:16px; margin-top:4px; }
.alert-card{
  break-inside:avoid; margin-bottom:16px; background:var(--panel);
  border:1px solid var(--border); border-radius:12px; padding:14px 16px 16px 16px;
  position:relative; padding-left:20px;
}
.alert-card::before{
  content:""; position:absolute; left:0; top:12px; bottom:12px; width:4px; border-radius:3px;
}
.alert-card.critical::before{ background:var(--crit); }
.alert-card.high::before{ background:var(--high); }
.alert-card.medium::before{ background:var(--med); }
.alert-card .row1{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.alert-entity{ font-family:'IBM Plex Mono', monospace; font-size:0.86rem; font-weight:600; }
.alert-risk{ font-family:'IBM Plex Mono', monospace; font-size:0.82rem; color:var(--text-muted); }
.alert-technique{ font-size:0.74rem; color:var(--text-dim); margin:6px 0 8px 0; }
.alert-reason{ font-size:0.82rem; color:var(--text-muted); line-height:1.45; }
.alert-time{ font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-dim); margin-top:10px; }

/* Streamlit Widget Fixes (Multiselect & Sliders) */
.stMultiSelect span[data-baseweb="tag"] {
  background-color: var(--accent-dim) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(63,211,198,0.3) !important;
}
.stMultiSelect span[data-baseweb="tag"] svg {
  fill: var(--accent) !important;
}

/* Tabs */
button[data-baseweb="tab"]{
  font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:0.85rem;
  text-transform:uppercase; letter-spacing:0.04em; color:var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--text) !important; }
div[data-baseweb="tab-highlight"]{ background-color:var(--accent) !important; }
div[data-baseweb="tab-border"]{ background-color:var(--border) !important; }

section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--border); }
.sidebar-label{
  font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--text-muted); font-weight:600; margin:18px 0 4px 0;
}
hr{ border-color:var(--border); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper primitives & UI components
# ---------------------------------------------------------------------------
def html(s: str) -> str:
    return re.sub(r">\s+<", "><", " ".join(line.strip() for line in s.strip().splitlines()))

def icon(name, size=20, color="currentColor"):
    s = f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'
    paths = {
        "shield": f'<svg {s}><path d="M12 3l7 3v6c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6l7-3z"/></svg>',
        "layers": f'<svg {s}><path d="M12 4l8 4-8 4-8-4 8-4z"/><path d="M4 13l8 4 8-4"/></svg>',
        "target": f'<svg {s}><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor"/></svg>',
        "check": f'<svg {s}><circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.2l2.4 2.4 4.6-5"/></svg>',
        "pulse": f'<svg {s}><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>',
    }
    return paths.get(name, "")

def kpi_card(icon_name, label, value, delta_text, positive=True):
    delta_class = "pos" if positive else ""
    return html(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon(icon_name, 22)}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {delta_class}">{delta_text}</div>
    </div>
    """)

def severity_badge(sev_key, sev_label):
    return f'<span class="badge {sev_key}">{sev_label}</span>'

def alert_card_html(row, risk_col):
    sev_key, sev_label = SEVERITY_MAP.get(row["predicted_label"], ("medium", "Medium"))
    technique = MITRE_ATTACK_NAMES.get(row["predicted_label"], "")
    ids = MITRE_ATTACK_MAP.get(row["predicted_label"], "")
    reason = row.get("soc_explanation", "")
    ts = row.get("timestamp", "")
    return html(f"""
    <div class="alert-card {sev_key}">
      <div class="row1">
        <span class="alert-entity">{row['entity_id']}</span>
        {severity_badge(sev_key, sev_label)}
      </div>
      <div class="alert-technique">{technique} &middot; <span class="tag">{ids}</span></div>
      <div class="alert-reason">{reason}</div>
      <div class="row1" style="margin-top:10px; margin-bottom:0;">
        <span class="alert-time">{ts}</span>
        <span class="alert-risk">RISK {row[risk_col]:.1f}</span>
      </div>
    </div>
    """)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#12161D", plot_bgcolor="#12161D",
        font=dict(family="Inter, sans-serif", color="#E6E9EF", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=["#3FD3C6", "#FF8A3D", "#FF4D6D", "#FFCB47", "#7C8595"],
    )
)

# ---------------------------------------------------------------------------
# Data loading & Schema resolution
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(SCORED_PATH):
        return None, "missing_scored"
    df = pd.read_parquet(SCORED_PATH)
    
    risk_col = "risk_score" if "risk_score" in df.columns else "unsupervised_risk_score"
    if risk_col not in df.columns:
        df["risk_score"] = 0.0
        risk_col = "risk_score"
        
    if os.path.exists(EXPLAINED_PATH):
        df_explained = pd.read_parquet(EXPLAINED_PATH)
        if "event_id" in df.columns and "event_id" in df_explained.columns:
            df = df.merge(df_explained[["event_id", "soc_explanation"]], on="event_id", how="left")
        else:
            df["soc_explanation"] = pd.NA
            df.loc[: len(df_explained) - 1, "soc_explanation"] = df_explained["soc_explanation"].values
        explained_coverage = len(df_explained)
    else:
        df["soc_explanation"] = pd.NA
        explained_coverage = 0
        
    df["soc_explanation"] = df["soc_explanation"].fillna(
        "SHAP explanation not pre-computed at this budget size."
    )
    df["mitre_technique"] = df["predicted_label"].map(MITRE_ATTACK_NAMES)
    
    entity_risk = df.groupby("entity_id")[risk_col].max().reset_index()
    entity_risk = entity_risk.sort_values(by=risk_col, ascending=False)
    ranked_entities = entity_risk["entity_id"].tolist()
    
    return {"df": df, "explained_coverage": explained_coverage, "risk_col": risk_col, "ranked_entities": ranked_entities}, None

result, error = load_data()
if error == "missing_scored":
    st.markdown(
        f'<div class="advisory warn"><b>Telemetry missing.</b> '
        f'Could not find <code>{SCORED_PATH}</code>. Run <code>python train_models.py</code> first.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

df_full = result["df"]
explained_coverage = result["explained_coverage"]
risk_col = result["risk_col"]
ranked_entities = result["ranked_entities"]

total_test_events = len(df_full)
total_true_attacks = int((df_full["label"] != "normal").sum())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    html(f"""
    <div class="console-header">{icon('shield', 28, '#3FD3C6')}<h1>Behavioral Threat Detection Console</h1></div>
    <div class="console-sub">Sequence and behavioral anomaly detection across utility networks,
    industrial edge devices, and enterprise IT identity telemetry.</div>
    """),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-label">Analyst review budget</div>', unsafe_allow_html=True)
budget_pct = st.sidebar.slider(
    "Percent of daily event volume", min_value=0.5, max_value=10.0, value=1.0, step=0.5,
    label_visibility="collapsed",
    help="Simulates SOC analyst headcount. Recall is capped by how many events the team can review.",
)

st.sidebar.markdown('<div class="sidebar-label">Threat class filter</div>', unsafe_allow_html=True)
available_threats = [c for c in df_full["predicted_label"].unique() if c != "normal"]
selected_types = st.sidebar.multiselect(
    "Threat class", options=available_threats, default=available_threats, label_visibility="collapsed"
)

st.sidebar.markdown('<div class="sidebar-label">Minimum risk score</div>', unsafe_allow_html=True)
min_risk = st.sidebar.slider("Minimum risk", 0.0, 100.0, 0.0, label_visibility="collapsed")

budget_count = max(1, int(total_test_events * (budget_pct / 100.0)))
if budget_count > explained_coverage:
    st.sidebar.markdown(
        f"""<div class="advisory warn"><b>Partial coverage.</b> SHAP explanations were pre-computed
        for the top <code>{explained_coverage:,}</code> events; this budget needs <code>{budget_count:,}</code>.
        Events beyond that show a placeholder. Re-run <code>explain_alerts.py</code> with a higher
        <code>alert_budget_pct</code> to cover this range.</div>""",
        unsafe_allow_html=True,
    )

df_budget = df_full.iloc[: min(budget_count, len(df_full))].copy()

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
caught = int((df_budget["label"] != "normal").sum())
recall_val = (caught / total_true_attacks * 100.0) if total_true_attacks else 0.0
false_pos = int((df_budget["label"] == "normal").sum())
fp_rate = (false_pos / len(df_budget) * 100.0) if len(df_budget) else 0.0
avg_risk = df_budget[risk_col].mean() if len(df_budget) else 0.0

kpi_html = '<div class="kpi-row">'
kpi_html += kpi_card("layers", "Investigation queue", f"{len(df_budget):,}", f"{budget_pct:.1f}% of daily volume")
kpi_html += kpi_card("target", "Recall at budget", f"{recall_val:.1f}%", f"{caught:,} / {total_true_attacks:,} caught", positive=True)
kpi_html += kpi_card("check", "Queue precision", f"{100.0 - fp_rate:.1f}%", f"{fp_rate:.2f}% false alarm rate", positive=True)
kpi_html += kpi_card("pulse", "Mean risk score", f"{avg_risk:.1f}", "scale 0 – 100")
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

if total_true_attacks > budget_count:
    max_possible = budget_count / total_true_attacks * 100.0
    st.markdown(
        f"""<div class="advisory"><b>{total_true_attacks:,} attacks</b> exist in this window against a
        <b>{budget_count:,}-event</b> budget — maximum achievable recall here is <b>{max_possible:.1f}%</b>
        regardless of model quality. A low false-alarm rate at this ceiling means the model is ranking
        correctly; the constraint is analyst headcount, not detection quality.</div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Alert Queue", "Entity Timeline", "ATT&CK Mapping"])

# ---------------------------------------------------------------------------
# Tab 1 — Masonry featured cards + Full sortable table
# ---------------------------------------------------------------------------
with tab1:
    df_display = df_budget[
        df_budget["predicted_label"].isin(selected_types) & (df_budget[risk_col] >= min_risk)
    ]

    st.markdown(
        '<div class="sidebar-label" style="margin-top:6px;">Highest-priority alerts</div>',
        unsafe_allow_html=True,
    )
    
    if df_display.empty:
        st.markdown('<div class="advisory">No alerts match your current filter criteria. Adjust the threat class or risk slider.</div>', unsafe_allow_html=True)
    else:
        featured = df_display.head(12)
        cards_html = '<div class="masonry">' + "".join(
            alert_card_html(row, risk_col) for _, row in featured.iterrows()
        ) + "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        with st.expander(f"View full queue — {len(df_display):,} events (sortable table)"):
            display_cols = ["timestamp", "entity_id", "entity_type", "predicted_label",
                            risk_col, "mitre_technique", "soc_explanation"]
            display_cols = [c for c in display_cols if c in df_display.columns]
            
            st.dataframe(
                df_display[display_cols],
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                    "entity_id": st.column_config.TextColumn("Entity ID"),
                    "entity_type": st.column_config.TextColumn("Entity Type"),
                    "predicted_label": st.column_config.TextColumn("Predicted Threat"),
                    risk_col: st.column_config.NumberColumn("Risk Score", format="%.1f"),
                    "mitre_technique": st.column_config.TextColumn("ATT&CK Mapping"),
                    "soc_explanation": st.column_config.TextColumn("Root Cause Attribution", width="large"),
                },
                use_container_width=True, 
                height=420,
                hide_index=True
            )

# ---------------------------------------------------------------------------
# Tab 2 — Entity deep dive
# ---------------------------------------------------------------------------
with tab2:
    default_entity = ranked_entities[0] if ranked_entities else sorted(df_full["entity_id"].unique())[0]
    selected_entity = st.selectbox(
        "Entity", ranked_entities, index=0, label_visibility="collapsed"
    )
    entity_logs = df_full[df_full["entity_id"] == selected_entity].sort_values("timestamp")

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(
            entity_logs, x="timestamp", y=risk_col, color="predicted_label", size=risk_col,
            hover_data=[c for c in ["mitre_technique", "soc_explanation"] if c in entity_logs.columns],
            title=f"Risk trajectory — {selected_entity}",
            color_discrete_map={"normal": "#3ADD8C", "brute_force": "#FF4D6D",
                                "impossible_travel": "#FF8A3D", "lateral_movement": "#FFCB47",
                                "device_spoof": "#3FD3C6"},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Risk score", xaxis_title=None, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        anom_count = int((entity_logs["predicted_label"] != "normal").sum())
        st.markdown(
            f"""
            <div class="alert-card medium" style="break-inside:auto;">
              <div class="kpi-label">Entity type</div>
              <div class="mono" style="margin-bottom:12px;">{entity_logs.iloc[0]['entity_type']}</div>
              <div class="kpi-label">Events in window</div>
              <div class="mono" style="margin-bottom:12px;">{len(entity_logs):,}</div>
              <div class="kpi-label">Peak risk score</div>
              <div class="mono" style="margin-bottom:12px;">{entity_logs[risk_col].max():.1f} / 100</div>
              <div class="kpi-label">Flagged intrusions</div>
              <div class="mono">{anom_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if anom_count > 0:
            latest = entity_logs[entity_logs["predicted_label"] != "normal"].iloc[-1]
            sev_key, sev_label = SEVERITY_MAP.get(latest["predicted_label"], ("medium", "Medium"))
            st.markdown(
                f"""<div class="alert-card {sev_key}" style="break-inside:auto; margin-top:12px;">
                <div class="row1">{severity_badge(sev_key, sev_label)}
                <span class="alert-time">{latest['timestamp']}</span></div>
                <div class="alert-reason">{latest['soc_explanation']}</div></div>""",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Tab 3 — ATT&CK landscape (Fixed Bar Chart Styling)
# ---------------------------------------------------------------------------
with tab3:
    attack_only = df_budget[df_budget["predicted_label"] != "normal"]
    c_left, c_right = st.columns(2)
    with c_left:
        if len(attack_only):
            counts = attack_only["mitre_technique"].value_counts().reset_index()
            counts.columns = ["technique", "count"]
            
            # Use Plotly bar with explicit marker color to fix the solid blue box issue
            fig_bar = px.bar(
                counts, x="count", y="technique", orientation="h",
                title="Alerts by ATT&CK technique", text="count",
                labels={"count": "Alert volume", "technique": ""}
            )
            fig_bar.update_traces(
                marker_color="#3FD3C6", 
                textposition="outside",
                marker_line_width=0
            )
            fig_bar.update_layout(
                template=PLOTLY_TEMPLATE, 
                height=340,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown('<div class="advisory">No attacks in the current filtered selection.</div>', unsafe_allow_html=True)
            
    with c_right:
        if len(attack_only):
            type_counts = attack_only["entity_type"].value_counts().reset_index()
            type_counts.columns = ["entity_type", "count"]
            fig_pie = px.pie(type_counts, values="count", names="entity_type",
                             title="Compromised entity types", hole=0.55)
            fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=340)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown('<div class="advisory">No attacks in the current filtered selection.</div>', unsafe_allow_html=True)