import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

st.set_page_config(layout="wide", page_title="Zero Trust SRE Gym", page_icon=None)

# --- Professional CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0d1117;
    }
    
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 4px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #c9d1d9;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #c9d1d9;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    
    .stAlert {
        border-radius: 4px;
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "https://aditi75432-zero-trust-safe-SRE-gym.hf.space"
REFRESH_INTERVAL = 1.5

@st.cache_data(ttl=1)
def fetch(endpoint):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", timeout=2)
        if r.status_code == 200:
            return r.json(), True
    except:
        pass
    return {}, False

def build_graph(nodes):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    edges = [
        ("api_gateway", "frontend"),
        ("api_gateway", "payment"),
        ("auth_service", "frontend"),
        ("auth_service", "payment"),
        ("frontend", "hr_db"),
    ]
    for e in edges:
        if e[0] in nodes and e[1] in nodes:
            G.add_edge(*e)
    return G

def draw_topology(nodes_state):
    if not nodes_state:
        return go.Figure()
    nodes = list(nodes_state.keys())
    G = build_graph(nodes)
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#30363d")
    )
    node_x, node_y, colors, texts = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        state = nodes_state.get(n, {}).get("status", "unknown")
        if state == "healthy": color = "#238636"
        elif state == "compromised": color = "#da3633"
        elif state == "isolated": color = "#d29922"
        else: color = "#484f58"
        colors.append(color)
        texts.append(f"{n}<br>Status: {state}")
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        hovertext=texts,
        textposition="bottom center",
        marker=dict(size=22, color=colors, line=dict(width=2, color="#30363d"))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#161b22",
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def draw_rewards(history):
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["cumulative_reward"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(color="#58a6ff", size=4)
    ))
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#161b22",
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        xaxis_title="Step",
        yaxis_title="Cumulative Reward",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def render_services(services):
    for name, info in services.items():
        status_color = {
            "healthy": "#238636",
            "degraded": "#d29922",
            "compromised": "#da3633",
            "offline": "#484f58"
        }.get(info.get("status", "unknown"), "#484f58")
        st.markdown(
            f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
            f"<span style='background:{status_color}; width:10px; height:10px; border-radius:50%; margin-right:8px;'> </span>"
            f"<span style='color:#c9d1d9;'>{name}</span>"
            f"<span style='color:#8b949e; margin-left:auto;'>{info.get('status', 'unknown')} | latency={info.get('latency_ms', 'N/A')}ms</span>"
            f"</div>",
            unsafe_allow_html=True
        )

def render_alerts(alerts):
    for a in alerts:
        sev_color = "#da3633" if a['severity'] == 'FATAL' else ("#d29922" if a['severity'] == 'WARNING' else "#8b949e")
        st.markdown(
            f"<div style='background:#161b22; border-left:3px solid {sev_color}; padding:8px; margin-bottom:6px; border-radius:4px;'>"
            f"<span style='color:{sev_color}; font-weight:600;'>{a['severity']}</span> "
            f"<span style='color:#c9d1d9;'>{a['target_node']}</span>"
            f"<div style='color:#8b949e; font-size:12px;'>{a['symptom']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

def render_audit(history):
    for h in reversed(history[-20:]):
        st.markdown(
            f"<div style='color:#8b949e; font-size:12px; margin-bottom:2px;'>"
            f"<span style='color:#c9d1d9;'>{h['step']}</span> | "
            f"{h['action']} | {h['target']} | "
            f"<span style='color:#58a6ff;'>reward={h['reward']:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

state, ok1 = fetch("state")
history_data, ok2 = fetch("history")
services, ok3 = fetch("services")

history = history_data.get("history", []) if ok2 else []

st.markdown("<h1 style='color:#c9d1d9; font-weight:600;'>Zero Trust SRE Gym</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;'>Enterprise Security Operations | Autonomous Containment Agent</p>", unsafe_allow_html=True)

if not ok1:
    st.error("Environment not reachable")
    st.stop()

st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S UTC')}")

# Top metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Uptime", f"{state.get('global_uptime', 100):.1f}%")
with m2:
    st.metric("Episode Reward", f"{state.get('episode_reward', 0):.2f}")
with m3:
    st.metric("Difficulty", state.get("curriculum", {}).get("difficulty", "warmup").upper())
with m4:
    st.metric("Active Ticket", state.get("active_ticket_id") or "None")

# Dashboard panels
row1_left, row1_right = st.columns(2)
with row1_left:
    st.markdown('<div class="section-header">Live Threat Intelligence</div>', unsafe_allow_html=True)
    cve = state.get("cve_context", "No active CVE data.")
    siem = state.get("siem_evidence_template", "Awaiting scenario generation.")
    st.text(f"CVE Context:\n{cve}\n\nSIEM Template:\n{siem}")

with row1_right:
    st.markdown('<div class="section-header">Compliance Judge Audit</div>', unsafe_allow_html=True)
    judge_persona = state.get("judge_persona", "senior").upper()
    st.markdown(f"Current persona: **{judge_persona}**")
    st.text_area("Last Justification Feedback", state.get("command_output", "No action taken yet."), height=200)

# Service Topology
st.markdown('<div class="section-header">Service Topology</div>', unsafe_allow_html=True)
st.plotly_chart(draw_topology(state.get("nodes", {})), use_container_width=True)

# Three-column microservices, alerts, SIEM output
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="section-header">Microservices</div>', unsafe_allow_html=True)
    if ok3:
        render_services(services)
    else:
        st.write("Service data unavailable.")

with col2:
    st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
    render_alerts(state.get("active_alerts", []))

with col3:
    st.markdown('<div class="section-header">SIEM Output</div>', unsafe_allow_html=True)
    st.caption(state.get("command_output", "No SIEM query yet."))

# Reward Trajectory
st.markdown('<div class="section-header">Reward Trajectory</div>', unsafe_allow_html=True)
st.plotly_chart(draw_rewards(history), use_container_width=True)

# Agent Actions
st.markdown('<div class="section-header">Agent Action Log</div>', unsafe_allow_html=True)
render_audit(history)

# Control buttons
cA, cB = st.columns(2)
with cA:
    if st.button("Reset Environment"):
        requests.post(f"{API_URL}/reset", json={"task_id": "auto"})
with cB:
    auto = st.toggle("Auto Refresh", value=True)

if auto:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()