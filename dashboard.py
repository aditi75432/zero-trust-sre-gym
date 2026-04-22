import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Zero Trust SOC | Live Telemetry",
    initial_sidebar_state="collapsed"
)

# Update this to your HuggingFace Space URL
API_URL = "https://aditi75432-zero-trust-safe-SRE-gym.hf.space"
REFRESH_INTERVAL = 1.5  # Faster, smoother polling

# ─── ENTERPRISE CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Strict Dark Mode Palette - CrowdStrike / Palantir vibe */
    .stApp { background-color: #050914; color: #cbd5e1; font-family: 'Inter', monospace; }
    
    /* HUD Metric Cards */
    [data-testid="stMetric"] { 
        background: #0f172a; 
        padding: 15px; 
        border-radius: 4px; 
        border: 1px solid #1e293b; 
        border-top: 3px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 700; letter-spacing: 1.5px; font-size: 0.75rem;}
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 600; font-family: 'JetBrains Mono', monospace;}
    
    /* Terminal Feed */
    .terminal-feed {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        background: #020617;
        padding: 15px;
        border: 1px solid #1e293b;
        border-left: 3px solid #10b981;
        border-radius: 4px;
        color: #a3e635;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.8);
    }
    
    /* Banners & Alerts */
    .alert-fatal { border-left: 4px solid #ef4444; padding: 12px; background: rgba(239, 68, 68, 0.1); margin: 4px 0; font-family: monospace; color: #fca5a5; font-size: 0.85rem; font-weight: bold;}
    .alert-warn { border-left: 4px solid #f59e0b; padding: 12px; background: rgba(245, 158, 11, 0.1); margin: 4px 0; font-family: monospace; color: #fcd34d; font-size: 0.85rem;}
    .alert-info { border-left: 4px solid #3b82f6; padding: 12px; background: rgba(59, 130, 246, 0.1); margin: 4px 0; font-family: monospace; color: #93c5fd; font-size: 0.85rem;}
    
    /* Judge Evaluation Box */
    .judge-box { border: 1px solid #8b5cf6; padding: 15px; background: rgba(139, 92, 246, 0.05); border-radius: 4px; border-top: 3px solid #8b5cf6;}
    .judge-title { color: #c4b5fd; font-weight: bold; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase;}
    
    /* Headers */
    h1, h2, h3 { color: #f1f5f9; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; font-size: 1.0rem !important; margin-bottom: 1rem;}
    hr { border-color: #1e293b; }
    
    /* Connection Lost Banner */
    .offline-banner { background-color: #7f1d1d; color: white; padding: 10px; text-align: center; font-weight: bold; font-family: monospace; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── ROBUST NETWORK HANDLING ────────────────────────────────────────────────
@st.cache_data(ttl=0.5)
def fetch_api(endpoint: str):
    """Enterprise-grade error handling. Never crash the UI on bad WiFi."""
    try:
        r = requests.get(f"{API_URL}/{endpoint}", timeout=2.0)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return {}, False

def draw_service_mesh(nodes_state: dict) -> go.Figure:
    """Dynamic DAG rendering with severity scaling."""
    G = nx.DiGraph()
    G.add_edges_from([
        ("api_gateway", "frontend"), ("api_gateway", "payment"),
        ("auth_service", "frontend"), ("auth_service", "payment"),
        ("frontend", "hr_db")
    ])
    
    pos = {
        "api_gateway": (0.0, 2.0), "auth_service": (0.0, 0.0),
        "frontend": (2.0, 1.5), "payment": (2.0, 0.5), "hr_db": (4.0, 1.5)
    }
    
    # Visual mapping
    STATUS_UI = {
        "healthy": {"color": "#10b981", "size": 35, "tag": "[OK]"},
        "compromised": {"color": "#ef4444", "size": 50, "tag": "[BREACHED]"}, # Larger size for active threats
        "isolated": {"color": "#f59e0b", "size": 35, "tag": "[ISOLATED]"},
        "offline": {"color": "#334155", "size": 25, "tag": "[OFFLINE]"}
    }
    
    node_colors, hover_texts, node_texts, node_sizes = [], [], [], []
    for node in G.nodes():
        state = nodes_state.get(node, {}).get("status", "offline")
        ui = STATUS_UI.get(state, STATUS_UI["offline"])
        
        node_colors.append(ui["color"])
        node_sizes.append(ui["size"])
        node_texts.append(f"{ui['tag']} {node.upper()}")
        hover_texts.append(f"NODE: {node.upper()}<br>STATUS: {state.upper()}")
    
    # Flatten edge traces (CORRECTED SYNTAX)
    ex, ey = [], []
    for e in G.edges():
        ex.extend([pos[e[0]][0], pos[e[1]][0], None])
        ey.extend([pos[e[0]][1], pos[e[1]][1], None])
    
    edge_trace = go.Scatter(
        x=ex, y=ey, 
        line=dict(width=1.5, color="#475569"), 
        mode="lines", 
        hoverinfo="none"
    )
    
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers+text", text=node_texts, hovertext=hover_texts,
        textposition="bottom center", textfont=dict(size=11, color="#e2e8f0", family="monospace"),
        marker=dict(size=node_sizes, color=node_colors, symbol="square", line=dict(width=2, color="#050914"))
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="closest", margin=dict(b=20, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5])
    )
    return fig

def draw_reward_curve(history: list) -> go.Figure:
    if not history: return go.Figure().update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    
    colors = ["#10b981" if r >= 0 else "#ef4444" for r in df.get("reward", [])]
    
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["cumulative_reward"],
        mode="lines+markers", line=dict(color="#3b82f6", width=2, shape="vh"),
        marker=dict(size=8, color=colors, symbol="square"),
        hovertemplate="STEP: %{x}<br>NET_YIELD: %{y:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=10, family="monospace"),
        xaxis=dict(title="EXECUTION STEP", gridcolor="#1e293b", showline=True, linewidth=1, linecolor="#334155"),
        yaxis=dict(title="NET POLICY YIELD", gridcolor="#1e293b", showline=True, linewidth=1, linecolor="#334155"),
        margin=dict(b=30, l=40, r=10, t=10), height=220
    )
    return fig

# ─── UI RENDERING ────────────────────────────────────────────────────────────

state_data, state_ok = fetch_api("state")
history_data, hist_ok = fetch_api("history")

if not state_ok or not hist_ok:
    st.markdown("<div class='offline-banner'>⚠️ CRITICAL: TELEMETRY LINK SEVERED. ATTEMPTING RECONNECTION...</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='color:#10b981; font-size:0.85rem; font-family:monospace; margin-bottom: 10px;'>[ {datetime.now().strftime('%H:%M:%S')} ] SECURE CONNECTION ESTABLISHED // LIVE FEED</div>", unsafe_allow_html=True)

curriculum = state_data.get("curriculum", {})
history = history_data.get("history", [])
output = state_data.get("command_output", "WAITING FOR COMMAND...")

# ── KPI Row ──
c1, c2, c3, c4, c5, c6 = st.columns(6)
uptime = state_data.get("global_uptime", 100.0)
uptime_color = "normal" if uptime >= 90 else "inverse" if uptime > 0 else "off"

c1.metric("GLOBAL UPTIME", f"{uptime:.1f}%", delta=f"{uptime - 100:.0f}%" if uptime < 100 else "NOMINAL", delta_color=uptime_color)
c2.metric("ACTIVE TICKET", state_data.get("active_ticket_id") or "NONE")
c3.metric("AUTH STATUS", "GRANTED" if state_data.get("ticket_approved") else "LOCKED")
c4.metric("THREAT LEVEL", curriculum.get("difficulty", "WARMUP").upper())
c5.metric("LLM JUDGE", state_data.get("judge_persona", "SENIOR").upper())
c6.metric("NET YIELD", f"{state_data.get('episode_reward', 0):.1f}")

st.markdown("<hr style='margin: 10px 0 20px 0;'>", unsafe_allow_html=True)

# ── Main Layout ──
col_left, col_mid, col_right = st.columns([1.3, 1.0, 1.0])

with col_left:
    st.markdown("<h3>SERVICE MESH TOPOLOGY</h3>", unsafe_allow_html=True)
    if state_data.get("nodes"):
        st.plotly_chart(draw_service_mesh(state_data["nodes"]), use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<h3>ACTIVE THREAT INTEL & ALERTS</h3>", unsafe_allow_html=True)
    alerts = state_data.get("active_alerts", [])
    if alerts:
        for alert in alerts:
            sev = alert.get("severity", "INFO")
            css_class = "alert-fatal" if sev in ["FATAL", "CRITICAL"] else "alert-warn"
            # Highlight CVEs if they exist in the symptom
            symptom = alert['symptom'].replace("CVE", "<strong style='color:#fff'>CVE</strong>")
            st.markdown(f"<div class='{css_class}'>[{sev}] {alert['target_node'].upper()} :: {symptom}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-info'>[INFO] NO ACTIVE ALERTS.</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown("<h3>POLICY OPTIMIZATION TRAJECTORY</h3>", unsafe_allow_html=True)
    st.plotly_chart(draw_reward_curve(history), use_container_width=True, config={'displayModeBar': False})
    
    # Expose the LLM Judge explicitly
    st.markdown("<h3>LLM CHANGE BOARD EVALUATION</h3>", unsafe_allow_html=True)
    if "APPROVED" in output:
        st.markdown(f"<div class='judge-box' style='border-color: #10b981; background: rgba(16,185,129,0.05);'><div class='judge-title' style='color:#34d399;'>VERDICT: APPROVED ✓</div><span style='font-family:monospace; font-size:0.85rem;'>{output}</span></div>", unsafe_allow_html=True)
    elif "REJECTED" in output:
        st.markdown(f"<div class='judge-box' style='border-color: #ef4444; background: rgba(239,68,68,0.05);'><div class='judge-title' style='color:#fca5a5;'>VERDICT: REJECTED ✗</div><span style='font-family:monospace; font-size:0.85rem;'>{output}</span></div>", unsafe_allow_html=True)
    elif "PENALTY" in output or "VIOLATION" in output:
        st.markdown(f"<div class='judge-box' style='border-color: #ef4444; background: rgba(239,68,68,0.1);'><div class='judge-title' style='color:#fca5a5;'>SECURITY INTERVENTION ⚠</div><span style='font-family:monospace; font-size:0.85rem;'>{output}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='judge-box'><div class='judge-title'>AWAITING TICKET SUBMISSION</div><span style='font-family:monospace; color:#64748b; font-size:0.85rem;'>Agent has not yet submitted forensic evidence to the Change Board.</span></div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<h3>AGENT AUDIT TRAIL</h3>", unsafe_allow_html=True)
    audit_html = "<div class='terminal-feed' style='height: 480px; overflow-y: auto;'>"
    if history:
        for action in reversed(history):
            r_val = action["reward"]
            color = "#10b981" if r_val >= 0 else "#ef4444"
            audit_html += f"<span style='color:#3b82f6;'>[STEP {action['step']:02d}]</span> <strong style='color:#e2e8f0;'>{action['action'].upper()}</strong><br>"
            audit_html += f"<span style='color:#64748b;'>TARGET:</span> {action['target'][:60]}<br>"
            audit_html += f"<span style='color:#64748b;'>REWARD:</span> <span style='color:{color}; font-weight:bold;'>{r_val:+.1f}</span> :: {action['message'][:80]}<br><br>"
    else:
        audit_html += "> AWAITING AUTONOMOUS EXECUTION..."
    audit_html += "</div>"
    st.markdown(audit_html, unsafe_allow_html=True)

# ─── CONTROLS ────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin: 20px 0 10px 0;'>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1, 1, 2, 2])

with c1:
    auto_refresh = st.toggle("LIVE POLLING", value=False)
with c2:
    if st.button("↻ REFRESH"): st.rerun()
with c3:
    if st.button("⚠ INJECT ZERO-DAY INCIDENT", type="primary"):
        requests.post(f"{API_URL}/reset", json={"task_id": "auto"}, timeout=5)
        time.sleep(0.5)
        st.rerun()

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()