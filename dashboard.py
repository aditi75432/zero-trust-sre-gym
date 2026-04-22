import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import time

# ─── CONFIG ──────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Zero Trust SOC Telemetry",
    initial_sidebar_state="collapsed"
)



# Update this to your HuggingFace Space URL when deployed
API_URL = "https://aditi75432-zero-trust-safe-SRE-gym.hf.space"
REFRESH_INTERVAL = 2

# ─── ENTERPRISE CSS (MUST COME AFTER CONFIG) ────────────────────────

st.markdown("""
<style>
    /* Strict Dark Mode Palette */
    .stApp { background-color: #0b0f19; color: #cbd5e1; font-family: 'Inter', sans-serif; }
    
    /* HUD Metric Cards */
    .stMetric { 
        background: #111827; 
        padding: 15px; 
        border-radius: 4px; 
        border: 1px solid #1e293b; 
        border-top: 3px solid #3b82f6;
    }
    .stMetric label { color: #64748b !important; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; font-size: 0.75rem;}
    
    /* Terminal/Monospace text for logs */
    .terminal-feed {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.85rem;
        background: #0f172a;
        padding: 10px;
        border: 1px solid #1e293b;
        border-radius: 4px;
        color: #a3e635;
    }
    
    /* Strict Alert Banners */
    .alert-fatal { border-left: 4px solid #ef4444; padding: 10px; background: #171717; margin: 4px 0; font-family: monospace; color: #ef4444; font-size: 0.85rem;}
    .alert-warn { border-left: 4px solid #f59e0b; padding: 10px; background: #171717; margin: 4px 0; font-family: monospace; color: #f59e0b; font-size: 0.85rem;}
    .alert-info { border-left: 4px solid #3b82f6; padding: 10px; background: #171717; margin: 4px 0; font-family: monospace; color: #3b82f6; font-size: 0.85rem;}
    
    /* Headers */
    h1, h2, h3 { color: #f8fafc; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; font-size: 1.1rem !important; }
    hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def fetch_state() -> dict:
    try:
        r = requests.get(f"{API_URL}/state", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def fetch_history() -> dict:
    try:
        r = requests.get(f"{API_URL}/history", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def draw_service_mesh(nodes_state: dict) -> go.Figure:
    """Renders the DAG with strict enterprise geometric styling."""
    G = nx.DiGraph()
    edges = [
        ("api_gateway", "frontend"), ("api_gateway", "payment"),
        ("auth_service", "frontend"), ("auth_service", "payment"),
        ("frontend", "hr_db")
    ]
    G.add_edges_from(edges)
    
    pos = {
        "api_gateway":  (0.0, 2.0), "auth_service": (0.0, 0.0),
        "frontend":     (2.0, 1.5), "payment":      (2.0, 0.5),
        "hr_db":        (4.0, 1.5)
    }
    
    STATUS_COLORS = {
        "healthy": "#10b981", "compromised": "#ef4444",
        "isolated": "#f59e0b", "offline": "#334155"
    }
    
    STATUS_TAGS = {
        "healthy": "[OK]", "compromised": "[BREACH]", 
        "isolated": "[ISOLATED]", "offline": "[OFFLINE]"
    }
    
    node_colors, hover_texts, node_texts = [], [], []
    for node in G.nodes():
        node_data = nodes_state.get(node, {})
        status = node_data.get("status", "offline")
        node_colors.append(STATUS_COLORS.get(status, "#334155"))
        hover_texts.append(f"NODE: {node.upper()}<br>STATE: {status.upper()}")
        node_texts.append(f"{STATUS_TAGS.get(status, '[??]')} {node.upper()}")
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1.5, color="#334155"),
        hoverinfo="none", mode="lines"
    )
    
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers+text", text=node_texts, hovertext=hover_texts,
        hoverinfo="text", textposition="bottom center",
        textfont=dict(size=10, color="#94a3b8", family="monospace"),
        marker=dict(size=35, color=node_colors, symbol="square", line=dict(width=2, color="#0b0f19"))
    )
    
    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            showlegend=False, hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5.0]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5])
        )
    )

def draw_reward_curve(history: list) -> go.Figure | None:
    """Plots RL optimization trajectory."""
    if not history: return None
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="solid", line_color="#334155", line_width=1)
    
    colors = ["#10b981" if r >= 0 else "#ef4444" for r in df.get("reward", [])]
    
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["cumulative_reward"],
        mode="lines+markers", line=dict(color="#3b82f6", width=1.5, shape="vh"),
        marker=dict(size=6, color=colors, symbol="square"),
        hovertemplate="SEQ: %{x}<br>NET_YIELD: %{y:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor="#0b0f19", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", size=10, family="monospace"),
        xaxis=dict(title="SEQUENCE STEP", gridcolor="#1e293b"),
        yaxis=dict(title="NET REWARD YIELD", gridcolor="#1e293b"),
        margin=dict(b=30, l=40, r=10, t=10), height=200
    )
    return fig


# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────

st.markdown("<h2 style='color:#f8fafc; margin-bottom: 0;'>ZERO TRUST SOC TELEMETRY</h2>", unsafe_allow_html=True)
st.markdown("<div style='color:#64748b; font-size:0.85rem; font-family:monospace; margin-bottom: 20px;'>STATUS: SECURE CONNECTION ESTABLISHED // LIVE FEED</div>", unsafe_allow_html=True)

state = fetch_state()
history_data = fetch_history()

if not state:
    st.error("[SYS_ERR] TELEMETRY SERVER UNREACHABLE ON PORT 7860.")

curriculum = state.get("curriculum", {})
history = history_data.get("history", [])

# ── KPI Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

uptime = state.get("global_uptime", 100.0)
uptime_delta = f"{uptime - 100:.0f}%" if uptime < 100 else "NOMINAL"

col1.metric("GLOBAL UPTIME", f"{uptime:.1f}%", delta=uptime_delta)
col2.metric("ACTIVE TICKET", state.get("active_ticket_id") or "NONE_ACTIVE")
col3.metric("AUTH STATUS", "GRANTED" if state.get("ticket_approved") else "LOCKED")
col4.metric("THREAT LEVEL", curriculum.get("difficulty", "WARMUP").upper())
col5.metric("EPOCH SEQ", curriculum.get("episode_count", 0))
col6.metric("NET YIELD", f"{state.get('episode_reward', 0):.1f}")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main layout ──────────────────────────────────────────────────
left, middle, right = st.columns([1.2, 1.0, 1.0])

# ── LEFT: Topology & SIEM ──
with left:
    st.markdown("<h3>SERVICE MESH TOPOLOGY</h3>", unsafe_allow_html=True)
    nodes = state.get("nodes", {})
    if nodes:
        st.plotly_chart(draw_service_mesh(nodes), use_container_width=True)
    else:
        st.info("[AWAITING TOPOLOGY DATA]")
    
    st.markdown("<h3>SIEM ALERT FEED</h3>", unsafe_allow_html=True)
    alerts = state.get("active_alerts", [])
    if alerts:
        for alert in alerts:
            sev = alert.get("severity", "INFO")
            css_class = "alert-fatal" if sev in ["FATAL", "CRITICAL"] else "alert-warn"
            st.markdown(f"<div class='{css_class}'>[{sev}] {alert['target_node'].upper()} :: {alert['symptom']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-info'>[INFO] NO ACTIVE ALERTS. SYSTEM NOMINAL.</div>", unsafe_allow_html=True)

# ── MIDDLE: Metrics & Curriculum ──
with middle:
    st.markdown("<h3>OPTIMIZATION TRAJECTORY</h3>", unsafe_allow_html=True)
    if history:
        st.plotly_chart(draw_reward_curve(history), use_container_width=True)
    else:
        st.markdown("<div class='terminal-feed'>AWAITING SEQUENCE DATA...</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>POLICY ADHERENCE MASTERY</h3>", unsafe_allow_html=True)
    
    mastery = curriculum.get("mastery", {})
    for threat_type, score in mastery.items():
        label = threat_type.replace("_", " ").upper()
        st.markdown(f"<div style='font-family:monospace; font-size:0.8rem; color:#94a3b8; margin-bottom:2px;'>{label}</div>", unsafe_allow_html=True)
        st.progress(float(score))

# ── RIGHT: Audit Trail ──
with right:
    st.markdown("<h3>AGENT AUDIT TRAIL</h3>", unsafe_allow_html=True)
    
    audit_html = "<div class='terminal-feed' style='height: 380px; overflow-y: auto;'>"
    if history:
        for action in reversed(history[-15:]):
            r_val = action["reward"]
            r_str = f"+{r_val:.1f}" if r_val >= 0 else f"{r_val:.1f}"
            color = "#10b981" if r_val >= 0 else "#ef4444"
            
            audit_html += f"<span style='color:#3b82f6;'>[SEQ {action['step']:02d}]</span> EXEC: {action['action'].upper()}<br>"
            audit_html += f"<span style='color:#64748b;'>TARGET:</span> {action['target'][:50]}<br>"
            audit_html += f"<span style='color:#64748b;'>EVAL:</span> <span style='color:{color};'>{r_str}</span> :: {action['message'][:60]}<br><br>"
    else:
        audit_html += "AWAITING TELEMETRY..."
    audit_html += "</div>"
    
    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>LAST TERMINAL OUTPUT</h3>", unsafe_allow_html=True)
    output = state.get("command_output", "WAITING FOR COMMAND...")
    
    # Clean terminal output formatting
    out_color = "#cbd5e1"
    if "APPROVED" in output or "SUCCESS" in output: out_color = "#10b981"
    elif "REJECTED" in output or "DENIED" in output or "FATAL" in output: out_color = "#ef4444"
    
    st.markdown(f"<div class='terminal-feed' style='color:{out_color}; height:120px; overflow-y:auto;'>{output}</div>", unsafe_allow_html=True)

# ─── CONTROLS ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("MANUAL REFRESH"): st.rerun()
with c2:
    auto_refresh = st.checkbox("ENABLE AUTO-POLLING", value=False)
with c3:
    if st.button("INJECT NEW INCIDENT (RESET)"):
        try:
            requests.post(f"{API_URL}/reset", json={"task_id": "auto"}, timeout=5)
            time.sleep(0.5)
            st.rerun()
        except: pass
with c4:
    if st.button("PURGE CURRICULUM STATE"):
        try:
            requests.post(f"{API_URL}/curriculum/reset", timeout=5)
            time.sleep(0.5)
            st.rerun()
        except: pass

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()