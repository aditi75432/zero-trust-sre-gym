import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import time
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Zero Trust SRE Gym", page_icon=None)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c10;
}

.main { background-color: #080c10; }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; }

.soc-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 4px;
}

.row-divider {
    border: none;
    border-top: 1px solid #0e1621;
    margin: 18px 0;
}

.panel {
    background: #0b1018;
    border: 1px solid #1a2332;
    border-radius: 4px;
    padding: 14px 16px;
    height: 100%;
}

.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    color: #2d4a6e;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 10px;
    border-bottom: 1px solid #0e1621;
    padding-bottom: 6px;
}

.led-fatal {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #e53e3e;
    box-shadow: 0 0 8px #e53e3e;
    margin-right: 8px;
    animation: pulse-red 1.2s infinite;
}

.led-warning {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #d69e2e;
    box-shadow: 0 0 6px #d69e2e;
    margin-right: 8px;
}

.led-clean {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #2f855a;
    margin-right: 8px;
}

.led-isolated {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #553c9a;
    box-shadow: 0 0 6px #553c9a;
    margin-right: 8px;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 5px #e53e3e; }
    50% { box-shadow: 0 0 16px #e53e3e, 0 0 28px rgba(229, 62, 62, 0.3); }
    100% { box-shadow: 0 0 5px #e53e3e; }
}

.led-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #0d1520;
}

.led-node-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #8da4c0;
    min-width: 110px;
}

.led-status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a6280;
}

.policy-armed {
    background: #0a1f0f;
    border: 1px solid #2f855a;
    border-radius: 4px;
    padding: 12px 14px;
    text-align: center;
}

.policy-blocked {
    background: #1a0a0a;
    border: 1px solid #c53030;
    border-radius: 4px;
    padding: 12px 14px;
    text-align: center;
    animation: pulse-border-red 1.5s infinite;
}

@keyframes pulse-border-red {
    0% { border-color: #c53030; }
    50% { border-color: #e53e3e; box-shadow: 0 0 12px rgba(197, 48, 48, 0.2); }
    100% { border-color: #c53030; }
}

.policy-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.policy-status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

.workflow-step {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a6280;
}

.step-done { color: #48bb78; }
.step-pending { color: #4a6280; }

.step-dot-done {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #48bb78;
    margin-right: 10px;
    flex-shrink: 0;
}

.step-dot-pending {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    border: 1px solid #2d4a6e;
    margin-right: 10px;
    flex-shrink: 0;
}

.step-connector {
    width: 1px;
    height: 10px;
    background: #1a2332;
    margin-left: 3px;
    margin-bottom: 0;
}

.cve-box {
    background: #060d16;
    border: 1px solid #1a2332;
    border-radius: 4px;
    padding: 10px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    line-height: 1.6;
    color: #3d7ab5;
    overflow: hidden;
    word-break: break-word;
}

.cve-id {
    color: #e53e3e;
    font-weight: 700;
    font-size: 11px;
    display: block;
    margin-bottom: 4px;
}

.judge-card {
    background: #060d16;
    border-radius: 4px;
    padding: 10px 12px;
}

.judge-persona-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 700;
}

.judge-persona-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    color: #4a6280;
    margin-top: 4px;
    line-height: 1.5;
}

.siem-terminal {
    background: #030609;
    border: 1px solid #112236;
    border-radius: 4px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    line-height: 1.7;
    color: #3a9e6e;
    min-height: 200px;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

.siem-cursor {
    display: inline-block;
    width: 7px;
    height: 12px;
    background: #3a9e6e;
    animation: blink 1s step-end infinite;
    vertical-align: text-bottom;
    margin-left: 2px;
}

@keyframes blink {
    50% { opacity: 0; }
}

.audit-row-positive {
    display: flex;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 5px 8px;
    border-bottom: 1px solid #0d1520;
    border-left: 2px solid #2f855a;
    margin-bottom: 2px;
}

.audit-row-negative {
    display: flex;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 5px 8px;
    border-bottom: 1px solid #0d1520;
    border-left: 2px solid #c53030;
    margin-bottom: 2px;
}

.audit-row-neutral {
    display: flex;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 5px 8px;
    border-bottom: 1px solid #0d1520;
    border-left: 2px solid #2d4a6e;
    margin-bottom: 2px;
}

.audit-step { color: #4a6280; min-width: 30px; }
.audit-tool { color: #8da4c0; min-width: 140px; }
.audit-target { color: #5a7a9e; min-width: 80px; }
.audit-reward-pos { color: #48bb78; margin-left: auto; }
.audit-reward-neg { color: #fc8181; margin-left: auto; }
.audit-reward-neu { color: #4a6280; margin-left: auto; }

.judge-vote-card {
    background: #060d16;
    border: 1px solid #1a2332;
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.judge-vote-approved {
    border-left: 3px solid #2f855a;
}

.judge-vote-rejected {
    border-left: 3px solid #c53030;
}

.judge-vote-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.judge-vote-ticket {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a6280;
}

.judge-vote-verdict-approved {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #48bb78;
    background: #0a1f0f;
    padding: 2px 8px;
    border-radius: 2px;
}

.judge-vote-verdict-rejected {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #fc8181;
    background: #1a0a0a;
    padding: 2px 8px;
    border-radius: 2px;
}

.judge-vote-justification {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    color: #5a7a9e;
    line-height: 1.5;
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px solid #0d1520;
}

.kpi-block {
    background: #0b1018;
    border: 1px solid #1a2332;
    border-radius: 4px;
    padding: 14px;
    text-align: center;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 10px;
    color: #4a6280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

.title-main {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #8da4c0;
    letter-spacing: 0.5px;
}

.title-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #2d4a6e;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

.stButton button {
    background: #0b1018;
    border: 1px solid #1a2332;
    color: #5a7a9e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-radius: 3px;
}

.stButton button:hover {
    border-color: #3d7ab5;
    color: #8da4c0;
}
</style>
""", unsafe_allow_html=True)

API_URL = "https://aditi75432-zero-trust-safe-SRE-gym.hf.space"
REFRESH_INTERVAL = 5

if "judge_vote_log" not in st.session_state:
    st.session_state.judge_vote_log = []
if "persona_stats" not in st.session_state:
    st.session_state.persona_stats = {"junior": {"approved": 0, "rejected": 0}, "senior": {"approved": 0, "rejected": 0}, "principal": {"approved": 0, "rejected": 0}}
if "last_ticket_action" not in st.session_state:
    st.session_state.last_ticket_action = ""
if "episode_results" not in st.session_state:
    st.session_state.episode_results = []


@st.cache_data(ttl=5)
def fetch(endpoint):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", timeout=3)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
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

    fixed_pos = {
        "api_gateway":  (-0.6, 0.8),
        "auth_service": (0.6, 0.8),
        "frontend":     (-0.6, 0.0),
        "payment":      (0.6, 0.0),
        "hr_db":        (0.0, -0.8),
    }
    pos = {n: fixed_pos.get(n, (0, 0)) for n in G.nodes()}

    status_colors = {
        "healthy":     "#2f855a",
        "compromised": "#c53030",
        "isolated":    "#553c9a",
        "offline":     "#2d3748",
    }
    protected = {"api_gateway", "auth_service"}

    edge_x, edge_y = [], []
    edge_colors = []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        src_status = nodes_state.get(e[0], {}).get("status", "healthy")
        color = "#c53030" if src_status == "compromised" else "#1a2d42"
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_colors.append(color)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#1a2d42"),
        hoverinfo="none"
    )

    node_x, node_y, colors, labels, hover_texts, sizes, symbols = [], [], [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        raw_status = nodes_state.get(n, {}).get("status", "unknown")
        colors.append(status_colors.get(raw_status, "#2d3748"))
        label = n.replace("_", " ").upper()
        labels.append(label)
        is_protected = n in protected
        hover_texts.append(
            f"<b>{n}</b><br>"
            f"Status: {raw_status}<br>"
            f"{'PROTECTED GATEWAY' if is_protected else 'INTERNAL SERVICE'}"
        )
        sizes.append(20 if is_protected else 16)
        symbols.append("diamond" if is_protected else "circle")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels,
        hovertext=hover_texts,
        hoverinfo="text",
        textposition="bottom center",
        textfont=dict(family="JetBrains Mono", size=8, color="#4a6280"),
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbols,
            line=dict(width=1.5, color="#0b1018"),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#080c10",
        paper_bgcolor="#080c10",
        font=dict(color="#8da4c0"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def draw_reward_curve(history):
    if not history:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor="#080c10",
            paper_bgcolor="#080c10",
            font=dict(color="#4a6280", family="JetBrains Mono"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        return fig

    df = pd.DataFrame(history)
    rewards = df["cumulative_reward"].values
    steps = df["step"].values

    window = max(3, len(rewards) // 8)
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        roll_steps = steps[window - 1:]
    else:
        rolling = rewards
        roll_steps = steps

    fig = go.Figure()

    fig.add_hrect(y0=-25, y1=0, fillcolor="rgba(197,48,48,0.04)", line_width=0)
    fig.add_hline(y=-20.0, line_dash="dot", line_color="#c53030", line_width=1,
                  annotation_text="Policy Violation Floor (-20)", annotation_font_size=9,
                  annotation_font_color="#c53030", annotation_position="bottom right")
    fig.add_hline(y=0, line_dash="dot", line_color="#2d4a6e", line_width=1)

    fig.add_trace(go.Scatter(
        x=steps, y=rewards,
        mode="lines",
        line=dict(color="#1a2d42", width=1),
        name="Raw reward",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=roll_steps, y=rolling,
        mode="lines",
        line=dict(color="#3d7ab5", width=2.5),
        name="Rolling mean",
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor="#080c10",
        paper_bgcolor="#080c10",
        font=dict(color="#4a6280", family="JetBrains Mono", size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="Step", gridcolor="#0e1621", showgrid=True),
        yaxis=dict(title="Cumulative Reward", gridcolor="#0e1621", showgrid=True),
    )
    return fig


def draw_persona_chart(persona_stats):
    personas = ["junior", "senior", "principal"]
    approved = [persona_stats[p]["approved"] for p in personas]
    rejected = [persona_stats[p]["rejected"] for p in personas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Approved", x=personas, y=approved,
        marker_color="#2f855a", marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="Rejected", x=personas, y=rejected,
        marker_color="#c53030", marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#080c10",
        paper_bgcolor="#080c10",
        font=dict(color="#4a6280", family="JetBrains Mono", size=9),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=10, t=10, b=30),
        xaxis=dict(gridcolor="#0e1621"),
        yaxis=dict(gridcolor="#0e1621", title="Count"),
    )
    return fig


def led_for_status(status):
    if status == "compromised":
        return "<span class='led-fatal'></span>"
    if status == "isolated":
        return "<span class='led-isolated'></span>"
    if status == "healthy":
        return "<span class='led-clean'></span>"
    return "<span class='led-warning'></span>"


def render_alert_leds(nodes_state, alerts):
    node_severity = {}
    for a in alerts:
        n = a.get("target_node", "")
        if a.get("severity") == "FATAL":
            node_severity[n] = "FATAL"
        elif n not in node_severity:
            node_severity[n] = "WARNING"

    html_parts = []
    for node, info in nodes_state.items():
        status = info.get("status", "unknown")
        sev = node_severity.get(node, "CLEAN")
        led = led_for_status(status)
        if sev == "FATAL":
            alert_badge = "<span style='font-family:JetBrains Mono;font-size:9px;color:#c53030;background:#1a0a0a;padding:1px 5px;border-radius:2px;margin-left:4px;'>FATAL</span>"
        elif sev == "WARNING":
            alert_badge = "<span style='font-family:JetBrains Mono;font-size:9px;color:#d69e2e;background:#1a1400;padding:1px 5px;border-radius:2px;margin-left:4px;'>WARN</span>"
        else:
            alert_badge = ""

        html_parts.append(
            f"<div class='led-row'>{led}<span class='led-node-name'>{node}</span>"
            f"<span class='led-status-text'>{status.upper()}</span>{alert_badge}</div>"
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_policy_status(command_output):
    is_blocked = "ACCESS DENIED" in command_output or "POLICY BLOCK" in command_output or "ROGUE" in command_output.upper()
    if is_blocked:
        last_rule = ""
        for line in command_output.split("\n"):
            if line.strip():
                last_rule = line.strip()[:60]
                break
        st.markdown(
            f"<div class='policy-blocked'>"
            f"<div class='policy-label' style='color:#c53030;'>Zero Trust Policy Engine</div>"
            f"<div class='policy-status-text' style='color:#fc8181;'>VIOLATION DETECTED</div>"
            f"<div style='font-family:JetBrains Mono;font-size:9px;color:#9b2c2c;margin-top:6px;'>{last_rule}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='policy-armed'>"
            "<div class='policy-label' style='color:#2f855a;'>Zero Trust Policy Engine</div>"
            "<div class='policy-status-text' style='color:#68d391;'>ARMED / COMPLIANT</div>"
            "</div>",
            unsafe_allow_html=True
        )


def render_ticket_workflow(active_ticket_id, ticket_approved, command_output):
    investigated = "SIEM ALERT" in command_output or "query_siem" in command_output.lower() or "STATUS: COMPROMISED" in command_output
    ticket_filed = active_ticket_id is not None
    approved = ticket_approved
    isolated = "ALL THREATS NEUTRALIZED" in command_output or "isolated" in command_output.lower()

    steps = [
        ("INVESTIGATE",   investigated),
        ("FILE TICKET",   ticket_filed),
        ("AUTHORIZED",    approved),
        ("ISOLATE",       isolated),
    ]

    parts = []
    for i, (label, done) in enumerate(steps):
        dot_class = "step-dot-done" if done else "step-dot-pending"
        text_class = "step-done" if done else "step-pending"
        parts.append(f"<div class='workflow-step {text_class}'><div class='{dot_class}'></div>{label}</div>")
        if i < len(steps) - 1:
            parts.append("<div class='step-connector'></div>")

    st.markdown("".join(parts), unsafe_allow_html=True)


def render_cve_feed(cve_context):
    if not cve_context or cve_context == "N/A":
        st.markdown("<div class='cve-box'><span style='color:#2d4a6e;'>Awaiting adversarial designer...</span></div>", unsafe_allow_html=True)
        return
    cve_id = ""
    summary = cve_context
    if "CVE-" in cve_context:
        parts = cve_context.split("CVE-", 1)
        if len(parts) > 1:
            rest = "CVE-" + parts[1]
            tokens = rest.split(" ", 1)
            cve_id = tokens[0]
            summary = tokens[1] if len(tokens) > 1 else ""
    st.markdown(
        f"<div class='cve-box'>"
        f"<span class='cve-id'>{cve_id if cve_id else 'LIVE THREAT INTEL'}</span>"
        f"{summary[:220]}"
        f"</div>",
        unsafe_allow_html=True
    )


def render_judge_assignment(judge_persona):
    persona_descriptions = {
        "junior":    ("JUNIOR SRE", "#d69e2e", "Lenient evaluation. Rewards investigative effort. Approves on partial evidence.", "#1a1400"),
        "senior":    ("SENIOR SRE", "#3d7ab5", "Standard enterprise threshold. Requires specific IPs and IAM role names.", "#0a1520"),
        "principal": ("PRINCIPAL CISO", "#c53030", "Zero tolerance. Demands exact anomaly metrics, timestamps, and IP attribution.", "#1a0a0a"),
    }
    name, color, desc, bg = persona_descriptions.get(judge_persona.lower(), ("UNKNOWN", "#4a6280", "", "#080c10"))
    st.markdown(
        f"<div class='judge-card' style='background:{bg};border:1px solid {color}20;'>"
        f"<div class='judge-persona-name' style='color:{color};'>{name}</div>"
        f"<div class='judge-persona-desc'>{desc}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_siem_terminal(command_output):
    display = command_output if command_output else "Awaiting agent action..."
    safe = display.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"<div class='siem-terminal'>{safe}<span class='siem-cursor'></span></div>",
        unsafe_allow_html=True
    )


def render_audit_trail(history):
    if not history:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;color:#2d4a6e;padding:8px;'>No actions recorded.</div>", unsafe_allow_html=True)
        return
    rows_html = []
    for h in reversed(history[-25:]):
        reward = h.get("reward", 0)
        if reward > 0:
            row_class = "audit-row-positive"
            reward_class = "audit-reward-pos"
            prefix = "+"
        elif reward < -5:
            row_class = "audit-row-negative"
            reward_class = "audit-reward-neg"
            prefix = ""
        else:
            row_class = "audit-row-neutral"
            reward_class = "audit-reward-neu"
            prefix = ""
        tool = h.get("action", "")[:20]
        target = str(h.get("target", ""))[:12]
        rows_html.append(
            f"<div class='{row_class}'>"
            f"<span class='audit-step'>{h.get('step', '')}</span>"
            f"<span class='audit-tool'>{tool}</span>"
            f"<span class='audit-target'>{target}</span>"
            f"<span class='{reward_class}'>{prefix}{reward:.2f}</span>"
            f"</div>"
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)


def update_judge_vote_log(history, command_output, judge_persona, active_ticket_id):
    if not history:
        return
    last = history[-1] if history else {}
    if last.get("action") == "file_ticket":
        ticket_key = f"{active_ticket_id or 'pending'}_{last.get('step', 0)}"
        existing_keys = [v.get("key") for v in st.session_state.judge_vote_log]
        if ticket_key not in existing_keys:
            approved = "TICKET" in command_output and "APPROVED" in command_output
            rejected = "REJECTED" in command_output
            if approved or rejected:
                verdict = "APPROVED" if approved else "REJECTED"
                justification_snippet = last.get("target", "")[:80]
                st.session_state.judge_vote_log.insert(0, {
                    "key":         ticket_key,
                    "persona":     judge_persona,
                    "ticket_id":   active_ticket_id or "PENDING",
                    "verdict":     verdict,
                    "feedback":    command_output[:300],
                    "step":        last.get("step", 0),
                    "justification": justification_snippet,
                })
                st.session_state.judge_vote_log = st.session_state.judge_vote_log[:20]
                if verdict == "APPROVED":
                    st.session_state.persona_stats[judge_persona]["approved"] += 1
                else:
                    st.session_state.persona_stats[judge_persona]["rejected"] += 1


def render_judge_vote_log():
    if not st.session_state.judge_vote_log:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;color:#2d4a6e;padding:8px;'>No ticket evaluations yet.</div>", unsafe_allow_html=True)
        return
    for vote in st.session_state.judge_vote_log[:6]:
        is_approved = vote["verdict"] == "APPROVED"
        card_class = "judge-vote-approved" if is_approved else "judge-vote-rejected"
        verdict_class = "judge-vote-verdict-approved" if is_approved else "judge-vote-verdict-rejected"
        persona_colors = {"junior": "#d69e2e", "senior": "#3d7ab5", "principal": "#c53030"}
        p_color = persona_colors.get(vote["persona"], "#4a6280")
        feedback_safe = vote["feedback"].replace("<", "&lt;").replace(">", "&gt;")[:200]
        st.markdown(
            f"<div class='judge-vote-card {card_class}'>"
            f"<div class='judge-vote-header'>"
            f"<span class='judge-vote-ticket'>"
            f"Step {vote['step']} | "
            f"<span style='color:{p_color};'>{vote['persona'].upper()}</span> | "
            f"{vote['ticket_id']}"
            f"</span>"
            f"<span class='{verdict_class}'>{vote['verdict']}</span>"
            f"</div>"
            f"<div class='judge-vote-justification'>{feedback_safe}</div>"
            f"</div>",
            unsafe_allow_html=True
        )


def compute_kpis(history):
    if not history:
        return {"success_rate": 0.0, "violation_rate": 0.0, "avg_steps": 0.0, "valid_json": 0.0, "avg_reward": 0.0}

    violation_actions = ["isolate_node"]
    violations = sum(1 for h in history if h.get("reward", 0) <= -15)
    total = len(history)
    positive = sum(1 for h in history if h.get("reward", 0) > 5)

    return {
        "success_rate":  round(positive / total * 100, 1) if total else 0.0,
        "violation_rate": round(violations / total * 100, 1) if total else 0.0,
        "avg_steps":     round(total / max(1, len(st.session_state.episode_results) or 1), 1),
        "avg_reward":    round(sum(h.get("reward", 0) for h in history) / total, 2) if total else 0.0,
    }


state, ok1 = fetch("state")
history_data, ok2 = fetch("history")
services, ok3 = fetch("services")
history = history_data.get("history", []) if ok2 else []

command_output = state.get("command_output", "")
active_ticket_id = state.get("active_ticket_id")
ticket_approved = state.get("ticket_approved", False)
judge_persona = state.get("judge_persona", "senior")
nodes_state = state.get("nodes", {})
alerts = state.get("active_alerts", [])
curriculum = state.get("curriculum", {})
cve_context = state.get("cve_context", "")
global_uptime = state.get("global_uptime", 100.0)
episode_reward = state.get("episode_reward", 0.0)
difficulty = curriculum.get("difficulty", "warmup").upper()

if history:
    update_judge_vote_log(history, command_output, judge_persona, active_ticket_id)

st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:16px;'>"
    f"<div>"
    f"<div class='title-main'>Zero Trust SRE Gym</div>"
    f"<div class='title-sub'>Enterprise Security Operations Centre / Autonomous Containment Agent</div>"
    f"</div>"
    f"<div style='font-family:JetBrains Mono;font-size:10px;color:#2d4a6e;'>"
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

if not ok1:
    st.error("Environment server unreachable.")
    st.stop()

st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)
st.markdown("<div class='soc-header'>Situation Room</div>", unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([2, 1.6, 1.6, 2.2, 1.6])

with r1c1:
    st.markdown("<div class='panel-label'>Node Status LEDs</div>", unsafe_allow_html=True)
    render_alert_leds(nodes_state, alerts)

with r1c2:
    st.markdown("<div class='panel-label'>Policy Engine</div>", unsafe_allow_html=True)
    render_policy_status(command_output)

with r1c3:
    st.markdown("<div class='panel-label'>ITIL Workflow</div>", unsafe_allow_html=True)
    render_ticket_workflow(active_ticket_id, ticket_approved, command_output)

with r1c4:
    st.markdown("<div class='panel-label'>Live CVE Threat Intel</div>", unsafe_allow_html=True)
    render_cve_feed(cve_context)

with r1c5:
    st.markdown("<div class='panel-label'>Assigned Judge</div>", unsafe_allow_html=True)
    render_judge_assignment(judge_persona)

st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)
st.markdown("<div class='soc-header'>Living Environment</div>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns([1.6, 1.8, 1.6])

with r2c1:
    st.markdown("<div class='panel-label'>Service Topology</div>", unsafe_allow_html=True)
    st.plotly_chart(draw_topology(nodes_state), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='panel-label' style='margin-top:8px;'>Microservice Health</div>", unsafe_allow_html=True)
    if ok3:
        for name, info in services.items():
            status = info.get("status", "unknown")
            latency = info.get("latency_ms", "N/A")
            status_colors = {"healthy": "#2f855a", "degraded": "#d69e2e", "compromised": "#c53030", "offline": "#2d3748"}
            sc = status_colors.get(status, "#2d3748")
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:4px 0;border-bottom:1px solid #0d1520;'>"
                f"<span style='width:7px;height:7px;border-radius:50%;background:{sc};margin-right:8px;flex-shrink:0;'></span>"
                f"<span style='font-family:JetBrains Mono;font-size:10px;color:#8da4c0;min-width:80px;'>{name}</span>"
                f"<span style='font-family:JetBrains Mono;font-size:10px;color:#4a6280;margin-left:auto;'>{latency}ms</span>"
                f"</div>",
                unsafe_allow_html=True
            )

with r2c2:
    st.markdown("<div class='panel-label'>SIEM Terminal Output</div>", unsafe_allow_html=True)
    render_siem_terminal(command_output)

    st.markdown("<div class='panel-label' style='margin-top:12px;'>Active Alerts</div>", unsafe_allow_html=True)
    if alerts:
        for a in alerts:
            sev = a.get("severity", "")
            sev_colors = {"FATAL": "#c53030", "WARNING": "#d69e2e"}
            sc = sev_colors.get(sev, "#4a6280")
            st.markdown(
                f"<div style='border-left:2px solid {sc};padding:5px 10px;margin-bottom:4px;background:#060d16;border-radius:0 4px 4px 0;'>"
                f"<div style='font-family:JetBrains Mono;font-size:9px;font-weight:700;color:{sc};'>{sev} — {a.get('target_node','')}</div>"
                f"<div style='font-family:IBM Plex Sans,sans-serif;font-size:10px;color:#4a6280;margin-top:2px;'>{a.get('symptom','')[:90]}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;color:#2d4a6e;padding:8px;'>No active alerts.</div>", unsafe_allow_html=True)

with r2c3:
    st.markdown("<div class='panel-label'>Agent Audit Trail</div>", unsafe_allow_html=True)
    render_audit_trail(history)

    kpis = compute_kpis(history)
    st.markdown("<div class='panel-label' style='margin-top:12px;'>Episode Metrics</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        color = "#48bb78" if global_uptime >= 80 else "#fc8181"
        st.markdown(f"<div class='kpi-block'><div class='kpi-value' style='color:{color};'>{global_uptime:.0f}%</div><div class='kpi-label'>Uptime</div></div>", unsafe_allow_html=True)
    with m2:
        color = "#48bb78" if episode_reward >= 0 else "#fc8181"
        prefix = "+" if episode_reward >= 0 else ""
        st.markdown(f"<div class='kpi-block'><div class='kpi-value' style='color:{color};'>{prefix}{episode_reward:.1f}</div><div class='kpi-label'>Ep Reward</div></div>", unsafe_allow_html=True)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"<div class='kpi-block'><div class='kpi-value' style='color:#3d7ab5;'>{difficulty}</div><div class='kpi-label'>Difficulty</div></div>", unsafe_allow_html=True)
    with m4:
        tid = active_ticket_id or "—"
        color = "#48bb78" if ticket_approved else "#8da4c0"
        st.markdown(f"<div class='kpi-block'><div class='kpi-value' style='color:{color};font-size:14px;margin-top:4px;'>{tid}</div><div class='kpi-label'>Active Ticket</div></div>", unsafe_allow_html=True)

st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)
st.markdown("<div class='soc-header'>Compliance Judge Feed</div>", unsafe_allow_html=True)

r3c1, r3c2 = st.columns([1.8, 1.2])

with r3c1:
    st.markdown("<div class='panel-label'>Ticket Evaluation Record</div>", unsafe_allow_html=True)
    render_judge_vote_log()

with r3c2:
    st.markdown("<div class='panel-label'>Approval Rate by Persona</div>", unsafe_allow_html=True)
    st.plotly_chart(draw_persona_chart(st.session_state.persona_stats), use_container_width=True, config={"displayModeBar": False})

    mastery = curriculum.get("mastery", {})
    if mastery:
        st.markdown("<div class='panel-label' style='margin-top:8px;'>Threat Mastery</div>", unsafe_allow_html=True)
        for threat, score in mastery.items():
            bar_width = int(score * 100)
            color = "#48bb78" if score > 0.6 else "#d69e2e" if score > 0.3 else "#c53030"
            st.markdown(
                f"<div style='margin-bottom:6px;'>"
                f"<div style='display:flex;justify-content:space-between;font-family:JetBrains Mono;font-size:9px;color:#4a6280;margin-bottom:3px;'>"
                f"<span>{threat.replace('_',' ').upper()}</span><span style='color:{color};'>{score:.0%}</span></div>"
                f"<div style='height:3px;background:#0e1621;border-radius:2px;'>"
                f"<div style='height:3px;width:{bar_width}%;background:{color};border-radius:2px;transition:width 0.4s;'></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)
st.markdown("<div class='soc-header'>Training Evidence</div>", unsafe_allow_html=True)

r4c1, r4c2 = st.columns([2, 1])

with r4c1:
    st.markdown("<div class='panel-label'>Reward Trajectory (Rolling Mean + Policy Floor)</div>", unsafe_allow_html=True)
    st.plotly_chart(draw_reward_curve(history), use_container_width=True, config={"displayModeBar": False})

with r4c2:
    st.markdown("<div class='panel-label'>Session Performance</div>", unsafe_allow_html=True)

    total_actions = len(history)
    violations = sum(1 for h in history if h.get("reward", 0) <= -15)
    positive_actions = sum(1 for h in history if h.get("reward", 0) > 0)
    ep_count = curriculum.get("episode_count", 0)

    perf_metrics = [
        ("Total Actions",   str(total_actions),                        "#8da4c0"),
        ("Policy Violations", str(violations),                         "#fc8181" if violations > 0 else "#48bb78"),
        ("Positive Rewards",  str(positive_actions),                   "#48bb78"),
        ("Episodes Done",   str(ep_count),                             "#3d7ab5"),
        ("Resolution Rate", f"{curriculum.get('resolution_rate', 0):.0%}", "#48bb78"),
        ("Avg Mastery",     f"{curriculum.get('avg_mastery', 0):.0%}", "#d69e2e"),
    ]

    for label, value, color in perf_metrics:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:7px 10px;border-bottom:1px solid #0d1520;"
            f"font-family:JetBrains Mono;font-size:11px;'>"
            f"<span style='color:#4a6280;'>{label}</span>"
            f"<span style='color:{color};font-weight:600;'>{value}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='panel-label' style='margin-top:14px;'>Controls</div>", unsafe_allow_html=True)
    if st.button("Reset Environment"):
        try:
            requests.post(f"{API_URL}/reset", json={"task_id": "auto"}, timeout=5)
            st.session_state.judge_vote_log = []
            st.cache_data.clear()
        except Exception:
            pass

    auto = st.toggle("Auto Refresh (5s)", value=True)

if auto:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()