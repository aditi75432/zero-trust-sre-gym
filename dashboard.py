import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

st.set_page_config(layout="wide")

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

    edge_x = []
    edge_y = []

    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1)
    )

    node_x = []
    node_y = []
    colors = []
    texts = []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        state = nodes_state.get(n, {}).get("status", "unknown")

        if state == "healthy":
            colors.append("green")
        elif state == "compromised":
            colors.append("red")
        elif state == "isolated":
            colors.append("orange")
        else:
            colors.append("gray")

        texts.append(f"{n}<br>{state}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        hovertext=texts,
        textposition="bottom center",
        marker=dict(size=20, color=colors)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)

    return fig


def draw_rewards(history):
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["cumulative_reward"],
        mode="lines+markers"
    ))

    return fig


def render_services(services):
    for name, info in services.items():
        st.write(f"{name} | {info.get('status')} | latency={info.get('latency_ms')}")


def render_alerts(alerts):
    for a in alerts:
        st.write(f"{a['severity']} | {a['target_node']} | {a['symptom']}")


def render_logs(state):
    st.write(state.get("command_output", ""))


def render_audit(history):
    for h in reversed(history[-20:]):
        st.write(f"{h['step']} | {h['action']} | {h['target']} | {h['reward']}")


state, ok1 = fetch("state")
history_data, ok2 = fetch("history")
services, ok3 = fetch("services")

history = history_data.get("history", []) if ok2 else []

st.title("Zero Trust RL Environment")

if not ok1:
    st.write("Environment not reachable")
    st.stop()

st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Uptime", f"{state.get('global_uptime', 100):.1f}")
c2.metric("Reward", f"{state.get('episode_reward', 0):.2f}")
c3.metric("Difficulty", state.get("curriculum", {}).get("difficulty", ""))
c4.metric("Ticket", state.get("active_ticket_id") or "None")

st.subheader("Service Topology")
st.plotly_chart(draw_topology(state.get("nodes", {})), use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Microservices")
    if ok3:
        render_services(services)

with col2:
    st.subheader("Active Alerts")
    render_alerts(state.get("active_alerts", []))

with col3:
    st.subheader("Latest SIEM Output")
    render_logs(state)

st.subheader("Reward Trajectory")
st.plotly_chart(draw_rewards(history), use_container_width=True)

st.subheader("Agent Actions")
render_audit(history)

colA, colB = st.columns(2)

with colA:
    if st.button("Reset Environment"):
        requests.post(f"{API_URL}/reset", json={"task_id": "auto"})

with colB:
    auto = st.toggle("Auto Refresh", value=True)

if auto:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()