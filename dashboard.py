import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Zero Trust SRE Twin")
API_URL = "https://aditi75432-zero-trust-sre-env.hf.space" 

# --- TOPOLOGY VISUALIZER (The DAG) ---
def draw_service_mesh(nodes_state):
    G = nx.DiGraph()
    edges = [
        ("api_gateway", "frontend"), ("api_gateway", "payment"),
        ("auth_service", "frontend"), ("auth_service", "payment"),
        ("frontend", "hr_db")
    ]
    G.add_edges_from(edges)
    
    # Static positions for a clean enterprise look
    pos = {
        "api_gateway": (0, 2), "auth_service": (0, 0),
        "frontend": (1, 1), "payment": (1, -1),
        "hr_db": (2, 1)
    }
    
    node_colors = []
    for node in G.nodes():
        status = nodes_state.get(node, {}).get("status", "offline")
        if status == "healthy": node_colors.append("green")
        elif status == "compromised": node_colors.append("red")
        elif status == "isolated": node_colors.append("orange")
        else: node_colors.append("grey") # Offline / Cascaded

    # Plotly magic for interactive network graphs
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=list(G.nodes()), textposition="bottom center",
        marker=dict(size=40, color=node_colors, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Live Microservice Topology',
                showlegend=False, hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
             )
    return fig

# --- DASHBOARD LAYOUT ---
st.title("🛡️ Autonomous Zero Trust SRE Console")

# 1. Top KPI Metrics
try:
    state_res = requests.get(f"{API_URL}/state", timeout=2)
    state = state_res.json()
except:
    st.error("Cannot connect to Digital Twin Environment.")
    state = {"global_uptime": 0.0, "nodes": {}}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Global Production Uptime", value=f"{state.get('global_uptime', 100)}%", delta="-20%" if state.get('global_uptime', 100) < 100 else "Stable")
with col2:
    st.metric(label="Active Jira Tickets", value=state.get('active_ticket_id', 'None'))
with col3:
    st.metric(label="Change Board Status", value="APPROVED" if state.get('ticket_approved') else "PENDING / LOCKED")

st.markdown("---")

# 2. Main Split Screen
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("Service Mesh Telemetry")
    # Draw the interactive DAG
    if "nodes" in state:
        st.plotly_chart(draw_service_mesh(state["nodes"]), use_container_width=True)
    else:
        st.info("Awaiting topology data...")

with right_col:
    st.subheader("Live Agent Audit Trail")
    # This simulates a feed of the agent's actions and the LLM Judge's responses
    # In reality, you would pull a 'history' array from your FastAPI backend
    st.container(height=400, border=True)
    st.markdown("**Action 1:** `query_siem_logs` (hr_db)")
    st.markdown("*> Result: Exfiltration detected from 10.0.5.42*")
    st.markdown("---")
    st.markdown("**Action 2:** `file_ticket`")
    st.markdown("*> Agent Justification: Found bad IP. Need to block.*")
    st.error("⚠️ JUDGE REJECTED: Insufficient forensic evidence provided. Request denied.")
    
st.button("Force Environment Sync", on_click=lambda: st.rerun())