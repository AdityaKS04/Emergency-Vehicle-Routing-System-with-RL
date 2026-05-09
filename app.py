import streamlit as st
import numpy as np

from simulation.graph_builder import build_synthetic_city_graph
from simulation.routing_engine import a_star_route
from simulation.incidents import detect_incident, apply_incident_to_graph
from ml.clearance_model import predict_clearance
from utils.visualizer import create_base_figure
from simulation.ev_animation import animate_ev
from rl.signal_agent import choose_signal


# --------------------------------------------------------------
# Compute graph coordinate distance for route
# --------------------------------------------------------------
def compute_route_units(route, positions):
    total = 0
    for i in range(len(route) - 1):
        p1 = np.array(positions[route[i]])
        p2 = np.array(positions[route[i + 1]])
        total += np.linalg.norm(p2 - p1)
    return total


# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.set_page_config(page_title="EV Traffic AI Simulation", layout="wide")
st.title("🚑 EV Traffic AI Simulation (ML + Dynamic Routing + RL)")

st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Select Clearance Prediction Model",
    ["XGBoost", "Tuned XGBoost", "Random Forest"]
)

queue_length = st.sidebar.slider("Queue Length (m)", 10, 500, 120)
speed = st.sidebar.slider("Speed (kmph)", 1, 60, 20)

simulate_btn = st.sidebar.button("Run Simulation")

# Build Graph (undirected inside builder)
G, positions = build_synthetic_city_graph()
start, goal = "Hospital", "Patient"


# --------------------------------------------------------------
# MAIN SIMULATION
# --------------------------------------------------------------
if simulate_btn:

    # ---------------- INCIDENT DETECTION ----------------
    st.write("### 🔍 Detecting incidents...")
    incident_detected, reason = detect_incident(queue_length, speed)
    st.write(f"Incident status: **{reason}**")

    incident_edge = apply_incident_to_graph(G, incident_detected)
    st.write("Incident edge:", incident_edge)

    blocked_nodes = set()
    if incident_edge:
        blocked_nodes.update(incident_edge)

    # --------------------------------------------------------------
    # FORCE REROUTE LOGIC:
    # If D or E is blocked → block entire lower grid (D, E, G, H)
    # --------------------------------------------------------------
    if "D" in blocked_nodes or "E" in blocked_nodes:
        blocked_nodes.update(["D", "E", "G", "H"])

    st.write("Blocked nodes:", blocked_nodes)

    # ---------------- PREDICT CLEARANCE ----------------
    st.write("### 🧠 Predicting clearance distance...")
    sample_dict = {
        "Queue_Length_m": queue_length,
        "Estimated_Flow_Speed_kmph": speed,
        "CarCount": 140,
        "BikeCount": 90,
        "BusCount": 4,
        "TruckCount": 10,
        "Weighted_Traffic_Index": 540.0,
        "EV_Distance_From_Signal_km": 1.5,
        "Cycle_Position_sec": 30,
    }

    clearance = predict_clearance(sample_dict, model_choice)
    st.write(f"Predicted Clearance: **{clearance:.3f} km**")

    # ---------------- APPLY WEIGHTS + BLOCKS ----------------
    for u, v in G.edges():
        if u in blocked_nodes or v in blocked_nodes:
            G[u][v]["dynamic_weight"] = 1e12
        else:
            G[u][v]["dynamic_weight"] = G[u][v]["distance"] + clearance * 10

    # ---------------- RUN ROUTING ----------------
    st.write("### 🧭 Finding best EV route...")
    route, cost = a_star_route(G, start, goal)

    st.write("Route:", route)
    st.write("Cost:", cost)

    if not route:
        st.error("No valid route found!")
        st.stop()

    st.success(f"Route: {' → '.join(route)}")

    # ----------------------------------------------------------
    # RL-BASED SIGNAL SELECTION
    # ----------------------------------------------------------
    chosen_signal, rl_debug = choose_signal(
        queue_length=queue_length,
        speed=speed,
        clearance=clearance,
        route=route,
        return_debug=True
    )

    # Safety fallback if RL chooses a signal not on the route
    if chosen_signal not in route:
        if "E" in route:
            chosen_signal = "E"
        elif "B" in route:
            chosen_signal = "B"
        else:
            st.error("No usable signal found on the route.")
            st.stop()

    st.success(f"Using signal at: **{chosen_signal}**")
    # ----------------------------------------------------------
    # REINFORCEMENT LEARNING DECISION DETAILS
    # ----------------------------------------------------------
    st.write("### 🤖 Reinforcement Learning Decision")

    st.write(
        """
    The DQN agent observes the current traffic state and selects
    the traffic signal that is expected to minimize emergency
    response time.
    """
    )

    st.write("**State Vector:**")
    st.json({
        "Queue Length (m)": queue_length,
        "Speed (kmph)": speed,
        "Predicted Clearance (km)": round(clearance, 3),
        "Signal E on Route": bool(rl_debug["has_E"]),
        "Signal B on Route": bool(rl_debug["has_B"]),
    })

    st.write("**DQN Action Output:**")
    st.code(
        f"Action = {rl_debug['action']}  →  Select Signal {rl_debug['action_meaning']}"
    )

    st.success(
        f"RL Agent selected Signal {chosen_signal} to create the optimal green corridor."
    )
    st.write("RL selected signal:", chosen_signal)

    # ----------------------------------------------------------
    # AUTO-SCALE GRAPH
    # Scale so clearance distance is reached BEFORE signal
    # ----------------------------------------------------------
    route_to_signal = route[: route.index(chosen_signal) + 1]
    graph_units = compute_route_units(route_to_signal, positions)

    clearance_m = clearance * 1000
    SCALE_MULTIPLIER = 2.5

    scale_factor = max(50, (clearance_m / graph_units) * SCALE_MULTIPLIER)

    st.info(f"Auto-scale: **1 unit = {scale_factor:.1f} meters**")

    # ----------------------------------------------------------
    # VISUAL + ANIMATION
    # ----------------------------------------------------------
    fig = create_base_figure(
        G,
        positions,
        route,
        incident_edge,
        chosen_signal
    )

    st.write("TRACE COUNT IN BASE FIGURE:", len(fig.data))

    placeholder = st.empty()

    animate_ev(
        fig=fig,
        route=route,
        positions=positions,
        placeholder=placeholder,
        clearance_km=clearance,
        chosen_signal=chosen_signal,
        scale_factor=scale_factor
    )

else:
    st.info("Click Run Simulation to begin.")

# Debug output
print(list(G.nodes()))
print(list(G.edges()))