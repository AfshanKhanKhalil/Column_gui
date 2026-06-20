import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# ------------------------------
# Core logic (same as before)
# ------------------------------

_BAR_NOS = ["None"] + [f"#{n}" for n in range(1, 19)]

def _bar_dia(bar_label: str) -> float:
    if not bar_label or bar_label == "None":
        return 0.0
    n = int(bar_label.lstrip("#"))
    return n / 8.0

def _bar_area(bar_label: str) -> float:
    if not bar_label or bar_label == "None":
        return 0.0
    d = _bar_dia(bar_label)
    return math.pi * d ** 2 / 4.0

# (keep your helper functions unchanged...)

# ------------------------------
# Plotting functions (figsize adjusted)
# ------------------------------

def plot_section(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, fc, fy):
    fig, ax = plt.subplots(figsize=(8, 8))   # bigger size
    # ... (rest of your plotting code unchanged)
    plt.tight_layout()
    return fig

def plot_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, load_cases_df):
    Mn, Pn, phiMn, phiPn = _compute_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    fig, ax = plt.subplots(figsize=(9, 7))   # bigger size
    # ... (rest unchanged)
    plt.tight_layout()
    return fig

def plot_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n):
    kappa, M = _compute_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    fig, ax = plt.subplots(figsize=(9, 7))   # bigger size
    # ... (rest unchanged)
    plt.tight_layout()
    return fig

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(layout="wide")
st.title("RC Column Interaction Diagram Tool (ACI 318)")

with st.sidebar:
    st.header("Column Geometry")
    b = st.number_input("Width b (in)", value=24.0, min_value=4.0, step=1.0)
    h = st.number_input("Depth d (in)", value=24.0, min_value=4.0, step=1.0)

    st.header("Reinforcement")
    cover = st.number_input("Clear cover (in)", value=1.5, min_value=0.5, step=0.25)

    top_bar_no = st.selectbox("Top bar size", _BAR_NOS, index=_BAR_NOS.index("#8"))
    top_bar_n = st.number_input("Number of top bars", value=4, min_value=1, step=1) if top_bar_no != "None" else 0

    bot_bar_no = st.selectbox("Bottom bar size", _BAR_NOS, index=_BAR_NOS.index("#8"))
    bot_bar_n = st.number_input("Number of bottom bars", value=4, min_value=1, step=1) if bot_bar_no != "None" else 0

    side_bar_no = st.selectbox("Side bar size", _BAR_NOS, index=_BAR_NOS.index("#6"))
    side_bar_n = st.number_input("Number of side bars per side", value=2, min_value=1, step=1) if side_bar_no != "None" else 0

    st.header("Materials")
    fc = st.number_input("Concrete f'c (ksi)", value=4.0, min_value=1.0, step=0.5)
    fy = st.number_input("Steel fy (ksi)", value=60.0, min_value=10.0, step=5.0)

    st.header("Load Cases")
    default_loads = pd.DataFrame({"Pu (kips)": [500.0], "Mu (kip·in)": [3000.0]})
    load_cases_df = st.data_editor(default_loads, num_rows="dynamic", use_container_width=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Section Plot", "P-M Diagram", "Moment-Curvature", "Results Summary"])

with tab1:
    fig1 = plot_section(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, fc, fy)
    st.pyplot(fig1, use_container_width=True)

with tab2:
    if load_cases_df.empty:
        st.warning("No load cases defined. Add some in the sidebar.")
    else:
        fig2 = plot_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, load_cases_df)
        st.pyplot(fig2, use_container_width=True)

with tab3:
    fig3 = plot_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    st.pyplot(fig3, use_container_width=True)

with tab4:
    if not load_cases_df.empty:
        Mn, Pn, phiMn, phiPn = _compute_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
        results = []
        for idx, row in load_cases_df.iterrows():
            pu = row['Pu (kips)']
            mu = row['Mu (kip·in)']
            ok = _point_in_diagram(mu, pu, phiMn, phiPn)
            results.append({"Load Case": idx+1, "Pu (kips)": pu, "Mu (kip·in)": mu, "Status": "OK" if ok else "FAIL"})
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("No load cases to display.")
