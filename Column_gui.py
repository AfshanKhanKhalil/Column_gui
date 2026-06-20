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

def _get_bar_positions(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n):
    stirrup_dia = 0.375
    top_active = top_bar_no not in (None, "None")
    bot_active = bot_bar_no not in (None, "None")
    side_active = side_bar_no not in (None, "None")

    top_dia = _bar_dia(top_bar_no) if top_active else 0.0
    bot_dia = _bar_dia(bot_bar_no) if bot_active else 0.0
    side_dia = _bar_dia(side_bar_no) if side_active else 0.0

    top_As = _bar_area(top_bar_no) if top_active else 0.0
    bot_As = _bar_area(bot_bar_no) if bot_active else 0.0
    side_As = _bar_area(side_bar_no) if side_active else 0.0

    def dc(dia):
        return cover + stirrup_dia + dia / 2

    n_top = top_bar_n if top_active else 0
    n_bot = bot_bar_n if bot_active else 0
    n_side = side_bar_n if side_active else 0

    bars = []
    # bottom bars
    if bot_active and n_bot > 0:
        y_bot = dc(bot_dia)
        bars.append((y_bot, n_bot * bot_As))
    # top bars
    if top_active and n_top > 0:
        y_top = h - dc(top_dia)
        bars.append((y_top, n_top * top_As))
    # side bars
    if side_active and n_side > 0:
        y_ref_bot = dc(bot_dia) if bot_active else dc(side_dia)
        y_ref_top = (h - dc(top_dia)) if top_active else (h - dc(side_dia))
        y_side_start = y_ref_bot + (dc(side_dia) - dc(bot_dia if bot_active else side_dia))
        y_side_end = y_ref_top - (dc(side_dia) - dc(top_dia if top_active else side_dia))
        for i in range(n_side):
            y = y_side_start + (i + 1) * (y_side_end - y_side_start) / (n_side + 1)
            bars.append((y, 2 * side_As))
    return bars

def _aci318_phi(et):
    if et <= 0.002:
        return 0.65
    elif et >= 0.005:
        return 0.90
    else:
        return 0.65 + (et - 0.002) * (0.90 - 0.65) / (0.005 - 0.002)

def _compute_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, n_points=60):
    Es = 29000.0
    eps_cu = 0.003
    beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4.0) / 1.0))
    bars = _get_bar_positions(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    Ag = b * h
    Ast = sum(a for _, a in bars)

    Mn_list, Pn_list, phiMn_list, phiPn_list = [], [], [], []

    # pure compression
    Pn0 = 0.85 * fc * (Ag - Ast) + fy * Ast
    Pn0_max = 0.80 * Pn0
    Mn_list.append(0.0); Pn_list.append(Pn0_max)
    phiMn_list.append(0.0); phiPn_list.append(0.65 * Pn0_max)

    c_values = np.concatenate([
        np.linspace(h * 2.0, h * 0.5, n_points // 3),
        np.linspace(h * 0.5, h * 0.1, n_points // 3),
        np.linspace(h * 0.1, h * 0.01, n_points // 3),
    ])

    for c in c_values:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * b * a
        bars_from_top = sorted([(h - y, As) for y, As in bars], key=lambda x: x[0])
        Fs_total, Fs_moment = 0.0, 0.0
        et_extreme = None
        for dist_from_top, As in bars_from_top:
            eps_s = eps_cu * (c - dist_from_top) / c
            fs = max(-fy, min(fy, Es * eps_s))
            if dist_from_top < a:
                fs_net = fs - 0.85 * fc
            else:
                fs_net = fs
            Fs = fs_net * As
            Fs_total += Fs
            arm = h / 2 - dist_from_top
            Fs_moment += Fs * arm
            if et_extreme is None or eps_s < et_extreme:
                et_extreme = eps_s
        Pn = Cc + Fs_total
        arm_cc = h / 2 - a / 2
        Mn = Cc * arm_cc + Fs_moment
        Mn = abs(Mn)
        et = et_extreme if et_extreme is not None else -fy / Es
        phi = _aci318_phi(et)
        Mn_list.append(Mn); Pn_list.append(Pn)
        phiMn_list.append(phi * Mn); phiPn_list.append(phi * Pn)

    # pure tension
    Pnt = -fy * Ast
    Mn_list.append(0.0); Pn_list.append(Pnt)
    phiMn_list.append(0.0); phiPn_list.append(0.90 * Pnt)

    return np.array(Mn_list), np.array(Pn_list), np.array(phiMn_list), np.array(phiPn_list)

def _point_in_diagram(Mu, Pu, phiMn, phiPn):
    pts = list(zip(phiMn, phiPn))
    pts.append(pts[0])
    inside = False
    n = len(pts) - 1
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if ((y1 > Pu) != (y2 > Pu)) and (Mu < (x2 - x1) * (Pu - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside

def _compute_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, n_points=50):
    Es = 29000.0
    eps_cu = 0.003
    beta1 = max(0.65, min(0.85, 0.85 - 0.05 * (fc - 4.0) / 1.0))
    bars = _get_bar_positions(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)

    kappa_max = eps_cu / (0.1 * h)
    kappas = np.linspace(1e-8, kappa_max, n_points)
    kappa_list, M_list = [0.0], [0.0]

    for kappa in kappas:
        c = h / 2
        for _ in range(100):
            eps_top = kappa * c
            a = min(beta1 * c, h)
            Cc = 0.85 * fc * b * a
            Fs_total = 0.0
            for y_bot, As in bars:
                dist_from_top = h - y_bot
                eps_s = kappa * (c - dist_from_top)
                fs = max(-fy, min(fy, Es * eps_s))
                if dist_from_top < a:
                    fs_net = fs - 0.85 * fc
                else:
                    fs_net = fs
                Fs_total += fs_net * As
            residual = Cc + Fs_total
            dR_dc = 0.85 * fc * b * beta1
            dc = -residual / (dR_dc + 1e-6)
            c = max(0.01 * h, c + dc)
            if abs(dc) < 1e-6 * h:
                break
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * b * a
        arm_cc = h / 2 - a / 2
        M = Cc * arm_cc
        for y_bot, As in bars:
            dist_from_top = h - y_bot
            eps_s = kappa * (c - dist_from_top)
            fs = max(-fy, min(fy, Es * eps_s))
            if dist_from_top < a:
                fs_net = fs - 0.85 * fc
            else:
                fs_net = fs
            arm = h / 2 - dist_from_top
            M += fs_net * As * arm
        eps_top = kappa * c
        if eps_top > eps_cu * 1.05:
            break
        kappa_list.append(kappa)
        M_list.append(abs(M))
    return np.array(kappa_list), np.array(M_list)

# ------------------------------
# Plotting functions
# ------------------------------

def plot_section(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, fc, fy):
    stirrup_dia = 0.375
    top_active = top_bar_no not in (None, "None")
    bot_active = bot_bar_no not in (None, "None")
    side_active = side_bar_no not in (None, "None")

    fig, ax = plt.subplots(figsize=(8, 8))
    # concrete outline
    rect = plt.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#D3D3D3')
    ax.add_patch(rect)
    # stirrup
    dc_stirrup = cover + stirrup_dia / 2
    stirrup_rect = plt.Rectangle((dc_stirrup, dc_stirrup), b-2*dc_stirrup, h-2*dc_stirrup,
                                  linewidth=1.2, edgecolor='grey', facecolor='none', linestyle='--')
    ax.add_patch(stirrup_rect)

    # bar radii
    bar_radius_top = _bar_dia(top_bar_no)/2 if top_active else 0
    bar_radius_bot = _bar_dia(bot_bar_no)/2 if bot_active else 0
    bar_radius_side = _bar_dia(side_bar_no)/2 if side_active else 0

    y_top_bar = h - (cover + stirrup_dia + bar_radius_top)
    y_bot_bar = cover + stirrup_dia + bar_radius_bot
    n_top = top_bar_n if top_active else 0
    n_bot = bot_bar_n if bot_active else 0
    n_side = side_bar_n if side_active else 0

    dc_top = cover + stirrup_dia + bar_radius_top
    dc_bot = cover + stirrup_dia + bar_radius_bot
    dc_side = cover + stirrup_dia + bar_radius_side

    # top bars
    if top_active and n_top > 0:
        xs_top = [b/2] if n_top == 1 else np.linspace(dc_top, b-dc_top, n_top).tolist()
        for x in xs_top:
            ax.add_patch(plt.Circle((x, y_top_bar), bar_radius_top, color='black', zorder=5))
    # bottom bars
    if bot_active and n_bot > 0:
        xs_bot = [b/2] if n_bot == 1 else np.linspace(dc_bot, b-dc_bot, n_bot).tolist()
        for x in xs_bot:
            ax.add_patch(plt.Circle((x, y_bot_bar), bar_radius_bot, color='black', zorder=5))
    # side bars
    if side_active and n_side > 0:
        y_side_start = y_bot_bar + (dc_side - dc_bot) if bot_active else dc_side
        y_side_end = y_top_bar - (dc_side - dc_top) if top_active else h - dc_side
        x_left, x_right = dc_side, b - dc_side
        for i in range(n_side):
            y = y_side_start + (i+1)*(y_side_end - y_side_start)/(n_side+1)
            ax.add_patch(plt.Circle((x_left, y), bar_radius_side, color='black', zorder=5))
            ax.add_patch(plt.Circle((x_right, y), bar_radius_side, color='black', zorder=5))

    # dimension arrows
    ax.annotate("", xy=(b, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(b/2, -2.5, f"b = {b}\"", ha="center", fontsize=9)
    ax.annotate("", xy=(b+1.5, h), xytext=(b+1.5, 0), arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(b+3.5, h/2, f"d = {h}\"", ha="center", fontsize=9, rotation=90)

    ax.set_xlim(-4, b+6); ax.set_ylim(-4, h+4)
    ax.set_aspect('equal')
    ax.set_title(f"Column Cross-Section ({b}\" × {h}\")\nf'c = {fc} ksi  |  fy = {fy} ksi", fontsize=11)
    ax.axis('off')

    # info box
    def bar_info(label, active, no):
        if not active:
            return f"{label}: — None —"
        return f"{label}: {no}  d={_bar_dia(no):.4f}\"  A={_bar_area(no):.4f} in²"
    info_lines = ["Rebar Sizes (ASTM)",
                  bar_info("Top ", top_active, top_bar_no),
                  bar_info("Bot ", bot_active, bot_bar_no),
                  bar_info("Side", side_active, side_bar_no)]
    ax.text(0.01, 0.01, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    rebar_patch = mpatches.Patch(color="black", label="Rebar")
    ax.legend(handles=[rect, stirrup_rect, rebar_patch], loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

def plot_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, load_cases_df):
    Mn, Pn, phiMn, phiPn = _compute_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    fig, ax = plt.subplots(figsize=(8, 10))
    # Updated legend labels
    ax.plot(Mn, Pn, '-b', lw=2, label='Nominal Curve')
    ax.plot(phiMn, phiPn, '--r', lw=2, label='Design Curve')
    ax.fill_betweenx(phiPn, 0, phiMn, alpha=0.08, color='red')

    if load_cases_df is not None and not load_cases_df.empty:
        for i, row in load_cases_df.iterrows():
            pu = row['Pu (kips)']
            mu = row['Mu (kip·in)']
            ok = _point_in_diagram(mu, pu, phiMn, phiPn)
            color = 'green' if ok else 'red'
            marker = '^' if ok else 'x'
            ax.scatter(mu, pu, color=color, marker=marker, s=120, zorder=6)
            ax.annotate(f"LC{i+1}: {'OK' if ok else 'FAIL'}", (mu, pu), xytext=(8,4),
                        textcoords='offset points', fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Moment Mn (kip·in)', fontsize=12)
    ax.set_ylabel('Axial Force Pn (kips)', fontsize=12)
    ax.set_title(f"P-M Interaction Diagram — ACI 318\n{b}\"×{h}\" Column  |  f'c={fc} ksi  |  fy={fy} ksi", fontsize=11)
    ax.axhline(0, color='k', ls=':'); ax.axvline(0, color='k', ls=':')
    ax.legend(fontsize=10); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    return fig

def plot_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n):
    kappa, M = _compute_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(kappa*1000, M, '-b', lw=2)
    ax.fill_between(kappa*1000, M, alpha=0.12, color='blue')
    ax.set_xlabel('Curvature φ (×10⁻³ / in)', fontsize=12)
    ax.set_ylabel('Moment M (kip·in)', fontsize=12)
    ax.set_title(f"Moment-Curvature Diagram (P = 0)\n{b}\"×{h}\" Column  |  f'c={fc} ksi  |  fy={fy} ksi", fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    return fig

# ------------------------------
# Streamlit UI with conditional number inputs
# ------------------------------

st.set_page_config(layout="wide")
st.title("RC Column Interaction Diagram Tool (ACI 318)")

# Sidebar inputs
with st.sidebar:
    st.header("Column Geometry")
    b = st.number_input("Width b (in)", value=24.0, min_value=4.0, step=1.0)
    h = st.number_input("Depth d (in)", value=24.0, min_value=4.0, step=1.0)

    st.header("Reinforcement")
    cover = st.number_input("Clear cover (in)", value=1.5, min_value=0.5, step=0.25)

    # Top bars
    top_bar_no = st.selectbox("Top bar size", _BAR_NOS, index=_BAR_NOS.index("#8"))
    if top_bar_no != "None":
        top_bar_n = st.number_input("Number of top bars", value=4, min_value=1, step=1)
    else:
        top_bar_n = 0
        st.info("No top bars (set bar size to 'None')")

    # Bottom bars
    bot_bar_no = st.selectbox("Bottom bar size", _BAR_NOS, index=_BAR_NOS.index("#8"))
    if bot_bar_no != "None":
        bot_bar_n = st.number_input("Number of bottom bars", value=4, min_value=1, step=1)
    else:
        bot_bar_n = 0
        st.info("No bottom bars (set bar size to 'None')")

    # Side bars
    side_bar_no = st.selectbox("Side bar size", _BAR_NOS, index=_BAR_NOS.index("#6"))
    if side_bar_no != "None":
        side_bar_n = st.number_input("Number of side bars per side", value=2, min_value=1, step=1)
    else:
        side_bar_n = 0
        st.info("No side bars (set bar size to 'None')")

    st.header("Materials")
    fc = st.number_input("Concrete f'c (ksi)", value=4.0, min_value=1.0, step=0.5)
    fy = st.number_input("Steel fy (ksi)", value=60.0, min_value=10.0, step=5.0)

    st.header("Load Cases")
    st.caption("Edit the table below (add/delete rows as needed)")
    default_loads = pd.DataFrame({"Pu (kips)": [500.0], "Mu (kip·in)": [3000.0]})
    load_cases_df = st.data_editor(default_loads, num_rows="dynamic", use_container_width=True)

# Main area tabs
tab1, tab2, tab3, tab4 = st.tabs(["Section Plot", "P-M Diagram", "Moment-Curvature", "Results Summary"])

with tab1:
    fig1 = plot_section(b, h, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, fc, fy)
    st.pyplot(fig1)

with tab2:
    if load_cases_df.empty:
        st.warning("No load cases defined. Add some in the sidebar.")
    else:
        fig2 = plot_pm_diagram(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n, load_cases_df)
        st.pyplot(fig2)

with tab3:
    fig3 = plot_moment_curvature(b, h, fc, fy, cover, top_bar_no, top_bar_n, bot_bar_no, bot_bar_n, side_bar_no, side_bar_n)
    st.pyplot(fig3)

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
