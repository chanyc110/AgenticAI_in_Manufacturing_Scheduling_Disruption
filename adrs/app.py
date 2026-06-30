"""
ADRS scheduling simulation (Streamlit + Plotly) over adrs_core.
Clock + Pause + input actual outsource returns + Reschedule (calls reschedule_from).
Run:  streamlit run app.py
"""
import time
from datetime import timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import adrs_core as core
import adrs_agents

st.set_page_config(page_title="ADRS Simulation", layout="wide")
MPD = core.MIN_PER_DAY
COMP_COLORS = {"C1": "#7fb3ff", "C2": "#ff6b6b", "C3": "#e8554e", "C4": "#4ec9b0",
               "C5": "#3fb6a8", "C6": "#f5d76e", "OutSrc": "#3b6bd6", "Assembly": "#b388ff"}
STATUS_COLORS = {"Completed_In House": "#5cb85c", "Completed_Outsource": "#2e7d32",
                 "Late": "#e8554e", "InProgress_In House": "#9e9e9e", "Not started": "#444"}


def wm_to_dt(m):
    """Working-minute -> real datetime (inverse of core.to_work_minutes)."""
    cur, rem = core.ORIGIN, max(0, m)
    while rem > 0:
        if cur.weekday() < 5:
            end = cur.replace(hour=17, minute=0)
            avail = (end - cur).total_seconds() / 60
            if rem <= avail:
                return cur + timedelta(minutes=rem)
            rem -= avail
        cur = (cur.replace(hour=17, minute=0) + timedelta(days=1)).replace(hour=9, minute=0)
    return cur


# ---------- state ----------
if "ops" not in st.session_state:
    st.session_state.ops = core.load_ops()
    st.session_state.committed = core.build_schedule(st.session_state.ops, now=0)["plan"]
    st.session_state.clock = 0
    st.session_state.actual_returns = {}
    st.session_state.playing = False

if "reset_nonce" not in st.session_state:
    st.session_state.reset_nonce = 0

ops = st.session_state.ops
committed = st.session_state.committed
horizon = max(p["end"] for p in committed.values())


def op_status(o, clock):
    p = committed[o["idx"]]
    actual = st.session_state.actual_returns
    end = p["start"] + (actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"])
    due = o["due"]
    if end <= clock:
        if o["outsourced"]:
            return "Completed_Outsource"
        return "Late" if end > due else "Completed_In House"
    if p["start"] <= clock < end:
        return "Late" if clock > due else "InProgress_In House"
    return "Not started"


# ---------- header / controls ----------
st.title("ADRS Scheduling Simulation")
c1, c2, c3, c4 = st.columns(4)
if c1.button("▶ Start"):
    st.session_state.playing = True
if c2.button("⏸ Pause"):
    st.session_state.playing = False
if c4.button("↺ Reset"):
    st.session_state.reset_nonce += 1         # CHANGE 2: bump keys -> inputs reset to defaults
    st.session_state.clock = 0
    st.session_state.actual_returns = {}
    st.session_state.committed = core.build_schedule(ops, now=0)["plan"]
    st.session_state.playing = False
    st.rerun()

st.session_state.clock = st.slider("Clock (working-days)", 0.0, round(horizon / MPD, 1),
                                   round(st.session_state.clock / MPD, 1), 0.1) * MPD
clock = st.session_state.clock
st.caption(f"Now: {wm_to_dt(clock).strftime('%a %d %b %Y %H:%M')}  ·  {clock/MPD:.1f} working-days")

spd1, spd2 = st.columns(2)
step_wd = spd1.slider("Step per frame (working-days)", 0.05, 1.0, 0.2, 0.05)
delay = spd2.slider("Frame delay (seconds)", 0.1, 2.0, 0.7, 0.1)

# ---------- input actual outsource returns + reschedule ----------
with st.expander("Pause and input actual outsource returns, then reschedule", expanded=True):
    inputs = {}
    for o in core.outsourced_ops(ops):
        planned = committed[o["idx"]]["start"] + o["dur"]
        val = st.number_input(f"Product {o['job']} · {o['comp']} actual return (working-day)",
                              min_value=0.0, value=round(planned / MPD, 1), step=0.5,
                              key=f"ar_{o['idx']}_{st.session_state.reset_nonce}")  # CHANGE 3
        inputs[o["idx"]] = int(round(val * MPD)) - committed[o["idx"]]["start"]  # -> actual lead
    if c3.button("⟳ Reschedule") or st.button("Apply actual returns and reschedule"):
        st.session_state.actual_returns = {i: d for i, d in inputs.items()}
        with st.spinner("ADRS agents reasoning…"):
            final = adrs_agents.build_app().invoke({       # build fresh, not cached
                "ops": ops,
                "committed_plan": committed,
                "now": int(clock),
                "actual_returns": st.session_state.actual_returns,
            })
        if "reschedule" in final:                      # agent chose to reschedule
            st.session_state.committed = final["reschedule"]["plan"]
            committed = final["reschedule"]["plan"]
        st.info(f"**Detection** — severity: {final['disruption']['severity']}. "
                f"{final['disruption']['reasoning']}")
        st.info(f"**Decision** — {final['decision']}. {final['impact_reasoning']}")
        st.success(final["explanation"])

# ---------- Gantt ----------
left, right = st.columns([1.3, 1])
with left:
    st.subheader("Gantt Chart (by product)")
    actual = st.session_state.actual_returns
    rows = []
    for o in ops:
        dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
        label = "OutSrc" if o["outsourced"] else o["comp"]   # 'Assembly' for the asm row
        rows.append(dict(Product=f"Product {o['job']}", Component=label,
                         WD=dur / MPD, Hours=round(dur / 60, 1)))
    gdf = pd.DataFrame(rows)
    comp_order = ["C1", "C2", "C3", "C4", "C5", "C6", "OutSrc", "Assembly"]
    prod_order = [f"Product {j}" for j in sorted({o['job'] for o in ops})]
    fig = px.bar(gdf, x="WD", y="Product", color="Component", orientation="h",
                 color_discrete_map=COMP_COLORS,
                 category_orders={"Component": comp_order, "Product": prod_order},
                 hover_data={"Hours": True, "WD": ":.2f"})
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="stack", height=480, legend_title="Component",
                      xaxis_title="processing time, stacked by component (working-days)",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------- component status grid ----------
with right:
    st.subheader("Product Components Status")
    pts = []
    for o in ops:
        if o["kind"] != "comp":
            continue
        pts.append(dict(Product=f"Product {o['job']}", Component=o["comp"],
                        Status=op_status(o, clock),
                        Machine=("OutSrc" if o["outsourced"] else o["machine"])))
    sdf = pd.DataFrame(pts)
    fig2 = px.scatter(sdf, x="Product", y="Component", color="Status", text="Machine",
                      color_discrete_map=STATUS_COLORS,
                      category_orders={"Component": ["C1", "C2", "C3", "C4", "C5", "C6"]})
    fig2.update_traces(marker=dict(size=26, symbol="square"), textposition="top center")
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


st.subheader("Completion Race  🏁")
jobs = sorted({o["job"] for o in ops})
palette = px.colors.qualitative.Plotly
cmap = {j: palette[(j - 1) % len(palette)] for j in jobs}
completion = {o["job"]: committed[o["idx"]]["end"] for o in ops if o["kind"] == "asm"}

fig3 = go.Figure()
for j in jobs:
    ct = completion[j]
    frac = 1.0 if ct == 0 else min(clock / ct, 1.0)
    done = clock >= ct
    fig3.add_trace(go.Scatter(
        x=[frac], y=[f"Product {j}"], mode="markers+text",
        marker=dict(size=26, color=cmap[j], symbol="square" if done else "circle",
                    line=dict(width=1, color="#111")),
        text=[f"{'📦' if done else '🏃'} P{j}"], textposition="middle right",
        showlegend=False,
        hovertemplate=(f"Product {j}<br>"
                       f"{'DONE' if done else f'{frac*100:.0f}% to assembled'}<br>"
                       f"completes at {ct/MPD:.1f} wd<extra></extra>")))
fig3.add_vline(x=0.0, line_color="#888")
fig3.add_vline(x=1.0, line_color="#5cb85c", line_dash="dash")
fig3.add_annotation(x=1.0, y=1.04, yref="paper", text="🏁 finish",
                    showarrow=False, font=dict(color="#5cb85c"))
fig3.update_yaxes(autorange="reversed")
fig3.update_xaxes(range=[-0.05, 1.25], tickformat=".0%",
                  title="progress to assembled")
fig3.update_layout(height=360, margin=dict(l=0, r=0, t=22, b=0))
st.plotly_chart(fig3, use_container_width=True)

# ---------- autoplay ----------
if st.session_state.playing and clock < horizon:
    time.sleep(delay)
    st.session_state.clock = min(clock + step_wd * MPD, horizon)
    st.rerun()