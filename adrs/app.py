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
import streamlit.components.v1 as components
import adrs_core as core
import adrs_agents
import adrs_simulation
from langgraph.types import Command

st.set_page_config(page_title="ADRS Simulation", layout="wide")
MPD = core.MIN_PER_DAY
COMP_COLORS = {"C1": "#7fb3ff", "C2": "#ff6b6b", "C3": "#e8554e", "C4": "#4ec9b0",
               "C5": "#3fb6a8", "C6": "#f5d76e", "OutSrc": "#3b6bd6", "Assembly": "#b388ff"}
STATUS_COLORS = {
    "Completed_In House": "#5cb85c",
    "Completed_Outsource": "#2e7d32",
    "InTransit_Outsource": "#3b6bd6",
    "Late": "#e8554e",
    "Late_Outsource": "#c0392b",       # distinct red for a late outsourced part
    "InProgress_In House": "#9e9e9e",
    "Not started": "#444",
}


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

if "graph" not in st.session_state:
    st.session_state.graph = adrs_agents.build_app()
    st.session_state.thread_id = 0
    st.session_state.cfg = None

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
    dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
    end = p["start"] + dur
    due = o["due"]
    if o["outsourced"]:
        if end <= clock:
            return "Late_Outsource" if end > due else "Completed_Outsource"
        if clock > due:
            return "Late_Outsource"          # deadline passed, part still not back
        return "InTransit_Outsource"
    if end <= clock:
        return "Late" if end > due else "Completed_In House"
    if p["start"] <= clock < end:
        return "Late" if clock > due else "InProgress_In House"
    return "Not started"


def render_option_kpis(option, ops):
    """Headline whole-plan KPIs for this option: tardiness, makespan, churn,
    ASM utilisation, and average queueing waits -- computed once per option,
    not clock-dependent."""
    det = core.option_deterministic_stats(ops, option["plan"], st.session_state.actual_returns)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Tardiness", core.fmt_wd(option["total_tardiness"]))
    k2.metric("Makespan", core.fmt_wd(option["makespan"]))
    k3.metric("Ops moved", option["ops_changed"])
    k4.metric("ASM utilisation", f"{det['asm_utilisation_pct']:.0f}%")
    k5.metric("Avg in-house wait", core.fmt_wd(det["avg_inhouse_wait"]))
    k6.metric("Avg ASM wait", core.fmt_wd(det["avg_asm_wait"]))


def render_option_stats(option, ops):
    """Static, whole-plan statistics for this option: per-machine utilisation,
    workload (queue length), and average wait, plus per-product completion times."""
    plan = option["plan"]
    util = core.compute_machine_utilisation(ops, plan)
    waits = core.compute_waiting_times(ops, plan, st.session_state.actual_returns)
    stats = core.compute_job_stats(ops, plan, st.session_state.actual_returns)

    machine_of = {(o["job"], o["comp"]): o["machine"] for o in ops
                 if o["kind"] == "comp" and not o["outsourced"]}
    wait_by_machine = {}
    for w in waits:
        if w["kind"] == "in-house":
            mc = machine_of.get((w["job"], w["comp"]))
            if mc:
                wait_by_machine.setdefault(mc, []).append(w["wait"])

    machines = sorted({o["machine"] for o in ops if o["machine"] and o["machine"] != core.ASM_STATION})
    rows = []
    for mc in machines:
        n_ops = sum(1 for o in ops if o["machine"] == mc)
        wlist = wait_by_machine.get(mc, [])
        rows.append(dict(Machine=mc, **{
            "Utilisation": f"{util.get(mc, {}).get('pct', 0):.0f}%",
            "Queue length (total jobs)": n_ops,
            "Avg wait": core.fmt_wd(sum(wlist) / len(wlist)) if wlist else "0.0wd",
        }))
    asm_waits = [w["wait"] for w in waits if w["kind"] == "assembly"]
    n_asm = sum(1 for o in ops if o["kind"] == "asm")
    rows.append(dict(Machine="ASM", **{
        "Utilisation": f"{util.get(core.ASM_STATION, {}).get('pct', 0):.0f}%",
        "Queue length (total jobs)": n_asm,
        "Avg wait": core.fmt_wd(sum(asm_waits) / len(asm_waits)) if asm_waits else "0.0wd",
    }))
    st.markdown("**Machine statistics**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                key=f"machinestats_{option['objective']}")

    st.markdown("**Product completion times**")
    comp_rows = []
    for j in sorted(stats):
        s = stats[j]
        comp_rows.append(dict(Product=f"P{j}", **{
            "Completion time": core.fmt_wd(s["completion"]),
            "Due": core.fmt_wd(s["due"]),
            "Status": "On time" if s["on_time"] else f"Late by {core.fmt_wd(s['tardiness'])}",
        }))
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True,
                key=f"completiontimes_{option['objective']}")


def render_options_panel():
    opts = st.session_state.get("options")
    advice = st.session_state.get("advice")
    if not opts or not advice:
        return
    st.subheader("Rescheduling options")

    def label(i):
        o = opts[i]
        star = " ⭐" if o["objective"] == advice["recommended"] else ""
        return (f"{o['label']}{star} — tardiness {core.fmt_wd(o['total_tardiness'])}, "
                f"makespan {core.fmt_wd(o['makespan'])}, {o['ops_changed']} ops moved")

    default = next((i for i, o in enumerate(opts)
                    if o["objective"] == advice["recommended"]), 0)
    choice = st.radio("Choose a plan to apply:", range(len(opts)), index=default,
                      format_func=label, key=f"optchoice_{st.session_state.opt_nonce}")

    st.caption("**Recommended because:** " + advice["reason"])
    for o in opts:
        ex = advice["explanations"].get(o["objective"])
        if ex:
            st.caption(f"• **{o['label']}** — {ex}")

    selected = opts[choice]
    render_option_kpis(selected, ops)

    st.markdown("**🏭 Factory-floor simulation**")
    st.caption("Runs a discrete-event simulation (SimPy) of the selected option and opens it "
               "as an animated factory floor in a new browser tab. The KPIs shown there are "
               "measured live from the simulation itself (arrival/start/end of every "
               "operation), not read off the optimiser's plan.")
    sim_result = adrs_simulation.simulate(ops, selected["plan"], st.session_state.actual_returns)
    widget = adrs_simulation.build_launch_widget(
        ops, selected["plan"], st.session_state.actual_returns,
        option_label=selected["label"], sim_result=sim_result)
    components.html(widget, height=70)

    with st.expander("Machine & product statistics for this option"):
        render_option_stats(selected, ops)

    if st.button("✓ Apply selected plan"):
        chosen = opts[choice]
        cfg = st.session_state.cfg
        if cfg is None:
            st.warning("No active reschedule to apply — run Reschedule first.")
            st.stop()
        with st.spinner("Explaining applied plan…"):
            final = st.session_state.graph.invoke(
                Command(resume=chosen["objective"]), cfg)   # resume into explainability
        st.session_state.committed = final["reschedule"]["plan"]   # rescheduling proceeds
        st.session_state.applied_explanation = final["explanation"]
        st.session_state.explanation = None
        st.session_state.options = None
        st.session_state.advice = None
        st.rerun()


def render_stats_panel():
    st.subheader("📊 Schedule Statistics")

    stats = core.compute_job_stats(ops, committed, st.session_state.actual_returns)
    util = core.compute_machine_utilisation(ops, committed)
    waits = core.compute_waiting_times(ops, committed, st.session_state.actual_returns)

    jobs_sorted = sorted(stats)
    on_time_jobs = [j for j in jobs_sorted if stats[j]["on_time"]]
    late_jobs = [j for j in jobs_sorted if not stats[j]["on_time"]]
    completed_now = [j for j in jobs_sorted if stats[j]["completion"] <= clock]
    remaining_now = [j for j in jobs_sorted if stats[j]["completion"] > clock]

    # 1. headline counts
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("On-time (predicted)", f"{len(on_time_jobs)}/{len(jobs_sorted)}")
    c2_.metric("Late (predicted)", f"{len(late_jobs)}/{len(jobs_sorted)}")
    c3_.metric("Completed so far", f"{len(completed_now)}/{len(jobs_sorted)}")
    c4_.metric("Remaining", f"{len(remaining_now)}/{len(jobs_sorted)}")

    # 2. machine utilisation
    st.markdown("**Machine utilisation** (busy time ÷ schedule makespan)")
    util_df = pd.DataFrame([
        dict(Machine=mc, Utilisation=f"{v['pct']:.0f}%", **{"Busy time": core.fmt_wd(v['busy'])})
        for mc, v in sorted(util.items())
    ])
    st.dataframe(util_df, hide_index=True, use_container_width=True)

    # 3. average waiting time
    st.markdown("**Average waiting time**")
    inhouse_waits = [w["wait"] for w in waits if w["kind"] == "in-house"]
    asm_waits = [w["wait"] for w in waits if w["kind"] == "assembly"]
    wc1, wc2 = st.columns(2)
    wc1.metric("Avg. in-house component queue wait",
              core.fmt_wd(sum(inhouse_waits) / len(inhouse_waits)) if inhouse_waits else "0.0wd")
    wc2.metric("Avg. assembly (ASM) queue wait",
              core.fmt_wd(sum(asm_waits) / len(asm_waits)) if asm_waits else "0.0wd")
    with st.expander("Waiting time detail per product/component"):
        wait_df = pd.DataFrame([
            dict(Product=f"Product {w['job']}", Component=w["comp"], Type=w["kind"],
                 **{"Wait": core.fmt_wd(w["wait"])})
            for w in sorted(waits, key=lambda w: (w["job"], w["kind"]))
        ])
        st.dataframe(wait_df, hide_index=True, use_container_width=True)

    # 5. late products
    st.markdown("**⚠️ Late products**")
    if late_jobs:
        late_df = pd.DataFrame([
            dict(Product=f"Product {j}",
                 **{"Late by": core.fmt_wd(stats[j]["tardiness"]),
                    "Expected completion": wm_to_dt(stats[j]["completion"]).strftime("%d %b %Y %H:%M"),
                    "Due": wm_to_dt(stats[j]["due"]).strftime("%d %b %Y %H:%M"),
                    "Caused by": stats[j]["cause"]})
            for j in late_jobs
        ])
        st.dataframe(late_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No products currently predicted late. 🎉")

    # 6. on-time products
    st.markdown("**✅ On-time products**")
    if on_time_jobs:
        ok_df = pd.DataFrame([
            dict(Product=f"Product {j}",
                 **{"Expected completion": wm_to_dt(stats[j]["completion"]).strftime("%d %b %Y %H:%M"),
                    "Due": wm_to_dt(stats[j]["due"]).strftime("%d %b %Y %H:%M"),
                    "Slack": core.fmt_wd(stats[j]["due"] - stats[j]["completion"])})
            for j in on_time_jobs
        ])
        st.dataframe(ok_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No products currently on time.")


render_options_panel()


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
    st.session_state.disruption = None
    st.session_state.decision = None
    st.session_state.impact_reasoning = None
    st.session_state.options = None
    st.session_state.advice = None
    st.session_state.explanation = None
    st.session_state.applied_explanation = None
    st.session_state.cfg = None
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
        ar = {i: d for i, d in inputs.items()}
        st.session_state.actual_returns = ar
        st.session_state.applied_explanation = None
        st.session_state.thread_id += 1
        cfg = {"configurable": {"thread_id": str(st.session_state.thread_id)}}
        st.session_state.cfg = cfg
        with st.spinner("ADRS agents reasoning…"):
            final = st.session_state.graph.invoke(
                {"ops": ops, "committed_plan": committed,
                 "now": int(clock), "actual_returns": ar}, cfg)            
        st.session_state.disruption = final["disruption"]
        st.session_state.decision = final["decision"]
        st.session_state.impact_reasoning = final["impact_reasoning"]
        if final.get("__interrupt__"):                 # suspended at human_pick
            st.session_state.options = final["options"]   # full options (with plan)
            st.session_state.advice = final["advice"]
            st.session_state.explanation = None
        else:                                          # no_action -> finished
            st.session_state.options = None
            st.session_state.advice = None
            st.session_state.explanation = final.get("explanation")
        st.session_state.opt_nonce = st.session_state.get("opt_nonce", 0) + 1
        st.rerun()

    if st.session_state.get("disruption"):
        st.info(f"**Detection** — severity: {st.session_state.disruption['severity']}.")
        st.info(f"**Decision** — {st.session_state.decision}. {st.session_state.impact_reasoning}")
    if st.session_state.get("explanation") and not st.session_state.get("options"):
        st.success(st.session_state.explanation)             # no_action narrative
    if st.session_state.get("applied_explanation"):
        st.success(st.session_state.applied_explanation)
        

# ---------- Gantt ----------

tab1, tab2 = st.tabs([ "Sequence & parallelism (real time)", "By machine"])

with tab1:
    actual = st.session_state.actual_returns
    prod_order = [f"Product {j}" for j in sorted({o["job"] for o in ops})]

    fig2 = go.Figure()
    seen = set()
    for o in ops:
        p = committed[o["idx"]]
        dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
        label = "OutSrc" if o["outsourced"] else o["comp"]
        mc = o["machine"] if o["machine"] else "Outsourced (off-site)"
        fig2.add_trace(go.Bar(
            y=[f"Product {o['job']}"], x=[dur / MPD], base=[p["start"] / MPD],
            orientation="h",
            marker=dict(color=COMP_COLORS.get(label, "#888"), opacity=0.85,
                    line=dict(width=0.5, color="#111")),
            name=label, legendgroup=label, showlegend=(label not in seen),
            text=label, textposition="inside", insidetextanchor="middle",
            textangle=0, constraintext="none",
            hovertemplate=(f"Product {o['job']} · {label}<br>"
                        f"Machine: {mc}<br>"
                        f"start {p['start']/MPD:.2f} wd<br>"
                        f"duration {dur/MPD:.2f} wd ({dur/60:.1f} h)<extra></extra>")))
        fig2.add_trace(go.Bar(
            y=[f"Product {o['job']}"], x=[dur / MPD], base=[p["start"] / MPD],
            orientation="h",
            marker=dict(color=COMP_COLORS.get(label, "#888"), opacity=0.85,
                        line=dict(width=0.5, color="#111")),
            name=label, legendgroup=label, showlegend=(label not in seen),
            text=label, textposition="inside", insidetextanchor="middle",
            textangle=0, constraintext="none",
            hovertemplate=(f"Product {o['job']} · {label}<br>"
                            f"start {p['start']/MPD:.2f} wd<br>"
                            f"duration {dur/MPD:.2f} wd ({dur/60:.1f} h)<extra></extra>")))
        seen.add(label)

    fig2.add_vline(x=clock / MPD, line_width=2, line_dash="dash", line_color="white")
    fig2.update_layout(barmode="overlay", height=760, legend_title="Component",
                        yaxis=dict(categoryorder="array",
                                categoryarray=list(reversed(prod_order))),
                        xaxis_title="working-days from release (Aug 21)",
                        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Real timeline — bar position = actual scheduled time. Overlapping bars on "
                "a product's row mean genuine parallel work on different machines. Hover any "
                "bar for exact start/duration; box-zoom to inspect short operations.")
    
with tab2:
    actual = st.session_state.actual_returns
    jobs = sorted({o["job"] for o in ops})
    palette = px.colors.qualitative.Plotly
    cmap = {j: palette[(j - 1) % len(palette)] for j in jobs}
    out_rows = sorted({f"OutSrc P{o['job']}" for o in ops if o["outsourced"]})
    order = out_rows + ["ASM", "M6", "M5", "M4", "M3", "M2", "M1"]
        
    fig3 = go.Figure()
    seen = set()
    for o in ops:
        p = committed[o["idx"]]
        dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
        resource = f"OutSrc P{o['job']}" if o["outsourced"] else o["machine"]
        fig3.add_trace(go.Bar(
            y=[resource], x=[dur / MPD], base=[p["start"] / MPD], orientation="h",
            marker_color=cmap[o["job"]], marker_line_width=0.5, marker_line_color="#111",
            text=f"P{o['job']}·{o['comp']}", textposition="inside", insidetextanchor="middle",
            textangle=0, constraintext="none", cliponaxis=False,
            legendgroup=f"P{o['job']}", name=f"P{o['job']}",
            showlegend=(o["job"] not in seen),
            hovertemplate=(f"<b>P{o['job']} · {o['comp']}</b><br>"
                        f"Machine: {resource}<br>"
                        f"start: {p['start']/MPD:.2f} wd<br>"
                        f"duration: {dur/MPD:.2f} wd ({dur/60:.1f} h)<extra></extra>")))
        seen.add(o["job"])

    fig3.add_vline(x=clock / MPD, line_width=2, line_dash="dash", line_color="white")
    fig3.update_layout(barmode="overlay", height=760, bargap=0.15,
                    xaxis_title="working-days from release (Aug 21)",
                    yaxis=dict(categoryorder="array", categoryarray=order),
                    legend_title="Product", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ---------- component status grid ----------

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


# statistics
render_stats_panel() 

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