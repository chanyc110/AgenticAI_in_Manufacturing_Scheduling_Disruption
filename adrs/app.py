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

st.set_page_config(page_title="ADRS Production Schedulling and Resource Allocation", layout="wide") # change back to scheduling simulation
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


def _agent_box(label, text, bg, accent):
    """Agent narrative panel.

    st.info/st.success render at Streamlit's default body size in a muted
    theme colour, which was hard to read on a projector and unreadable once
    screenshotted into a document. This is the same shape -- tinted panel with a
    coloured left rule -- at a larger size with black body text."""
    head = (f'<span style="color:{accent};font-weight:700">{label}</span> — '
            if label else "")
    st.markdown(
        f'<div style="background:{bg};border-left:6px solid {accent};'
        f'border-radius:6px;padding:14px 18px;margin:10px 0;">'
        f'<p style="color:#000000;font-size:22px;line-height:1.6;margin:0;">'
        f'{head}{text}</p></div>',
        unsafe_allow_html=True)

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

@st.cache_data(show_spinner=False)
def _cached_simulate(plan_key, actual_key, _ops, _plan, _actual):
    """Args prefixed with _ are not hashed by Streamlit; plan_key/actual_key are
    the real cache keys. Stops the clock's autoplay loop from re-running SimPy
    three times per frame when nothing about the plan has actually changed."""
    return adrs_simulation.simulate(_ops, _plan, _actual)


def cached_simulate(ops, plan, actual):
    plan_key = tuple(sorted((i, p["start"], p["end"]) for i, p in plan.items()))
    actual_key = tuple(sorted(actual.items()))
    return _cached_simulate(plan_key, actual_key, ops, plan, actual)



if "graph" not in st.session_state:
    st.session_state.graph = adrs_agents.build_app()
    st.session_state.thread_id = 0
    st.session_state.cfg = None

# ---------- state ----------
if "ops" not in st.session_state:
    st.session_state.ops = core.load_ops()
if "committed" not in st.session_state:
    st.session_state.committed = core.build_schedule(st.session_state.ops, now=0)["plan"]
if "original_committed" not in st.session_state:
    st.session_state.original_committed = st.session_state.committed
if "clock" not in st.session_state:
    st.session_state.clock = 0
if "actual_returns" not in st.session_state:
    st.session_state.actual_returns = {}
if "playing" not in st.session_state:
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
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tardiness", core.fmt_wd(option["total_tardiness"]))
    k2.metric("Makespan", core.fmt_wd(option["makespan"]))
    k3.metric("ASM utilisation", f"{det['asm_utilisation_pct']:.0f}%")
    k4.metric("Avg in-house wait", core.fmt_wd(det["avg_inhouse_wait"]))
    k5.metric("Avg ASM wait", core.fmt_wd(det["avg_asm_wait"]))


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
                f"makespan {core.fmt_wd(o['makespan'])}")

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
    sim_result = cached_simulate(ops, selected["plan"], st.session_state.actual_returns)
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


def render_before_after_panel():
    """Without Agentic AI vs With Agentic AI, for whatever disruption is
    currently entered. Deliberately uses adrs_simulation.simulate() rather
    than core.compute_job_stats()/compute_machine_utilisation() -- those are
    computed from the static committed plan's start/end times and don't
    right-shift for real queueing delay, so they understate (often to zero)
    the true impact of a disruption when nothing has been rescheduled yet.
    simulate()'s event log reflects what actually happens on the shop floor."""
    actual = st.session_state.actual_returns
    if not actual:
        return

    st.subheader("📊 Without vs With Agentic AI")
    dis_jobs = {ops[i]["job"] for i in actual
            if i < len(ops) and actual[i] != ops[i]["dur"]}

    des_dn = cached_simulate(ops, st.session_state.original_committed, actual)
    des_wi = cached_simulate(ops, committed, actual)  # == do-nothing until a reschedule is applied

    def _stats(des):
        jobs_kpi = des["jobs"]
        late = [j for j, v in jobs_kpi.items() if v["tardiness"] > 0]
        collateral = [j for j in late if j not in dis_jobs]
        asm_util = des["resources"].get(core.ASM_STATION, {}).get("utilisation_pct", 0.0)
        return dict(late=len(late), total=len(jobs_kpi),
                    tardiness_wd=des["overall"]["total_tardiness"] / MPD,
                    collateral=len(collateral), asm_util=asm_util)

    dn, wi = _stats(des_dn), _stats(des_wi)

    c1, c2 = st.columns(2)
    for col, label, s in [(c1, "Without Agentic AI", dn), (c2, "With Agentic AI", wi)]:
        with col:
            st.markdown(f"**{label}**")
            m1, m2 = st.columns(2)
            m1.metric("Late products", f"{s['late']}/{s['total']}")
            m2.metric("Total tardiness", f"{s['tardiness_wd']:.1f}wd")
            m3, m4 = st.columns(2)
            m3.metric("Collateral delays", s["collateral"])
            m4.metric("ASM utilisation", f"{s['asm_util']:.0f}%")

    if committed is st.session_state.original_committed:
        st.caption("No reschedule applied yet -- both sides show the same do-nothing outcome.")


def render_stats_panel(clock):
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
    st.dataframe(util_df, hide_index=True, use_container_width=True, key="util_df")

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
        st.dataframe(wait_df, hide_index=True, use_container_width=True, key="wait_df")

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
        st.dataframe(late_df, hide_index=True, use_container_width=True, key="late_df")
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
        st.dataframe(ok_df, hide_index=True, use_container_width=True,key="ontime_df")
    else:
        st.caption("No products currently on time.")


render_options_panel()


# ---------- header / controls ----------
st.title("ADRS Production Schedulling and Resource Allocation")
c1, c2, c4 = st.columns(3)
if c1.button("▶ Start"):
    st.session_state.playing = True
if c2.button("⏸ Pause"):
    st.session_state.playing = False
if c4.button("↺ Reset"):
    st.session_state.reset_nonce += 1
    st.session_state.clock = 0
    st.session_state.actual_returns = {}
    st.session_state.committed = core.build_schedule(ops, now=0)["plan"]
    st.session_state.original_committed = st.session_state.committed
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
spd1, spd2 = st.columns(2)
step_wd = spd1.slider("Step per frame (working-days)", 0.05, 1.0, 0.2, 0.05)
delay = spd2.slider("Frame delay (seconds)", 0.1, 2.0, 0.7, 0.1)



def render_live_view(step_wd):
    """Everything that changes as the clock ticks lives here. During autoplay this
    fragment reruns on its own (scope='fragment'), so the title, control buttons and
    the options/simulation panel above are NOT torn down and rebuilt every frame --
    that iframe reload was the worst of the flicker. A reschedule still triggers a
    normal full-app rerun, which is what we want."""
    st.session_state.clock = st.slider(
        "Clock (working-days)", 0.0, round(horizon / MPD, 1),
        round(st.session_state.clock / MPD, 1), 0.1) * MPD
    clock = st.session_state.clock
    st.caption(f"Now: {wm_to_dt(clock).strftime('%a %d %b %Y %H:%M')}  ·  {clock/MPD:.1f} working-days")

    # ---------- input actual outsource returns + reschedule ----------
    with st.expander("Pause and input actual outsource returns, then reschedule", expanded=True):
        inputs = {}
        for o in core.outsourced_ops(ops):
            dispatch_day = committed[o["idx"]]["start"] / MPD
            planned = committed[o["idx"]]["start"] + o["dur"]
            default_val = round(planned / MPD, 1)
            key = f"ar_{o['idx']}_{st.session_state.reset_nonce}"

            current_val = st.session_state.get(key, default_val)
            
            val = st.number_input(
                f"Product {o['job']} · {o['comp']} actual return (working-day)",
                min_value=round(dispatch_day, 1),
                value=current_val,
                step=0.5,
                key=key
            )
            
            inputs[o["idx"]] = int(round(val * MPD)) - committed[o["idx"]]["start"]

        
        if st.button("⟳ Apply actual returns and reschedule"):
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
            if final.get("__interrupt__"):
                st.session_state.options = final["options"]
                st.session_state.advice = final["advice"]
                st.session_state.explanation = None
            else:
                st.session_state.options = None
                st.session_state.advice = None
                st.session_state.explanation = final.get("explanation")
            st.session_state.opt_nonce = st.session_state.get("opt_nonce", 0) + 1
            st.rerun()          # full-app rerun on purpose -- options panel must rebuild

        if st.session_state.get("disruption"):
            _agent_box("Detection", f"severity: {st.session_state.disruption['severity']}.",
                   "#eef4ff", "#3d6bb3")
            _agent_box("Decision",
                   f"{st.session_state.decision}. {st.session_state.impact_reasoning}",
                   "#eef4ff", "#3d6bb3")
        if st.session_state.get("explanation") and not st.session_state.get("options"):
            _agent_box(None, st.session_state.explanation, "#eefaf0", "#2f8f5b")
        if st.session_state.get("applied_explanation"):
            _agent_box(None, st.session_state.applied_explanation, "#eefaf0", "#2f8f5b")

    # ---------- Gantt ----------
    tab1, tab2 = st.tabs(["Sequence & parallelism (real time)", "By machine"])

    with tab1:
        actual = st.session_state.actual_returns
        prod_order = [f"Product {j}" for j in sorted({o["job"] for o in ops})]
        groups = {}
        for o in ops:
            p = committed[o["idx"]]
            dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
            label = "OutSrc" if o["outsourced"] else o["comp"]
            g = groups.setdefault(label, dict(y=[], x=[], base=[], cd=[]))
            g["y"].append(f"Product {o['job']}")
            g["x"].append(dur / MPD)
            g["base"].append(p["start"] / MPD)
            g["cd"].append([o["job"], label, o["machine"] or "Outsourced (off-site)",
                            p["start"] / MPD, dur / MPD, dur / 60])

        fig2 = go.Figure()
        for label, g in groups.items():
            fig2.add_trace(go.Bar(
                y=g["y"], x=g["x"], base=g["base"], orientation="h",
                marker=dict(color=COMP_COLORS.get(label, "#888"), opacity=0.85,
                            line=dict(width=0.5, color="#111")),
                name=label, legendgroup=label, customdata=g["cd"],
                text=[label] * len(g["y"]), textposition="inside",
                insidetextanchor="middle", textangle=0, constraintext="none",
                hovertemplate=("Product %{customdata[0]} · %{customdata[1]}<br>"
                               "Machine: %{customdata[2]}<br>"
                               "start %{customdata[3]:.2f} wd<br>"
                               "duration %{customdata[4]:.2f} wd "
                               "(%{customdata[5]:.1f} h)<extra></extra>")))

        fig2.add_vline(x=clock / MPD, line_width=2, line_dash="dash", line_color="white")
        fig2.update_layout(barmode="overlay", height=760, legend_title="Component",
                            yaxis=dict(categoryorder="array",
                                    categoryarray=list(reversed(prod_order))),
                            xaxis_title="working-days from release (Aug 21)",
                            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True, key="gantt_sequence")
        st.caption("Real timeline — bar position = actual scheduled time. Overlapping bars on "
                    "a product's row mean genuine parallel work on different machines. Hover any "
                    "bar for exact start/duration; box-zoom to inspect short operations.")

    with tab2:
        actual = st.session_state.actual_returns
        jobs = sorted({o["job"] for o in ops})
        palette = px.colors.qualitative.Plotly
        cmap = {j: palette[(j - 1) % len(palette)] for j in jobs}

        # each in-house machine produces exactly one component type in this
        # dataset -- fold that into the row label instead of repeating the
        # component on every single bar
        machine_comp = {}
        for o in ops:
            if o["kind"] == "comp" and not o["outsourced"]:
                machine_comp.setdefault(o["machine"], o["comp"])
        machine_order = ["M6", "M5", "M4", "M3", "M2", "M1"]
        row_label = {m: f"{m} ({machine_comp[m]})" for m in machine_order}
        row_label["ASM"] = "Assembly"

        out_rows = sorted({f"OutSrc P{o['job']}" for o in ops if o["outsourced"]})
        order = out_rows + ["Assembly"] + [row_label[m] for m in machine_order]

        jgroups = {}
        for o in ops:
            p = committed[o["idx"]]
            dur = actual.get(o["idx"], o["dur"]) if o["outsourced"] else o["dur"]
            if o["outsourced"]:
                resource = f"OutSrc P{o['job']}"
            else:
                resource = row_label.get(o["machine"], o["machine"])
            g = jgroups.setdefault(o["job"], dict(y=[], x=[], base=[], text=[], cd=[]))
            g["y"].append(resource)
            g["x"].append(dur / MPD)
            g["base"].append(p["start"] / MPD)
            g["text"].append(f"P{o['job']}")   # component is now implied by the row
            g["cd"].append([o["job"], o["comp"], resource,
                            p["start"] / MPD, dur / MPD, dur / 60])

        figm = go.Figure()
        for j, g in sorted(jgroups.items()):
            figm.add_trace(go.Bar(
                y=g["y"], x=g["x"], base=g["base"], orientation="h",
                marker_color=cmap[j], marker_line_width=0.5, marker_line_color="#111",
                text=g["text"], textposition="inside", insidetextanchor="middle",
                textangle=0, constraintext="none", cliponaxis=False,
                legendgroup=f"P{j}", name=f"P{j}", customdata=g["cd"],
                hovertemplate=("<b>P%{customdata[0]} · %{customdata[1]}</b><br>"
                               "Machine: %{customdata[2]}<br>"
                               "start: %{customdata[3]:.2f} wd<br>"
                               "duration: %{customdata[4]:.2f} wd "
                               "(%{customdata[5]:.1f} h)<extra></extra>")))

        figm.add_vline(x=clock / MPD, line_width=2, line_dash="dash", line_color="white")
        figm.update_layout(
            barmode="overlay", height=760, bargap=0.15,
            title=dict(text="Each machine produces one component type: "
                            "M1→C1, M2→C2, M3→C3, M4→C4, M5→C5, M6→C6",
                       font=dict(size=12, color="#888"), x=0, xanchor="left"),
            xaxis_title="working-days from release (Aug 21)",
            yaxis=dict(categoryorder="array", categoryarray=order),
            legend_title="Product", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(figm, use_container_width=True, key="gantt_by_machine")

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
    figs = px.scatter(sdf, x="Product", y="Component", color="Status", text="Machine",
                        color_discrete_map=STATUS_COLORS,
                        category_orders={"Component": ["C1", "C2", "C3", "C4", "C5", "C6"]})
    figs.update_traces(marker=dict(size=26, symbol="square"), textposition="top center")
    figs.update_yaxes(autorange="reversed")
    figs.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figs, use_container_width=True, key="status_grid")

    render_stats_panel(clock)

    # ---------- Completion Race ----------
    st.subheader("Completion Race  🏁")
    jobs = sorted({o["job"] for o in ops})
    palette = px.colors.qualitative.Plotly
    cmap = {j: palette[(j - 1) % len(palette)] for j in jobs}
    completion = {o["job"]: committed[o["idx"]]["end"] for o in ops if o["kind"] == "asm"}
    product_order = [f"Product {j}" for j in jobs]

    figr = go.Figure()
    for i, j in enumerate(jobs):
        ct = completion[j]
        ct_wd = ct / MPD
        done = clock >= ct
        pos_wd = min(clock, ct) / MPD
        figr.add_shape(type="line", x0=ct_wd, x1=ct_wd, y0=i - 0.45, y1=i + 0.45,
                       xref="x", yref="y",
                       line=dict(color="#e8554e", width=2, dash="dot"))
        figr.add_trace(go.Scatter(
            x=[pos_wd], y=[f"Product {j}"], mode="markers+text",
            marker=dict(size=26, color=cmap[j], symbol="square" if done else "circle",
                        line=dict(width=1, color="#111")),
            text=[f"{'📦' if done else '🏃'} P{j}"], textposition="middle right",
            showlegend=False,
            hovertemplate=(f"Product {j}<br>"
                           f"{'DONE' if done else f'{min(clock/ct,1.0)*100:.0f}% to assembled'}<br>"
                           f"finish line at {ct_wd:.1f} wd<extra></extra>")))

    max_ct_wd = max(completion.values()) / MPD
    figr.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=product_order,
                      tickmode="array", tickvals=product_order, ticktext=product_order)
    figr.update_xaxes(range=[-0.3, max_ct_wd * 1.12],
                      title="working-days from release (each product's own 🏁 finish line in red)")
    figr.update_layout(height=max(360, 34 * len(jobs)), margin=dict(l=0, r=0, t=22, b=0))
    st.plotly_chart(figr, use_container_width=True, key="completion_race")

    # ---------- autoplay ----------
    # No st.rerun() and no time.sleep() here: when playing, this function is wrapped
    # in a fragment with run_every=delay, so Streamlit itself reruns just this
    # fragment on a timer. We only advance the clock; the timer does the rest.
    if st.session_state.playing and clock < horizon:
        st.session_state.clock = min(clock + step_wd * MPD, horizon)


render_before_after_panel()

# Wrap in a fragment whose timer is active only while playing. When paused,
# run_every=None means no timed reruns at all -- so a paused dashboard sits
# completely still instead of repainting in the background.
_runner = st.fragment(run_every=(delay if st.session_state.playing else None))(render_live_view)
_runner(step_wd)