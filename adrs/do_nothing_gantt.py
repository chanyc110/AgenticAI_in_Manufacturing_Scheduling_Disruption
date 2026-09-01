"""
Generates the DO-NOTHING Gantt chart for the late_1 DEMO instance
(Job 1, Component C3: assumed 3wd, actually returns 12wd late).

Verified against your actual adrs_simulation.py: simulate() emits one event
per operation with idx, start, end, dur, resource, outsourced -- so this
matches the real, queue-delayed schedule under no rescheduling (as opposed to
app.py's dashboard Gantt, which draws every bar from the STATIC committed
plan and never right-shifts downstream ops when a part comes back late).

Run from the same directory as your other ADRS files:
    python do_nothing_gantt.py
"""
import plotly.graph_objects as go

import adrs_core as core
import adrs_simulation as sim

# NOTE: compute_now / DEMO_REORDER_INSTANCES are inlined here rather than
# imported from adrs_experiments.py -- that module imports adrs_agents at
# the top level, which instantiates the Groq LLM client on import and
# crashes without GROQ_API_KEY set, even though this script makes zero LLM
# calls. Values below are copied verbatim from adrs_experiments.py.

MPD = core.MIN_PER_DAY
COMP_COLORS = {"C1": "#7fb3ff", "C2": "#ff6b6b", "C3": "#e8554e", "C4": "#4ec9b0",
               "C5": "#3fb6a8", "C6": "#f5d76e", "OutSrc": "#3b6bd6", "Assembly": "#b388ff"}


def compute_now(ops, committed, ar):
    """Copied from adrs_experiments.py -- causality rule for when the
    disruption is discovered (see that module for the full docstring)."""
    if not ar:
        return 0
    times = []
    for idx, actual in ar.items():
        dispatch = committed[idx]["start"]
        assumed = ops[idx]["dur"]
        times.append(dispatch + assumed if actual > assumed else dispatch + actual)
    return max(times)


DEMO_LATE_1 = {"ar": {1: 12 * MPD},
               "desc": "Job 1 C3: LATE by 9wd (assumed 3wd)"}  # copied from DEMO_REORDER_INSTANCES["late_1"]

# ---- reproduce the exact late_1 demo instance ----
ops = core.load_ops(path="adrs/job_shop_20products.xlsx")
nominal = core.build_schedule(ops, now=0)
committed = nominal["plan"]

ar = DEMO_LATE_1["ar"]
desc = DEMO_LATE_1["desc"]
now = compute_now(ops, committed, ar)

# ---- do-nothing analytical tardiness (matches the CSV row, ASM-level only) ----
dn = core.evaluate_donothing(ops, committed, ar, now=now)
print(f"Scenario: {desc}")
print(f"Do-nothing total tardiness: {dn['total_tardiness'] / MPD:.1f} wd")

# ---- DES run: gives REAL, queue-delayed start times for EVERY operation ----
des_dn = sim.simulate(ops, committed, ar)
events = des_dn["events"]
print(f"DES do-nothing total tardiness: {des_dn['overall']['total_tardiness'] / MPD:.1f} wd")
print(f"Late jobs: {des_dn['overall']['jobs_total'] - des_dn['overall']['jobs_on_time']} / "
      f"{des_dn['overall']['jobs_total']}")

ev_by_idx = {e["idx"]: e for e in events}
prod_order = [f"Product {j}" for j in sorted({o["job"] for o in ops})]

fig = go.Figure()
seen = set()
for o in ops:
    ev = ev_by_idx[o["idx"]]         # every op has exactly one event -- direct lookup
    start, dur = ev["start"], ev["dur"]

    label = "Assembly" if o["kind"] == "asm" else ("OutSrc" if o["outsourced"] else o["comp"])
    mc = ev["resource"]
    fig.add_trace(go.Bar(
        y=[f"Product {o['job']}"], x=[dur / MPD], base=[start / MPD],
        orientation="h",
        marker=dict(color=COMP_COLORS.get(label, "#888"), opacity=0.85,
                    line=dict(width=0.5, color="#111")),
        name=label, legendgroup=label, showlegend=(label not in seen),
        text=label, textposition="inside", insidetextanchor="middle",
        textangle=0, constraintext="none",
        hovertemplate=(f"Product {o['job']} · {label}<br>"
                        f"Resource: {mc}<br>"
                        f"real start {start/MPD:.2f} wd<br>"
                        f"duration {dur/MPD:.2f} wd ({dur/60:.1f} h)<br>"
                        f"waited {ev['wait']/MPD:.2f} wd<extra></extra>")))
    seen.add(label)

fig.add_vline(x=now / MPD, line_width=2, line_dash="dash", line_color="white",
              annotation_text="disruption known", annotation_position="top")
fig.update_layout(
    barmode="overlay", height=760,
    title=f"Do-Nothing Schedule (real, queue-delayed) — {desc}",
    legend_title="Component",
    yaxis=dict(categoryorder="array", categoryarray=list(reversed(prod_order))),
    xaxis_title="working-days from release",
    margin=dict(l=0, r=0, t=40, b=0))

fig.write_html("do_nothing_gantt.html")
print("Wrote do_nothing_gantt.html")
try:
    fig.write_image("do_nothing_gantt.png", width=1600, height=900, scale=2)
    print("Wrote do_nothing_gantt.png")
except Exception as e:
    # PNG export needs Chrome available to kaleido -- if it's not installed,
    # run `plotly_get_chrome` once, or just screen-record/screenshot the
    # HTML version directly, which is equally fine for the video.
    print(f"(PNG export skipped -- {e}. The HTML file above works fine for recording.)")