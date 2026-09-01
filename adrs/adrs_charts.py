"""
ADRS chart generator -- renders the experiment results with AG Charts.

Why a separate file
-------------------
AG Charts is a JavaScript library, so it cannot draw into a Python figure. This
script reads the CSVs that adrs_experiments.py already writes and emits
SELF-CONTAINED HTML pages: open one in a browser and every chart renders from
the AG Charts CDN. For the dissertation, screenshot the chart or use the
browser's Print -> Save as PDF. Nothing here re-runs the experiments or touches
the LLM agents.

    # per-scenario page (Gantt charts + that scenario's event timeline)
    python adrs_charts.py --scenario s1.csv --out s1_charts.html

    # cross-scenario page (tardiness, utilisation, queueing, waiting, late jobs)
    python adrs_charts.py --combine s0.csv s1.csv s2.csv s3.csv s4.csv s5.csv \
                          --out all_charts.html

Licensing note
--------------
Everything drawn here uses chart types in AG Charts COMMUNITY (free): bar, line,
scatter and combinations. Heatmap and range-bar are Enterprise features and
render with a watermark without a licence, so the Product x Machine heatmaps
(fig4/5/8/9) stay in matplotlib in adrs_experiments.py, and the Gantt charts
below are built from a STACKED BAR rather than a range-bar: each resource row is
a run of alternating transparent "gap" segments and coloured "operation"
segments, which produces the same picture using only Community features.
"""
import argparse
import json
import os

import pandas as pd

import adrs_core as core

# Deliberately standalone: importing adrs_experiments would pull in langgraph
# and the Groq-backed agents, so drawing a chart would need the whole LLM stack
# configured. These few definitions are duplicated instead -- keep them in step
# with adrs_experiments.py if the scenario set ever changes.
SCENARIO_KINDS = ["baseline", "late_1", "late_2", "early_1", "early_2", "mixed"]
PLOT_LABELS = {"baseline": "Baseline", "late_1": "Scenario 1", "late_2": "Scenario 2",
               "early_1": "Scenario 3", "early_2": "Scenario 4", "mixed": "Scenario 5"}


def _details_path(out):
    return os.path.splitext(out)[0] + "_details.json"


def _condition_labels(kind):
    """The baseline compares scheduling METHODS (list schedule vs optimiser);
    every other scenario holds the optimiser fixed and varies only whether the
    agent rescheduled."""
    if kind == "baseline":
        return "Without Optimisation", "With Optimisation"
    return "Without Agentic AI", "With Agentic AI"


def load_run(path):
    """Read one scenario's result CSV plus its sidecar detail JSON."""
    combined = pd.read_csv(path)
    run_df = combined[combined.record == "run"].copy()
    ev_df = combined[combined.record == "event"].copy()
    if ev_df.empty:
        ev_df = None
    details = []
    det_path = _details_path(path)
    if os.path.exists(det_path):
        with open(det_path) as f:
            details = json.load(f)
    else:
        print(f"(note: {det_path} not found)")
    return run_df, ev_df, details

MPD = core.MIN_PER_DAY
CDN = "https://cdn.jsdelivr.net/npm/ag-charts-community/dist/umd/ag-charts-community.js"

# colour-blind-safe qualitative palette, reused across every chart
PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951",
           "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"]
PRODUCT_COLOURS = [
    "#0000FF",  # P1  blue
    "#FF0000",  # P2  red
    "#00FF00",  # P3  green
    "#FF00B6",  # P4  magenta
    "#005300",  # P5  dark green
    "#FFD300",  # P6  yellow
    "#009FFF",  # P7  sky blue
    "#9A4D42",  # P8  brick brown
    "#00FFBE",  # P9  aquamarine
    "#783FC1",  # P10 violet
    "#1F9698",  # P11 teal
    "#FFACFD",  # P12 pale pink
    "#B1CC71",  # P13 olive
    "#101820",  # P14 near-black
    "#FE8F42",  # P15 orange
    "#F91F60",  # P16 crimson
    "#720055",  # P17 plum
    "#766C95",  # P18 grey-violet
    "#29A908",  # P19 mid green
    "#BFF324",  # P20 lime
]
WITHOUT_COLOUR, WITH_COLOUR = "#d1495b", "#2a9d8f"


# Applied to every AG chart from _page, so styling stays consistent and there is
# one place to change it. Axis labels are pure black rather than the default
# grey, which reproduces poorly once a screenshot is printed.
CHART_THEME = {
    "overrides": {
        "common": {
            "legend": {"item": {"label": {"fontSize": 16, "color": "#000000"}}},
        }
    }
}


def _page(title, blocks):
    """One HTML page holding every chart. `blocks` is a list of
    (heading, note, options_dict) -- `note` is accepted for call-site
    compatibility but no longer rendered. Sized for A4-width screenshots."""
    divs, scripts = [], []
    for n, (heading, _note, opts) in enumerate(blocks):
        opts = dict(opts, theme=CHART_THEME)
        divs.append(f'<section><h2>{heading}</h2>'
                    f'<div id="chart{n}" class="chart"></div></section>')
        scripts.append(
            f'agCharts.AgCharts.create(Object.assign('
            f'{json.dumps(opts)}, {{container: document.getElementById("chart{n}")}}));')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="{CDN}"></script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          margin: 0 auto; padding: 24px; max-width: 1100px; background: #fff; color: #000; }}
  h1 {{ text-align: center; }}
  h2 {{ font-size: 22px; margin: 32px 0 10px; font-weight: 700; color: #000;
        text-align: center; }}
  .chart {{ height: 520px; margin-bottom: 12px; }}
  section {{ page-break-inside: avoid; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head><body>
<h1>{title}</h1>
{"".join(divs)}
<script>{"".join(scripts)}</script>
</body></html>"""


def _axes(x_title, y_title, x_type="category"):
    """Axis config in the DICTIONARY form the current AG Charts API expects
    (keys `x` and `y`). The older array-of-axes form is silently ignored by
    recent versions: the chart still renders, because axis types are inferred
    from the data, but every explicit setting -- titles included -- is dropped.
    That is why axis titles were missing despite being configured."""
    title = lambda text: {"text": text, "fontSize": 18,
                          "fontWeight": "bold", "color": "#000000"}
    label = {"fontSize": 16, "color": "#000000"}
    return {"x": {"type": x_type, "position": "bottom",
                  "title": title(x_title), "label": label},
            "y": {"type": "number", "position": "left",
                  "title": title(y_title), "label": label}}


# ======================================================= cross-scenario charts ====
def chart_tardiness(df):
    """Line: total tardiness per scenario, without vs with agentic AI."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        eff = r["donothing_tardiness_wd"] - r["tardiness_reduction_wd"]
        data.append({"scenario": PLOT_LABELS[k],
                     "without": round(float(r["donothing_tardiness_wd"]), 2),
                     "with": round(float(eff), 2)})
    return {
        "data": data,
        "series": [
            {"type": "line", "xKey": "scenario", "yKey": "without",
             "yName": "Without Agentic AI", "stroke": WITHOUT_COLOUR,
             "strokeWidth": 2,
             "marker": {"fill": WITHOUT_COLOUR, "stroke": WITHOUT_COLOUR, "size": 8}},
            {"type": "line", "xKey": "scenario", "yKey": "with",
             "yName": "With Agentic AI", "stroke": WITH_COLOUR,
             "strokeWidth": 2,
             "marker": {"fill": WITH_COLOUR, "stroke": WITH_COLOUR, "size": 8}}],
        "axes": _axes("Scenario", "Total tardiness (working-days)"),
        "legend": {"position": "bottom"},
    }


def chart_late_jobs(df):
    """Grouped bar: how many of the 20 products miss their date."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    data = [{"scenario": PLOT_LABELS[k],
             "without": int(df[df.scenario == k].iloc[0]["late_jobs_donothing"]),
             "with": int(df[df.scenario == k].iloc[0]["late_jobs_reschedule"])}
            for k in kinds]
    return {
        "data": data,
        "series": [
            {"type": "bar", "xKey": "scenario", "yKey": "without",
             "yName": "Without Agentic AI", "fill": WITHOUT_COLOUR,
             "label": {"enabled": True, "color": "#000000",
                       "fontSize": 16, "fontWeight": "bold"}},
            {"type": "bar", "xKey": "scenario", "yKey": "with",
             "yName": "With Agentic AI", "fill": WITH_COLOUR,
             "label": {"enabled": True, "color": "#000000",
                       "fontSize": 16, "fontWeight": "bold"}}],
        "axes": _axes("Scenario", "Late products (of 20)"),
        "legend": {"position": "bottom"},
    }


def _resource_series(df, prefix, y_title, unit_note):
    """Per-resource line chart with a TWO-PART legend instead of fourteen entries.

    Colour identifies the resource and line style identifies the condition, so
    the legend needs seven colour entries plus two style entries -- not one per
    combination. AG Charts has no second legend, so the style key is drawn as
    two invisible marker-only series carrying the labels, and every real series
    is hidden from the legend except one per resource."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    resources = sorted([c[len(f"des_donothing_{prefix}_"):] for c in df.columns
                        if c.startswith(f"des_donothing_{prefix}_")],
                       key=lambda r: (r == core.ASM_STATION, r))
    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        row = {"scenario": PLOT_LABELS[k]}
        acted = r["impact_decision"] == "reschedule"
        for res in resources:
            dn = r.get(f"des_donothing_{prefix}_{res}")
            rs = r.get(f"des_reschedule_{prefix}_{res}")
            row[f"{res}_without"] = round(float(dn), 2) if pd.notna(dn) else None
            use = rs if (acted and pd.notna(rs)) else dn
            row[f"{res}_with"] = round(float(use), 2) if pd.notna(use) else None
        data.append(row)

    series = []
    for i, res in enumerate(resources):
        c = PALETTE[i % len(PALETTE)]
        # only the solid line carries the resource's legend entry
        series.append({"type": "line", "xKey": "scenario", "yKey": f"{res}_with",
                       "yName": res, "stroke": c, "strokeWidth": 2.5,
                       "marker": {"fill": c, "stroke": c, "size": 7}})
        series.append({"type": "line", "xKey": "scenario", "yKey": f"{res}_without",
                       "yName": res, "stroke": c, "strokeWidth": 2,
                       "lineDash": [6, 4], "strokeOpacity": 0.8,
                       "marker": {"enabled": False}, "showInLegend": False})
    return ({"data": data, "series": series,
             "axes": _axes("Scenario", y_title),
             "footnote": {"text": "Solid line = with agentic AI      "
                                  "Dashed line = without agentic AI",
                          "fontSize": 16, "fontWeight": "bold",
                          "color": "#000000", "spacing": 10},
             "legend": {"position": "right",
                        "item": {"paddingY": 6,
                                 "label": {"fontSize": 16, "color": "#000000"}}}},
            unit_note)


def chart_utilisation(df):
    opts, _ = _resource_series(df, "util", "Utilisation (%)", "")
    return opts


def chart_machine_waiting(df):
    opts, _ = _resource_series(df, "wait", "Average waiting time (hours)", "")
    return opts

def chart_waiting_stacked(df):
    """Total queueing time per scenario, split by resource.

    Two bars per scenario -- without and with agentic AI -- each stacked by
    resource. This carries three things the line version could not: the total
    congestion in the shop, how much of it rescheduling removes, and which
    resource it comes from. On a shared line axis ASM's range dwarfed the six
    machines and flattened them; stacked, that dominance becomes the point
    rather than an obstacle.

    `stackGroup` keeps the two conditions as separate side-by-side stacks
    instead of piling them into one column."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    resources = sorted([c[len("des_donothing_wait_"):] for c in df.columns
                        if c.startswith("des_donothing_wait_")],
                       key=lambda r: (r == core.ASM_STATION, r))

    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        acted = r["impact_decision"] == "reschedule"
        row = {"scenario": PLOT_LABELS[k]}
        for res in resources:
            dn = r.get(f"des_donothing_wait_{res}")
            rs = r.get(f"des_reschedule_wait_{res}")
            row[f"{res}_without"] = round(float(dn), 2) if pd.notna(dn) else 0
            use = rs if (acted and pd.notna(rs)) else dn
            row[f"{res}_with"] = round(float(use), 2) if pd.notna(use) else 0
        data.append(row)

    series = []
    for i, res in enumerate(resources):
        c = PALETTE[i % len(PALETTE)]
        series.append({"type": "bar", "xKey": "scenario", "yKey": f"{res}_without",
                       "yName": res, "stackGroup": "without", "fill": c,
                       "stroke": "#ffffff", "strokeWidth": 1})
        series.append({"type": "bar", "xKey": "scenario", "yKey": f"{res}_with",
                       "yName": res, "stackGroup": "with", "fill": c,
                       "stroke": "#ffffff", "strokeWidth": 1,
                       "showInLegend": False})
    return {
        "data": data, "series": series,
        "axes": _axes("Scenario", "Total queueing time, all resources (hours)"),
        "footnote": {"text": "Left bar of each pair = without agentic AI      "
                             "Right bar = with agentic AI",
                     "fontSize": 16, "fontWeight": "bold",
                     "color": "#000000", "spacing": 10},
        "legend": {"position": "right",
                   "item": {"paddingY": 6,
                            "label": {"fontSize": 16, "color": "#000000"}}},
    }


def chart_waiting_with_only(df):
    """Queueing time per resource, WITH agentic AI only.

    The paired version drew fourteen lines and the dashed without-AI series sat
    almost on top of the solid ones for every machine, so the overlap obscured
    both. Showing one condition gives a clean profile of where components queue
    once the system is running; the without-vs-with comparison is carried by the
    stacked chart above."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    resources = sorted([c[len("des_donothing_wait_"):] for c in df.columns
                        if c.startswith("des_donothing_wait_")],
                       key=lambda r: (r == core.ASM_STATION, r))
    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        acted = r["impact_decision"] == "reschedule"
        row = {"scenario": PLOT_LABELS[k]}
        for res in resources:
            dn = r.get(f"des_donothing_wait_{res}")
            rs = r.get(f"des_reschedule_wait_{res}")
            use = rs if (acted and pd.notna(rs)) else dn
            row[res] = round(float(use), 2) if pd.notna(use) else None
        data.append(row)

    series = [{"type": "line", "xKey": "scenario", "yKey": res, "yName": res,
               "stroke": PALETTE[i % len(PALETTE)], "strokeWidth": 2.5,
               "marker": {"fill": PALETTE[i % len(PALETTE)],
                          "stroke": PALETTE[i % len(PALETTE)], "size": 7}}
              for i, res in enumerate(resources)]
    return {
        "data": data, "series": series,
        "axes": _axes("Scenario", "Mean waiting time (hours)"),
        "legend": {"position": "right",
                   "item": {"paddingY": 6,
                            "label": {"fontSize": 16, "color": "#000000"}}},
    }

def chart_avg_utilisation(df):
    """Mean utilisation across every resource, one line per condition.

    Averaging is safe here precisely because the per-resource panels were
    redundant: busy time on each resource is invariant across scenarios (same
    operations, same durations), so utilisation differs only through the shared
    makespan denominator and every resource traces the same curve. The mean
    therefore represents all of them rather than blurring different behaviours.

    ASM is included, matching the shop-wide queue chart's convention. To report
    machines only, drop the `or True` guard below."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    resources = [c[len("des_donothing_util_"):] for c in df.columns
                 if c.startswith("des_donothing_util_")]
    resources = [r for r in resources if r != core.ASM_STATION or True]

    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        acted = r["impact_decision"] == "reschedule"
        dn, wi = [], []
        for res in resources:
            a = r.get(f"des_donothing_util_{res}")
            b = r.get(f"des_reschedule_util_{res}")
            if pd.notna(a):
                dn.append(float(a))
                wi.append(float(b) if (acted and pd.notna(b)) else float(a))
        data.append({"scenario": PLOT_LABELS[k],
                     "without": round(sum(dn) / len(dn), 2),
                     "with": round(sum(wi) / len(wi), 2)})
    return {
        "data": data,
        "series": [
            {"type": "line", "xKey": "scenario", "yKey": "without",
             "yName": "Without Agentic AI", "stroke": WITHOUT_COLOUR,
             "strokeWidth": 2, "lineDash": [6, 4],
             "marker": {"fill": WITHOUT_COLOUR, "stroke": WITHOUT_COLOUR, "size": 7}},
            {"type": "line", "xKey": "scenario", "yKey": "with",
             "yName": "With Agentic AI", "stroke": WITH_COLOUR, "strokeWidth": 2,
             "marker": {"fill": WITH_COLOUR, "stroke": WITH_COLOUR, "size": 7}}],
        "axes": _axes("Scenario", "Mean utilisation across all resources (%)"),
        "legend": {"position": "bottom"},
    }

def _resource_matrix(df, prefix):
    """{resource: {scenario: (without, with)}} for the per-resource metrics."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    resources = sorted([c[len(f"des_donothing_{prefix}_"):] for c in df.columns
                        if c.startswith(f"des_donothing_{prefix}_")],
                       key=lambda r: (r == core.ASM_STATION, r))
    out = {res: {} for res in resources}
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        acted = r["impact_decision"] == "reschedule"
        for res in resources:
            dn = r.get(f"des_donothing_{prefix}_{res}")
            rs = r.get(f"des_reschedule_{prefix}_{res}")
            use = rs if (acted and pd.notna(rs)) else dn
            out[res][k] = (float(dn) if pd.notna(dn) else None,
                           float(use) if pd.notna(use) else None)
    return resources, kinds, out


def chart_resource_delta(df, prefix, y_title):
    """Grouped bar of the CHANGE rescheduling makes at each resource.

    The paired-line version needed fourteen series -- seven resources times two
    conditions -- and most pairs sat on top of each other, so the one resource
    that does move was lost in the crowd. Plotting the difference instead drops
    it to one series per scenario: bars below zero are reductions, which for
    waiting time and queueing is the improvement."""
    resources, kinds, m = _resource_matrix(df, prefix)
    data = []
    for res in resources:
        row = {"resource": res}
        for k in kinds:
            dn, wi = m[res][k]
            row[k] = round(wi - dn, 2) if (dn is not None and wi is not None) else None
        data.append(row)
    return {
        "data": data,
        "series": [{"type": "bar", "xKey": "resource", "yKey": k,
                    "yName": PLOT_LABELS[k], "fill": PALETTE[i % len(PALETTE)]}
                   for i, k in enumerate(kinds)],
        "axes": _axes("Resource", f"Change in {y_title.lower()} (with AI - without AI)"),
        "legend": {"position": "bottom"},
    }


def chart_shopwide_queue(df):
    """Line: mean queue length across every resource, one point per scenario."""
    kinds = [k for k in SCENARIO_KINDS if k in set(df["scenario"])]
    data = []
    for k in kinds:
        r = df[df.scenario == k].iloc[0]
        acted = r["impact_decision"] == "reschedule"
        rs = r.get("des_reschedule_avg_qlen_shopwide")
        data.append({"scenario": PLOT_LABELS[k],
                     "without": round(float(r["des_donothing_avg_qlen_shopwide"]), 3),
                     "with": round(float(rs if (acted and pd.notna(rs))
                                         else r["des_donothing_avg_qlen_shopwide"]), 3)})
    return {
        "data": data,
        "series": [
            {"type": "line", "xKey": "scenario", "yKey": "without",
             "yName": "Without Agentic AI", "stroke": WITHOUT_COLOUR,
             "marker": {"fill": WITHOUT_COLOUR, "stroke": WITHOUT_COLOUR}},
            {"type": "line", "xKey": "scenario", "yKey": "with",
             "yName": "With Agentic AI", "stroke": WITH_COLOUR,
             "marker": {"fill": WITH_COLOUR, "stroke": WITH_COLOUR}}],
        "axes": _axes("Scenario", "Average queue length (jobs)"),
        "legend": {"position": "bottom"},
    }


def chart_event_timeline(event_df):
    """Combination: a line per scenario showing tardiness exposure as each
    disruption lands, with a scatter overlay marking the events where the agent
    actually rescheduled."""
    if event_df is None or event_df.empty:
        return None
    kinds = [k for k in SCENARIO_KINDS if k in set(event_df["scenario"])]
    rows = {}
    for i, k in enumerate(kinds):
        for _, e in event_df[event_df.scenario == k].sort_values("now_wd").iterrows():
            d = rows.setdefault(round(float(e["now_wd"]), 2), {"clock": round(float(e["now_wd"]), 2)})
            d[f"{k}_tard"] = round(float(e["donothing_tardiness_wd"]), 2)
            if e["decision"] == "reschedule":
                d[f"{k}_acted"] = round(float(e["donothing_tardiness_wd"]), 2)
    data = [rows[t] for t in sorted(rows)]

    series = []
    for i, k in enumerate(kinds):
        c = PALETTE[i % len(PALETTE)]
        series.append({"type": "line", "xKey": "clock", "yKey": f"{k}_tard",
                       "yName": PLOT_LABELS[k], "stroke": c,
                       "marker": {"fill": c, "stroke": c, "size": 7}})
        series.append({"type": "scatter", "xKey": "clock", "yKey": f"{k}_acted",
                       "yName": f"{PLOT_LABELS[k]} rescheduled",
                       "fill": "transparent", "stroke": c,
                       "marker": {"size": 16, "strokeWidth": 2},
                       "showInLegend": False})
    return {"data": data, "series": series,
            "axes": _axes("Clock at which the disruption became known (working-days)",
                          "Tardiness if no action taken (working-days)", x_type="number"),
            "legend": {"position": "bottom"}}


# ================================================================ Gantt charts ====
# AG Charts Community has no range-bar (it is an Enterprise feature), and the
# stacked-bar workaround cannot carry a text label inside each segment -- which
# is precisely what makes the dashboard's "By machine" view readable. The Gantt
# is therefore drawn with Plotly, mirroring app.py tab 2: one row per resource,
# one bar per operation, coloured by product and labelled "P{job}.{component}".
import plotly.graph_objects as go
import plotly.express as px


def gantt_figure(ops, plan, title, clock_wd=None, subtitle=None):
    """Labelled Gantt in the AG Charts layout, drawn with Plotly.

    Rows are every resource the shop uses: the shared assembly station on top,
    then the machines (each tagged with the component type it processes, since
    the mapping is one-to-one in this dataset), then one row per outsourced
    component. Outsourced legs get their own rows because they occupy no
    shop-floor capacity and would otherwise overlap illegibly on a shared row.

    Bars carry no text: with ~95 operations the in-bar labels collided badly.
    Colour identifies the product, and hovering gives the exact operation."""
    jobs = sorted({o["job"] for o in ops})
    cmap = {j: PRODUCT_COLOURS[(j - 1) % len(PRODUCT_COLOURS)] for j in jobs}

    # machine -> component type, so rows read "M1 (C1)"
    mcomp = {}
    for o in ops:
        if o["kind"] == "comp" and not o["outsourced"]:
            mcomp.setdefault(o["machine"], o["comp"])
    machines = sorted(mcomp)
    mlabel = {m: f"{m} ({mcomp[m]})" for m in machines}

    def row_of(o):
        if o["outsourced"]:
            return f"Outsourced P{o['job']} ({o['comp']})"
        return core.ASM_STATION if o["kind"] == "asm" else mlabel[o["machine"]]

    out_rows = sorted({row_of(o) for o in ops if o["outsourced"]},
                      key=lambda s: int(s.split("P")[1].split(" ")[0]))
    # Plotly draws the FIRST category at the bottom, so build bottom-up
    order = list(reversed([core.ASM_STATION] + [mlabel[m] for m in machines] + out_rows))

    fig = go.Figure()
    seen = set()
    for o in ops:
        p = plan.get(o["idx"]) or plan.get(str(o["idx"]))
        if not p:
            continue
        dur = p["end"] - p["start"]
        fig.add_trace(go.Bar(
            y=[row_of(o)], x=[dur / MPD], base=[p["start"] / MPD], orientation="h",
            marker=dict(color=cmap[o["job"]],
                        line=dict(width=1.0, color="#ffffff")),
            legendgroup=f"P{o['job']}", name=f"P{o['job']}",
            showlegend=(o["job"] not in seen),
            hovertemplate=(f"<b>Product {o['job']} &middot; {o['comp']}</b><br>"
                           f"{row_of(o)}<br>"
                           f"start {p['start']/MPD:.2f} wd &middot; "
                           f"{dur/MPD:.2f} wd ({dur/60:.1f} h)<extra></extra>")))
        seen.add(o["job"])

    if clock_wd:
        fig.add_vline(x=clock_wd, line_width=1.5, line_dash="dash", line_color="#666",
                      annotation_text="last disruption observed",
                      annotation_position="top", annotation_font_size=19)
    black = dict(size=22, color="#000000")
    fig.update_layout(
        barmode="overlay", bargap=0.3, height=48 * len(order) + 260,
        template="plotly_white",
        font=dict(size=18, color="#000000"),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=30, color="#000000")),
        xaxis=dict(title=dict(text="Working-days from release", font=black),
                   tickfont=dict(size=20, color="#000000"),
                   showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        yaxis=dict(title=dict(text="Resource", font=black),
                   tickfont=dict(size=18, color="#000000"),
                   categoryorder="array", categoryarray=order, showgrid=False),
        legend=dict(title=dict(text="Product", side="top",
                               font=dict(size=20, color="#000000")),
                    orientation="h", yanchor="top", y=-0.16,
                    xanchor="center", x=0.5, font=dict(size=16, color="#000000"),
                    itemwidth=30, tracegroupgap=4),
        margin=dict(l=10, r=10, t=100, b=200))
    return fig


def write_gantt(fig, stem):
    """Write PNG when a static export engine is available (best for the
    dissertation) and always write an interactive HTML alongside it."""
    fig.write_html(f"{stem}.html", include_plotlyjs="cdn")
    print(f"Wrote {stem}.html")
    try:
        fig.write_image(f"{stem}.png", width=1500, height=800, scale=2)
        print(f"Wrote {stem}.png")
    except Exception as exc:
        print(f"  (no PNG: {type(exc).__name__}. Direct image export needs "
              f"'pip install kaleido' AND a Chrome install -- run 'plotly_get_chrome' "
              f"if Kaleido reports it missing. The HTML above always works: open it "
              f"and screenshot, or use the camera icon in its toolbar to save a PNG.)")


# ======================================================================== main ====
def build_combined(paths, dataset, out):
    ops = core.load_ops(path=dataset)
    runs, evs, details = [], [], []
    for p in paths:
        r, e, d = load_run(p)
        runs.append(r)
        if e is not None:
            evs.append(e)
        details.extend(d)
        print(f"  loaded {p}: {list(r['scenario'])}")
    df = pd.concat(runs, ignore_index=True, sort=False)
    event_df = pd.concat(evs, ignore_index=True, sort=False) if evs else None

    blocks = [
        ("Total tardiness across scenarios",
         "Working-days past the promised delivery date, summed over all 20 products, "
         "at the end of each scenario's run.",
         chart_tardiness(df)),
        ("Late products across scenarios",
         "How many of the 20 products miss their date -- the count behind the "
         "tardiness magnitude above.",
         chart_late_jobs(df)),
        ("Shop-wide queue length",
         "Average queue length across all machines and the assembly station -- "
         "a single congestion indicator.",
         chart_shopwide_queue(df)),
        ]
    blocks.append((
        "Average resource utilisation",
        "Mean busy-time share across all machines and the assembly station. A ",
        chart_avg_utilisation(df)))
    blocks.append((
        "Total queueing time by resource",
        "",
        chart_waiting_stacked(df)))
    blocks.append((
        "Component waiting time at each machine (with agentic AI)",
        "",
        chart_waiting_with_only(df)))

    html = _page("ADRS results across scenarios", blocks)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({len(blocks)} charts) -- open it in a browser")


def build_scenario(path, dataset, out):
    ops = core.load_ops(path=dataset)
    run_df, event_df, details = load_run(path)
    if not details:
        raise SystemExit(f"{_details_path(path)} not found -- the Gantt charts need it")
    d = details[0]
    kind = d["scenario"]
    lab_dn, lab_rs = _condition_labels(kind)
    if "plan_without" not in d:
        raise SystemExit("This result file predates plan export. Re-run the scenario "
                         "after adding plan_without/plan_with to summarise_run.")
    r = run_df.iloc[0]
    stem = os.path.splitext(out)[0]
    clock = float(r["now_wd"]) if pd.notna(r.get("now_wd")) else None

    dn_t = float(r["donothing_tardiness_wd"])
    rs_t = (float(r["reschedule_tardiness_wd"])
            if pd.notna(r["reschedule_tardiness_wd"]) else dn_t)
    if kind == "baseline":
        # No disruption and no agent, so a without/with pair would be comparing
        # scheduling METHODS rather than rescheduling policies -- a different
        # question from every other scenario, and one a shared axis label would
        # misrepresent. Only the committed optimised schedule is shown.
        write_gantt(gantt_figure(ops, d["plan_with"], PLOT_LABELS[kind], clock),
                    f"{stem}_gantt")
    else:
        write_gantt(gantt_figure(ops, d["plan_without"],
                                 f"{PLOT_LABELS[kind]} - {lab_dn}", clock),
                    f"{stem}_gantt_without")
        write_gantt(gantt_figure(ops, d["plan_with"],
                                 f"{PLOT_LABELS[kind]} - {lab_rs}", clock),
                    f"{stem}_gantt_with")

    tl = chart_event_timeline(event_df)
    if tl:
        html = _page(f"ADRS results - {PLOT_LABELS[kind]}",
                     [("Disruption timeline",
                       "Tardiness exposure as each disruption arrives. Ringed markers "
                       "are events where the agent chose to reschedule.", tl)])
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote {out} (event timeline)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="job_shop_20products_tight.xlsx")
    p.add_argument("--scenario", metavar="CSV",
                   help="one scenario's result CSV -> Gantt charts + its event timeline")
    p.add_argument("--combine", nargs="+", metavar="CSV",
                   help="several scenario CSVs -> the cross-scenario charts")
    p.add_argument("--out", default="adrs_charts.html")
    a = p.parse_args()
    if a.combine:
        build_combined(a.combine, a.dataset, a.out)
    elif a.scenario:
        build_scenario(a.scenario, a.dataset, a.out)
    else:
        p.error("give either --scenario or --combine")


if __name__ == "__main__":
    main()