"""
ADRS experiment runner -- SEQUENTIAL (rolling-horizon) disruption experiments.

Design
------
Each scenario is ONE run of the factory, not a set of independent replications.
Within that run, a number of DISRUPTION EVENTS arrive at increasing clock times.
Each event is handled in turn by the full agent pipeline (detection -> impact ->
[objectives -> options_advisor -> human_pick] -> explainability), and if the
agent decides to reschedule, the chosen plan BECOMES the committed plan that the
next event is measured against. State therefore accumulates across the run,
exactly as it would on a real shop floor:

    event 1 @ 3.0wd   P10 C5 assumed 6wd -> actual 10wd   -> reschedule
    event 2 @ 4.5wd   P8  C3 assumed 7wd -> actual  9wd   -> reschedule
    event 3 @ 7.0wd   P2  C1 assumed 4wd -> actual  8wd   -> no_action
    ...

Every actual return, once observed, stays known for all later solves.

Scenarios
---------
The scenario defines the SHAPE of a single event; the run repeats that shape
`--disruptions` times.

    baseline   no disruption at all (nominal plan, for reference)
    late_1     Scenario 1 -- one outsourced component delayed
    late_2     Scenario 2 -- two outsourced components delayed together
    early_1    Scenario 3 -- one outsourced component returns early
    early_2    Scenario 4 -- two outsourced components return early together
    mixed      Scenario 5 -- one early and one late in the same event

Event timing obeys causality (see event_clock): a LATE return becomes knowable
once its originally-assumed return day passes without the part arriving; an
EARLY return only becomes knowable when it actually arrives. Event times are
therefore DERIVED from the disruption, not invented, and events are processed in
ascending clock order. The clock is also forced to be non-decreasing.

Outputs
-------
    <out>                       one row per SCENARIO (end-of-run comparison).
                                This is what every figure consumes.
    <out stem>_events.csv       one row per DISRUPTION EVENT -- the audit trail
                                of what the agent saw and decided, step by step.
    <out stem>_details.json     per-scenario job/machine breakdowns for the
                                heatmap figures.

Headline comparison, per scenario:
    WITHOUT agentic AI = the ORIGINAL committed plan, carried through every
                         disruption with no rescheduling ever (right-shift).
    WITH    agentic AI = the FINAL committed plan, after each accepted
                         reschedule was applied in sequence.
Both are evaluated against the same full set of observed actual returns, and
cross-checked by an independent SimPy discrete-event simulation.

Requires a working GROQ_API_KEY (.env) -- this exercises the real LLM agents.

    python adrs_experiments.py --scenario late_1 --disruptions 3 --plot
    python adrs_experiments.py --all-scenarios --plot
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langgraph.types import Command

import adrs_core as core
import adrs_agents
import adrs_simulation as sim
import json

MPD = core.MIN_PER_DAY

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(SCRIPT_DIR, "job_shop_20products_tight.xlsx")

SCENARIOS = [
    ("baseline", "Baseline (no outsourcing disruption)"),
    ("late_1",   "Scenario 1 - delay, one outsourced component"),
    ("late_2",   "Scenario 2 - delay, two outsourced components"),
    ("early_1",  "Scenario 3 - early, one outsourced component"),
    ("early_2",  "Scenario 4 - early, two outsourced components"),
    ("mixed",    "Scenario 5 - one early + one late"),
]
SCENARIO_KINDS = [k for k, _ in SCENARIOS]

# Short axis labels used on every figure (full descriptions stay in the console
# output and CSV only -- no description after the number).
PLOT_LABELS = {
    "baseline": "Baseline",
    "late_1":   "Scenario 1",
    "late_2":   "Scenario 2",
    "early_1":  "Scenario 3",
    "early_2":  "Scenario 4",
    "mixed":    "Scenario 5",
}

# How many components deviate in ONE event of each scenario, and in which
# direction. ("late",) means a single late component; ("early", "late") means
# one of each in the same event.
EVENT_SHAPE = {
    "baseline": (),
    "late_1":   ("late",),
    "late_2":   ("late", "late"),
    "early_1":  ("early",),
    "early_2":  ("early", "early"),
    "mixed":    ("early", "late"),
}

RESCHEDULE_OBJECTIVES = ("tardiness", "makespan", "stability", "balanced")

DELAY_MIN_DAYS, DELAY_MAX_DAYS = 3, 10   # a LATE return is this many days late
EARLY_MIN_SLACK_DAYS = 2                 # an EARLY return needs this much assumed lead


# ============================================================ unit conversion ====
def leads_from_return_days(committed, return_days):
    """{op_idx: absolute return DAY} -> {op_idx: actual LEAD (working-minutes)},
    measured against the CURRENT committed plan.

    A component disrupted at a later event was not frozen during earlier
    reschedules, so its dispatch may have moved. If it moved past the return day
    already observed, the lead would go negative -- a part arriving before it was
    sent. That is physically impossible and would corrupt every downstream
    measure, so it is clamped and reported."""
    out = {}
    for idx, rd in return_days.items():
        lead = int(round(rd * MPD)) - committed[idx]["start"]
        if lead < 0:
            print(f"  WARNING: op {idx} return day {rd:.1f}wd precedes its dispatch at "
                  f"{committed[idx]['start'] / MPD:.1f}wd -- an earlier reschedule moved "
                  f"dispatch past the observed return. Clamped to 0.")
            lead = 0
        out[idx] = lead
    return out


def event_clock(ops, committed, return_days):
    """Working-minute at which this event's deviation becomes knowable.

    - LATE: knowable once the originally-assumed return day passes without the
      part showing up -- you do not have to wait for the late arrival itself.
    - EARLY (or on time): only knowable once it has actually arrived; you cannot
      react to news before it occurs.
    An event with several components is knowable when the LAST of them is."""
    if not return_days:
        return 0
    times = []
    for idx, rd in return_days.items():
        dispatch = committed[idx]["start"]
        assumed_return = dispatch + ops[idx]["dur"]
        actual_return = int(round(rd * MPD))
        times.append(assumed_return if actual_return > assumed_return else actual_return)
    return max(times)


# ============================================================== event sampling ====
def _eligible(ops, direction):
    """Outsourced ops that can plausibly deviate in `direction`. An EARLY return
    needs enough assumed lead time that 'earlier' is meaningfully different from
    a same-day turnaround."""
    if direction == "early":
        return [o for o in ops if o["outsourced"] and o["dur"] >= EARLY_MIN_SLACK_DAYS * MPD]
    return [o for o in ops if o["outsourced"]]


def sample_event(ops, committed, rng, kind, used_ops):
    """Sample ONE disruption event of the shape defined by `kind`.

    Returns (return_days, parts) where return_days maps op index -> absolute
    return DAY, and parts is a list of human-readable strings describing each
    deviation. `used_ops` is mutated so no component is disrupted twice in the
    same run -- a component's actual return is a one-off physical fact, and
    re-disrupting it mid-run would overwrite information the agent already
    holds."""
    shape = EVENT_SHAPE[kind]
    if not shape:
        return {}, []

    return_days, parts, by_job = {}, [], {}
    for direction in shape:
        pool = [o for o in _eligible(ops, direction) if o["idx"] not in used_ops]
        if not pool:
            print(f"  (note: no unused outsourced component left for a '{direction}' "
                  f"deviation -- event truncated)")
            break
        o = rng.choice(pool)
        used_ops.add(o["idx"])
        dispatch = committed[o["idx"]]["start"]
        assumed_days = o["dur"] / MPD

        if direction == "late":
            delay_days = rng.randint(DELAY_MIN_DAYS, DELAY_MAX_DAYS)
            rd = (dispatch + o["dur"]) / MPD + delay_days
            parts.append(f"P{o['job']} {o['comp']} LATE by {delay_days}wd "
                         f"(assumed {assumed_days:.0f}wd, returns day {rd:.1f})")
            by_job[o["job"]] = f"{o['comp']} late +{delay_days}wd"
        else:
            # anywhere from a same-day turnaround up to one day short of assumed
            actual_days = rng.randint(0, max(1, int(assumed_days) - 1))
            rd = dispatch / MPD + actual_days
            parts.append(f"P{o['job']} {o['comp']} EARLY, takes {actual_days}wd "
                         f"(assumed {assumed_days:.0f}wd, returns day {rd:.1f})")
            by_job[o["job"]] = f"{o['comp']} early -{int(assumed_days) - actual_days}wd"
        return_days[o["idx"]] = rd
    return return_days, parts, by_job


def plan_events(ops, committed, rng, kind, n_events):
    """Build the whole run's event list up front, ordered by the clock time at
    which each becomes knowable.

    Times are computed against the ORIGINAL committed plan purely to establish
    an ORDER; each event's clock is recomputed against the live plan when it is
    actually processed, since an earlier reschedule may have moved dispatch
    times. Sampling components up front (without replacement) also guarantees no
    component is disrupted twice in one run."""
    used_ops, events = set(), []
    for _ in range(n_events):
        rds, parts, by_job = sample_event(ops, committed, rng, kind, used_ops)
        if not rds:
            break
        events.append(dict(return_days=rds, parts=parts, by_job=by_job,
                           provisional_clock=event_clock(ops, committed, rds)))
    events.sort(key=lambda ev: ev["provisional_clock"])
    return events


def build_naive_schedule(ops):
    """EDD list schedule -- the shop-floor default the optimiser is compared to
    in the baseline scenario. Jobs are taken in earliest-due-date order and each
    operation is placed at the first moment its machine is free and its
    predecessors are done. No lookahead and no re-sequencing: a classic
    dispatching rule, standing in for how the shop would plan without CP-SAT."""
    by_job = {}
    for o in ops:
        by_job.setdefault(o["job"], []).append(o)
    order = sorted(by_job, key=lambda j: (by_job[j][0]["due"], j))
    free, plan = {}, {}
    for j in order:
        jops = by_job[j]
        comps = [o for o in jops if o["kind"] == "comp"]
        asm = next(o for o in jops if o["kind"] == "asm")
        ready = 0
        for o in comps:
            if o["outsourced"]:              # no shared capacity; runs from release
                s = o["release"]
            else:
                m = o["machine"]
                s = max(o["release"], free.get(m, 0))
                free[m] = s + o["dur"]
            plan[o["idx"]] = {"start": s, "end": s + o["dur"]}
            ready = max(ready, s + o["dur"])
        m = core.ASM_STATION
        s = max(ready, free.get(m, 0), asm["release"])
        free[m] = s + asm["dur"]
        plan[asm["idx"]] = {"start": s, "end": s + asm["dur"]}
    return plan

# ============================================================ detail helpers ======
def _job_completions(ops, plan):
    """{job: completion_minutes} -- when that job's assembly finishes."""
    return {o["job"]: plan[o["idx"]]["end"] for o in ops if o["kind"] == "asm"}


def _job_wait_times(ops, events):
    """{job: total_wait_minutes} across that job's own in-house component
    operations (queueing at a machine before processing; excludes ASM waiting
    and outsourced legs, which never queue for shared capacity)."""
    out = {j: 0.0 for j in sorted({o["job"] for o in ops})}
    for e in events:
        if e["kind"] == "comp" and not e.get("outsourced"):
            out[e["job"]] += e["wait"]
    return out


def _job_machine_wait_times(ops, events):
    """{job: {machine: wait_minutes}} -- per-product, per-machine waiting time,
    unsummed so a change on one machine isn't diluted by unaffected ones."""
    machines = sorted({o["machine"] for o in ops if o["kind"] == "comp" and not o["outsourced"]})
    out = {j: {m: 0.0 for m in machines} for j in sorted({o["job"] for o in ops})}
    for e in events:
        if e["kind"] == "comp" and not e.get("outsourced"):
            out[e["job"]][e["resource"]] += e["wait"]
    return out


def _job_machine_leadtime_times(ops, events):
    """{job: {machine: production_time}} -- (end - arrival), i.e. waiting plus
    processing, per product per machine, unsummed in both directions."""
    machines = sorted({o["machine"] for o in ops if o["kind"] == "comp" and not o["outsourced"]})
    out = {j: {m: 0.0 for m in machines} for j in sorted({o["job"] for o in ops})}
    for e in events:
        if e["kind"] == "comp" and not e.get("outsourced"):
            out[e["job"]][e["resource"]] += (e["end"] - e["arrival"])
    return out


def _component_leads(ops, events):
    """{component_label: avg_lead_minutes} across every operation of that type."""
    comp_labels = sorted({o["comp"] for o in ops if o["kind"] == "comp"}, key=lambda c: (len(c), c))
    out = {}
    for comp in comp_labels:
        evs = [e for e in events if e["kind"] == "comp" and e["comp"] == comp]
        out[comp] = (sum(e["end"] - e["arrival"] for e in evs) / len(evs)) if evs else 0
    return out


def _collateral(tardiness_by_job, dis_jobs):
    """{job: tardiness} for jobs OTHER than the directly-disrupted ones -- i.e.
    lateness caused by contention at the shared ASM station, not by a job's own
    outsourced part."""
    return {j: t for j, t in tardiness_by_job.items() if t > 0 and j not in dis_jobs}


def _shopwide_avg_queue(des_result):
    """Mean avg_queue_length across every resource -- one shop-wide congestion
    number, not tied to any single machine."""
    return float(np.mean([r["avg_queue_length"] for r in des_result["resources"].values()]))


def _machine_start_times_changed(ops, plan, reference):
    """COUNT of in-house MACHINE operations whose start time differs from the
    reference plan. Not a resequencing count -- an operation that keeps its
    queue position but runs later still counts."""
    return sum(1 for o in ops
               if o["kind"] == "comp" and not o["outsourced"]
               and plan[o["idx"]]["start"] != reference[o["idx"]]["start"])
    
def _realised_plan(des_events):
    """{op_idx: {start, end}} as ACTUALLY realised by the discrete-event
    simulation, including queueing. The committed plan is what was intended;
    this is what happened, and it is what every tardiness figure is measured
    from -- so it is what the Gantt charts must show."""
    return {e["idx"]: {"start": e["arrival"] + e["wait"], "end": e["end"]}
            for e in des_events}


# ====================================================== one disruption event ======
def run_agent_pipeline(app, ops, committed, ar, now, thread_id, pick_objective=None):
    """Invoke the graph end to end for ONE event. Resumes past the human_pick
    interrupt with `pick_objective` when given, else with the agent's own
    recommendation, so a whole run can be held to a single objective."""
    cfg = {"configurable": {"thread_id": str(thread_id)}}
    init = {"ops": ops, "committed_plan": committed, "now": now, "actual_returns": ar}
    final = app.invoke(init, cfg)
    if final.get("__interrupt__"):
        choice = pick_objective or final["advice"]["recommended"]
        if not any(o["objective"] == choice for o in final.get("options", [])):
            choice = final["advice"]["recommended"]
        final = app.invoke(Command(resume=choice), cfg)
    return final


def process_event(app, ops, committed, known_return_days, event, clock_floor,
                  thread_id, pick_objective=None):
    """Handle ONE disruption event against the CURRENT committed plan.

    `known_return_days` already contains every return day observed earlier in
    the run and is updated in place with this event's. Returns
    (event_row, new_committed) -- new_committed is the rescheduled plan if the
    agent acted, otherwise the plan passed in, unchanged."""
    known_return_days.update(event["return_days"])
    ar = leads_from_return_days(committed, known_return_days)

    # recompute against the LIVE plan, and never let the clock go backwards
    now = max(clock_floor, event_clock(ops, committed, event["return_days"]))

    dn = core.evaluate_donothing(ops, committed, ar, now=now)
    dis_jobs = {ops[i]["job"] for i in ar}
    collateral_dn = _collateral(dn["tardiness"], dis_jobs)

    row = dict(
        now_wd=now / MPD,
        disruption=" ; ".join(event["parts"]),
        n_components=len(event["return_days"]),
        cumulative_components=len(known_return_days),
        donothing_tardiness_wd=dn["total_tardiness"] / MPD,
        collateral_count=len(collateral_dn),
    )

    final = run_agent_pipeline(app, ops, committed, ar, now, thread_id,
                               pick_objective=pick_objective)
    row["severity"] = final["disruption"]["severity"]
    row["detection_reasoning"] = final["disruption"]["reasoning"]
    row["decision"] = final["decision"]
    row["impact_reasoning"] = final["impact_reasoning"]
    row["achievable_gain_wd"] = final.get("achievable_gain", 0) / MPD
    row["recommended_objective"] = final.get("advice", {}).get("recommended")
    row["explanation"] = final["explanation"]

    if final["decision"] == "reschedule":
        chosen = final.get("chosen", {})
        rs = final["reschedule"]
        # impact() measures the achievable gain on the BEST of the four option
        # plans, but the plan committed here is whichever objective this run is
        # held to. When another objective scores better -- common on instances
        # that return FEASIBLE rather than OPTIMAL -- the guard can pass on a
        # plan that is never applied while a worse one is committed. Re-check
        # the plan actually being committed, and refuse it if it loses to doing
        # nothing.
        if rs["total_tardiness"] >= dn["total_tardiness"]:
            worse = (rs["total_tardiness"] - dn["total_tardiness"]) / MPD
            print(f"      REJECTED: committed objective '{chosen.get('objective')}' "
                  f"gives a plan {worse:.2f}wd worse than doing nothing -- "
                  f"keeping the current plan")
            row["decision"] = "no_action"
            row["impact_reasoning"] = (
                f"Rescheduling under the committed objective would have been "
                f"{worse:.2f} working-days worse than the current plan, so it was "
                f"not applied.")
            row["objective"] = None
            row["objective_label"] = None
            row["solver_status"] = chosen.get("status")
            row["reschedule_tardiness_wd"] = None
            row["tardiness_reduction_wd"] = 0.0
            row["start_times_changed"] = 0
            row["machine_start_times_changed"] = 0
            row["_tard_before"] = dn["tardiness"]
            row["_tard_after"] = dn["tardiness"]
            row["_by_job"] = event.get("by_job", {})
            return row, committed
        row["objective"] = chosen.get("objective")
        row["objective_label"] = chosen.get("label")
        row["solver_status"] = chosen.get("status")
        row["reschedule_tardiness_wd"] = rs["total_tardiness"] / MPD
        row["tardiness_reduction_wd"] = row["donothing_tardiness_wd"] - row["reschedule_tardiness_wd"]
        row["start_times_changed"] = chosen.get("start_times_changed")
        row["machine_start_times_changed"] = _machine_start_times_changed(
            ops, rs["plan"], committed)
        # underscore keys are per-product dicts consumed by the product table;
        # write_single_csv strips them so they never reach the CSV
        row["_tard_before"] = dn["tardiness"]
        row["_tard_after"] = core.plan_tardiness(ops, rs["plan"])
        row["_by_job"] = event.get("by_job", {})
        return row, rs["plan"]

    row["objective"] = None
    row["objective_label"] = None
    row["solver_status"] = None
    row["reschedule_tardiness_wd"] = None
    row["tardiness_reduction_wd"] = 0.0
    row["start_times_changed"] = 0
    row["machine_start_times_changed"] = 0
    row["_tard_before"] = dn["tardiness"]
    row["_tard_after"] = dn["tardiness"]      # nothing changed
    row["_by_job"] = event.get("by_job", {})
    return row, committed


# ============================================================= one scenario run ===
def run_scenario_run(app, ops, base_committed, kind, n_disruptions, rng,
                     thread_id_start=0, pick_objective="tardiness", nominal_des=None):
    """Run ONE scenario end to end as a sequence of disruption events.

    This is the function to call to run an experiment programmatically:

        run_scenario_run(app, ops, committed, "late_1", n_disruptions=3, rng=rng)

    `kind`            which scenario (see SCENARIOS / EVENT_SHAPE)
    `n_disruptions`   how many events arrive during the run
    `pick_objective`  which of the four objective plans is committed whenever
                      the agent decides to reschedule; None follows the agent's
                      own recommendation each time

    Returns (summary_row, event_rows, detail) where summary_row is the
    end-of-run WITHOUT-vs-WITH comparison consumed by the figures."""
    label = dict(SCENARIOS)[kind]
    if kind == "baseline":
        # No disruption means nothing to reschedule, so the usual without/with
        # comparison would be two copies of the same plan. Instead the baseline
        # isolates the OPTIMISER's contribution: an EDD list schedule (what the
        # shop would produce unaided) against the CP-SAT plan. Note this is a
        # different comparison from Scenarios 1-5, which hold the optimiser
        # fixed and vary only whether the agent reschedules.
        return summarise_baseline(ops, build_naive_schedule(ops), base_committed, label)
    events = plan_events(ops, base_committed, rng, kind, n_disruptions)
    print(f"\n=== {label}: {len(events)} disruption event(s) planned ===")

    committed = base_committed
    known_return_days, event_rows = {}, []
    clock_floor, thread_id = 0, thread_id_start

    for n, event in enumerate(events, start=1):
        thread_id += 1
        print(f"  [event {n}/{len(events)}] {' ; '.join(event['parts'])}")
        er, committed = process_event(app, ops, committed, known_return_days, event,
                                      clock_floor, thread_id, pick_objective)
        clock_floor = int(er["now_wd"] * MPD)
        er.update(scenario=kind, label=label, event=n)
        event_rows.append(er)
        gain = er["tardiness_reduction_wd"]
        print(f"      clock {er['now_wd']:.1f}wd | do-nothing {er['donothing_tardiness_wd']:.2f}wd "
              f"| {er['decision']}" + (f" -> {er['reschedule_tardiness_wd']:.2f}wd "
              f"(saved {gain:.2f}wd, {er['machine_start_times_changed']} machine start times changed)"
              if er["decision"] == "reschedule" else ""))

    if nominal_des is None:                     # nominal = no disruption at all
        nominal_des = sim.simulate(ops, base_committed, {})
    summary, detail, products = summarise_run(ops, base_committed, committed,
                                              known_return_days, event_rows, kind,
                                              label, clock_floor, nominal_des)
    return summary, event_rows, detail, products

def summarise_baseline(ops, naive_plan, optimised_plan, label):
    """Baseline: unoptimised EDD list schedule vs the CP-SAT plan, no
    disruptions. Populates the same columns as summarise_run so the figures
    need no special-casing -- 'without' is the naive schedule, 'with' is the
    optimised one."""
    des_n = sim.simulate(ops, naive_plan, {})
    des_o = sim.simulate(ops, optimised_plan, {})
    tard_n = core.plan_tardiness(ops, naive_plan)
    tard_o = core.plan_tardiness(ops, optimised_plan)

    row = dict(scenario="baseline", label=label,
               description="No disruption -- EDD list schedule vs CP-SAT optimised plan",
               n_events=0, n_reschedules=0, n_disrupted=0, now_wd=0,
               replication=1, is_demo=False,
               objective="tardiness", objective_label="Fewest late deliveries",
               impact_decision="reschedule", severity=None,
               donothing_tardiness_wd=sum(tard_n.values()) / MPD,
               des_donothing_tardiness_wd=des_n["overall"]["total_tardiness"] / MPD,
               reschedule_tardiness_wd=sum(tard_o.values()) / MPD,
               des_reschedule_tardiness_wd=des_o["overall"]["total_tardiness"] / MPD)
    row["tardiness_reduction_wd"] = row["donothing_tardiness_wd"] - row["reschedule_tardiness_wd"]
    for tag, des in (("donothing", des_n), ("reschedule", des_o)):
        for res, info in des["resources"].items():
            row[f"des_{tag}_util_{res}"] = info["utilisation_pct"]
            row[f"des_{tag}_wait_{res}"] = info["avg_wait"] / 60.0
        row[f"des_{tag}_avg_wait_wd"] = des["overall"]["avg_waiting_time"] / MPD
        row[f"des_{tag}_avg_qlen_shopwide"] = _shopwide_avg_queue(des)
    row["late_jobs_donothing"] = des_n["overall"]["jobs_total"] - des_n["overall"]["jobs_on_time"]
    row["late_jobs_reschedule"] = des_o["overall"]["jobs_total"] - des_o["overall"]["jobs_on_time"]
    row["collateral_count_donothing"] = sum(1 for v in tard_n.values() if v > 0)
    row["collateral_count_reschedule"] = sum(1 for v in tard_o.values() if v > 0)
    row["collateral_tardiness_donothing_wd"] = row["donothing_tardiness_wd"]
    row["collateral_tardiness_reschedule_wd"] = row["reschedule_tardiness_wd"]
    row["machine_start_times_changed"] = _machine_start_times_changed(
        ops, optimised_plan, naive_plan)
    row["start_times_changed"] = sum(
        1 for o in ops if optimised_plan[o["idx"]]["start"] != naive_plan[o["idx"]]["start"])

    detail = dict(scenario="baseline", label=label, replication=1, is_demo=False,
                  description=row["description"], n_events=0,
                  donothing_tardiness_wd=row["donothing_tardiness_wd"],
                  objective=row["objective"], objective_label=row["objective_label"],
                  machine_start_times_changed=row["machine_start_times_changed"],
                  job_tardiness_donothing=tard_n, job_tardiness_reschedule=tard_o,
                  job_completion_donothing=_job_completions(ops, naive_plan),
                  job_completion_reschedule=_job_completions(ops, optimised_plan),
                  job_wait_donothing=_job_wait_times(ops, des_n["events"]),
                  job_wait_reschedule=_job_wait_times(ops, des_o["events"]),
                  job_machine_wait_donothing=_job_machine_wait_times(ops, des_n["events"]),
                  job_machine_wait_reschedule=_job_machine_wait_times(ops, des_o["events"]),
                  job_machine_leadtime_donothing=_job_machine_leadtime_times(ops, des_n["events"]),
                  job_machine_leadtime_reschedule=_job_machine_leadtime_times(ops, des_o["events"]),
                  component_lead_donothing=_component_leads(ops, des_n["events"]),
                  component_lead_reschedule=_component_leads(ops, des_o["events"]))

    detail["plan_without"] = {str(i): v for i, v in _realised_plan(des_n["events"]).items()}
    detail["plan_with"] = {str(i): v for i, v in _realised_plan(des_o["events"]).items()}

    release = {o["job"]: o["release"] for o in ops if o["kind"] == "asm"}
    due = {o["job"]: o["due"] for o in ops if o["kind"] == "asm"}
    wn = _job_wait_times(ops, des_n["events"])
    wo = _job_wait_times(ops, des_o["events"])
    products = []
    for j in sorted({o["job"] for o in ops}):
        kn, ko = des_n["jobs"][j], des_o["jobs"][j]
        products.append(dict(
            scenario="baseline", label=label, product=j, due_wd=due[j] / MPD,
            own_part_disrupted=False, own_part_disrupted_at_events="",
            without_completion_wd=kn["completion"] / MPD,
            without_production_time_hrs=(kn["completion"] - release[j]) / 60.0,
            without_tardiness_wd=kn["tardiness"] / MPD, without_on_time=kn["on_time"],
            without_machine_wait_hrs=wn[j] / 60.0,
            final_completion_wd=ko["completion"] / MPD,
            final_production_time_hrs=(ko["completion"] - release[j]) / 60.0,
            final_tardiness_wd=ko["tardiness"] / MPD, final_on_time=ko["on_time"],
            final_machine_wait_hrs=wo[j] / 60.0,
            tardiness_saved_wd=(kn["tardiness"] - ko["tardiness"]) / MPD,
            production_time_saved_hrs=(kn["completion"] - ko["completion"]) / 60.0))

    print(f"  --- baseline: EDD list schedule {row['donothing_tardiness_wd']:.2f}wd -> "
          f"CP-SAT {row['reschedule_tardiness_wd']:.2f}wd | late jobs "
          f"{row['late_jobs_donothing']} -> {row['late_jobs_reschedule']}")
    return row, [], detail, products

def summarise_run(ops, base_committed, final_committed, known_return_days,
                  event_rows, kind, label, clock_floor, nominal_des):
    """End-of-run comparison, evaluated against the FULL set of observed returns.

    WITHOUT agentic AI: the original committed plan, never rescheduled.
    WITH    agentic AI: the plan actually in force after the run's reschedules.
    Each is measured deterministically (right-shift) and independently
    re-derived by the SimPy DES, which simulates real resource contention."""
    ar_base = leads_from_return_days(base_committed, known_return_days)
    ar_final = leads_from_return_days(final_committed, known_return_days)
    dis_jobs = {ops[i]["job"] for i in ar_base}
    n_resched = sum(1 for r in event_rows if r["decision"] == "reschedule")

    row = dict(
        scenario=kind, label=label,
        description=" | ".join(f"@{r['now_wd']:.1f}wd {r['disruption']}" for r in event_rows)
                    or "No disruption",
        n_events=len(event_rows), n_reschedules=n_resched,
        n_disrupted=len(known_return_days),
        now_wd=clock_floor / MPD,
        # kept so the figures' filters and labels keep working unchanged
        replication=1, is_demo=False,
        objective=next((r["objective"] for r in reversed(event_rows) if r["objective"]), None),
        objective_label=next((r["objective_label"] for r in reversed(event_rows)
                              if r["objective_label"]), None),
        impact_decision="reschedule" if n_resched else "no_action",
        severity=(max((r["severity"] for r in event_rows), key=lambda s:
                      ["minor", "moderate", "severe"].index(s)) if event_rows else None),
    )

    # ---- WITHOUT agentic AI: original plan carried through every disruption ----
    dn = core.evaluate_donothing(ops, base_committed, ar_base, now=clock_floor)
    des_dn = sim.simulate(ops, base_committed, ar_base)
    row["donothing_tardiness_wd"] = dn["total_tardiness"] / MPD
    row["des_donothing_tardiness_wd"] = des_dn["overall"]["total_tardiness"] / MPD
    for res, info in des_dn["resources"].items():
        row[f"des_donothing_util_{res}"] = info["utilisation_pct"]
        row[f"des_donothing_wait_{res}"] = info["avg_wait"] / 60.0     # hours, fig6
    row["des_donothing_avg_wait_wd"] = des_dn["overall"]["avg_waiting_time"] / MPD
    row["des_donothing_avg_qlen_shopwide"] = _shopwide_avg_queue(des_dn)
    row["late_jobs_donothing"] = des_dn["overall"]["jobs_total"] - des_dn["overall"]["jobs_on_time"]
    coll_dn = _collateral(dn["tardiness"], dis_jobs)
    row["collateral_count_donothing"] = len(coll_dn)
    row["collateral_tardiness_donothing_wd"] = sum(coll_dn.values()) / MPD

    detail = dict(scenario=kind, label=label, replication=1, is_demo=False,
                  description=row["description"],
                  donothing_tardiness_wd=row["donothing_tardiness_wd"],
                  objective=row["objective"], objective_label=row["objective_label"],
                  job_tardiness_donothing=dn["tardiness"],
                  job_completion_donothing=dn["completion"],
                  job_wait_donothing=_job_wait_times(ops, des_dn["events"]),
                  job_machine_wait_donothing=_job_machine_wait_times(ops, des_dn["events"]),
                  job_machine_leadtime_donothing=_job_machine_leadtime_times(ops, des_dn["events"]),
                  component_lead_donothing=_component_leads(ops, des_dn["events"]))

    # ---- WITH agentic AI: the plan actually in force at the end of the run ----
    if n_resched:
        rs_tard = core.plan_tardiness(ops, final_committed)
        des_rs = sim.simulate(ops, final_committed, ar_final)
        row["reschedule_tardiness_wd"] = sum(rs_tard.values()) / MPD
        row["des_reschedule_tardiness_wd"] = des_rs["overall"]["total_tardiness"] / MPD
        for res, info in des_rs["resources"].items():
            row[f"des_reschedule_util_{res}"] = info["utilisation_pct"]
            row[f"des_reschedule_wait_{res}"] = info["avg_wait"] / 60.0
        row["des_reschedule_avg_wait_wd"] = des_rs["overall"]["avg_waiting_time"] / MPD
        row["des_reschedule_avg_qlen_shopwide"] = _shopwide_avg_queue(des_rs)
        row["late_jobs_reschedule"] = des_rs["overall"]["jobs_total"] - des_rs["overall"]["jobs_on_time"]
        row["tardiness_reduction_wd"] = row["donothing_tardiness_wd"] - row["reschedule_tardiness_wd"]
        coll_rs = _collateral(rs_tard, dis_jobs)
        row["collateral_count_reschedule"] = len(coll_rs)
        row["collateral_tardiness_reschedule_wd"] = sum(coll_rs.values()) / MPD
        row["machine_start_times_changed"] = _machine_start_times_changed(
            ops, final_committed, base_committed)
        row["start_times_changed"] = sum(
            1 for o in ops
            if final_committed[o["idx"]]["start"] != base_committed[o["idx"]]["start"])
        detail.update(job_tardiness_reschedule=rs_tard,
                      job_completion_reschedule=_job_completions(ops, final_committed),
                      job_wait_reschedule=_job_wait_times(ops, des_rs["events"]),
                      job_machine_wait_reschedule=_job_machine_wait_times(ops, des_rs["events"]),
                      job_machine_leadtime_reschedule=_job_machine_leadtime_times(ops, des_rs["events"]),
                      component_lead_reschedule=_component_leads(ops, des_rs["events"]),
                      machine_start_times_changed=row["machine_start_times_changed"])
    else:
        # nothing was rescheduled: the WITH side is identical to the WITHOUT side
        row["reschedule_tardiness_wd"] = None
        row["tardiness_reduction_wd"] = 0.0
        row["des_reschedule_avg_qlen_shopwide"] = row["des_donothing_avg_qlen_shopwide"]
        row["late_jobs_reschedule"] = row["late_jobs_donothing"]
        row["collateral_count_reschedule"] = row["collateral_count_donothing"]
        row["collateral_tardiness_reschedule_wd"] = row["collateral_tardiness_donothing_wd"]
        row["machine_start_times_changed"] = 0
        row["start_times_changed"] = 0
        detail.update(job_tardiness_reschedule=detail["job_tardiness_donothing"],
                      job_completion_reschedule=detail["job_completion_donothing"],
                      job_wait_reschedule=detail["job_wait_donothing"],
                      job_machine_wait_reschedule=detail["job_machine_wait_donothing"],
                      job_machine_leadtime_reschedule=detail["job_machine_leadtime_donothing"],
                      component_lead_reschedule=detail["component_lead_donothing"],
                      machine_start_times_changed=0)

    print(f"  --- run summary: {n_resched}/{len(event_rows)} events rescheduled | "
          f"without AI {row['donothing_tardiness_wd']:.2f}wd -> with AI "
          f"{(row['reschedule_tardiness_wd'] if row['reschedule_tardiness_wd'] is not None else row['donothing_tardiness_wd']):.2f}wd "
          f"| late jobs {row['late_jobs_donothing']} -> {row['late_jobs_reschedule']}")
    # plans are needed by the Gantt charts in adrs_charts.py; keys are stringified
    # by JSON and converted back on load
    detail["plan_without"] = {str(i): v for i, v in _realised_plan(des_dn["events"]).items()}
    detail["plan_with"] = {str(i): v for i, v in
                           _realised_plan((des_rs if n_resched else des_dn)["events"]).items()}
    products = build_product_rows(ops, nominal_des, base_committed, final_committed,
                                  known_return_days, kind, label, n_resched, event_rows)
    return row, detail, products

def build_product_rows(ops, nominal_des, base_committed, final_committed,
                       known_return_days, kind, label, n_resched, event_rows=()):
    """One row per PRODUCT (all 20), giving each metric's INITIAL and FINAL
    value for this scenario's run.

    Three states are recorded per product:
      initial_*   the nominal committed plan at load-up, before any disruption
      without_*   that same plan carried through every disruption, never
                  rescheduled (the counterfactual)
      final_*     the plan actually in force at the end of the run
    All are measured by the DES, so they reflect real resource contention
    rather than the optimiser's own start times."""
    ar_base = leads_from_return_days(base_committed, known_return_days)
    ar_final = leads_from_return_days(final_committed, known_return_days)
    des_without = sim.simulate(ops, base_committed, ar_base)
    des_final = (sim.simulate(ops, final_committed, ar_final) if n_resched else des_without)

    release = {o["job"]: o["release"] for o in ops if o["kind"] == "asm"}
    due = {o["job"]: o["due"] for o in ops if o["kind"] == "asm"}
    disrupted = {ops[i]["job"] for i in known_return_days}

    waits = {tag: _job_wait_times(ops, d["events"])
             for tag, d in (("initial", nominal_des), ("without", des_without),
                            ("final", des_final))}

    rows = []
    for j in sorted({o["job"] for o in ops}):
        r = dict(scenario=kind, label=label, product=j,
                 due_wd=due[j] / MPD,
                 own_part_disrupted=j in disrupted)
        for tag, des in (("initial", nominal_des), ("without", des_without),
                         ("final", des_final)):
            k = des["jobs"][j]
            r[f"{tag}_completion_wd"] = k["completion"] / MPD
            r[f"{tag}_production_time_hrs"] = (k["completion"] - release[j]) / 60.0
            r[f"{tag}_tardiness_wd"] = k["tardiness"] / MPD
            r[f"{tag}_on_time"] = k["on_time"]
            r[f"{tag}_machine_wait_hrs"] = waits[tag][j] / 60.0
        # ---- per-event trace: what this product's tardiness was when each
        # disruption landed, and what it became after that event's decision.
        # `eN_disruption` is filled only when THIS product's own component was
        # the one that deviated at event N.
        hit_at = []
        for er in event_rows:
            n = er["event"]
            own = er.get("_by_job", {}).get(j)
            r[f"e{n}_disruption"] = own or ""
            if own:
                hit_at.append(str(n))
            r[f"e{n}_clock_wd"] = er["now_wd"]
            r[f"e{n}_decision"] = er["decision"]
            r[f"e{n}_tardiness_before_wd"] = er.get("_tard_before", {}).get(j, 0) / MPD
            r[f"e{n}_tardiness_after_wd"] = er.get("_tard_after", {}).get(j, 0) / MPD
        r["own_part_disrupted_at_events"] = ",".join(hit_at)

        # headline deltas, so the table is readable without arithmetic
        r["tardiness_saved_wd"] = r["without_tardiness_wd"] - r["final_tardiness_wd"]
        r["production_time_saved_hrs"] = (r["without_production_time_hrs"]
                                          - r["final_production_time_hrs"])
        rows.append(r)
    return rows

# ==================================================================== plotting ====
def _drop_demo(df):
    """Filter out demo rows safely. `is_demo` arrives from CSV as object dtype
    (it is blank on the event and product rows of the combined file), and `~` on
    an object Series does integer bitwise negation -- turning False into -1 and
    producing a column lookup instead of a mask. Coercing to bool first avoids
    that."""
    if "is_demo" not in df.columns:
        return df.copy()
    return df[~df["is_demo"].fillna(False).astype(bool)].copy()

def _short_labels(kinds):
    return [PLOT_LABELS[k] for k in kinds]


def _objective_note(df):
    """Human-readable label of the committed objective, for figure titles.
    Empty when the column is missing or runs used different objectives."""
    if "objective_label" not in df.columns:
        return ""
    vals = _drop_demo(df)["objective_label"].dropna().unique()
    return str(vals[0]) if len(vals) == 1 else ""


def _present(df):
    """Scenario keys actually present in this result set, in canonical order --
    so a single-scenario run plots just that scenario instead of blank gaps."""
    return [k for k in SCENARIO_KINDS if k in set(df["scenario"])]


def make_fig1_tardiness(df, out_prefix):
    """Total tardiness per scenario, without vs with agentic AI, measured at the
    end of the run against the full set of observed returns."""
    df = _drop_demo(df)
    df["effective_tardiness_wd"] = df["donothing_tardiness_wd"] - df["tardiness_reduction_wd"]
    kinds = _present(df)
    labels = _short_labels(kinds)

    dn = [df[df.scenario == k]["donothing_tardiness_wd"].mean() for k in kinds]
    eff = [df[df.scenario == k]["effective_tardiness_wd"].mean() for k in kinds]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.plot(x, dn, marker="o", color="crimson",
            label="Without Agentic AI (committed plan carried through every disruption)")
    ax.plot(x, eff, marker="o", color="seagreen",
            label="With Agentic AI (rescheduled as events arrived)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Scenario"); ax.set_ylabel("Total tardiness (working-days)")
    n_ev = int(df["n_events"].max()) if "n_events" in df.columns else 0
    obj = _objective_note(df)
    ax.set_title(f"Total Tardiness Across Scenarios (one run each, up to {n_ev} disruption events)"
                 + (f"\nRescheduling objective: {obj}" if obj else ""))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig1_tardiness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig2_machine_utilisation(df, out_prefix):
    """DES-measured utilisation of every resource (M1..M6 + ASM), one coloured
    line each; solid = with agentic AI, dashed = without."""
    df = _drop_demo(df)
    kinds = _present(df)
    labels = _short_labels(kinds)
    resources = sorted([c[len("des_donothing_util_"):] for c in df.columns
                        if c.startswith("des_donothing_util_")],
                       key=lambda r: (r == core.ASM_STATION, r))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, res in enumerate(resources):
        dn_col, rs_col = f"des_donothing_util_{res}", f"des_reschedule_util_{res}"
        if rs_col not in df.columns:
            df[rs_col] = np.nan
        df["_eff"] = np.where((df["impact_decision"] == "reschedule") & df[rs_col].notna(),
                              df[rs_col], df[dn_col])
        ax.plot(x, [df[df.scenario == k]["_eff"].mean() for k in kinds],
                marker="o", label=res, color=colors[i % 10])
        ax.plot(x, [df[df.scenario == k][dn_col].mean() for k in kinds],
                marker="x", linestyle="--", alpha=0.6, color=colors[i % 10])

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Scenario"); ax.set_ylabel("Utilisation (%)")
    obj = _objective_note(df)
    ax.set_title("Resource Utilisation Across Scenarios (DES-measured, all machines + ASM)"
                 + (f"\nRescheduling objective: {obj}" if obj else ""))
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    style = [Line2D([0], [0], color="black", marker="o", label="With Agentic AI"),
             Line2D([0], [0], color="black", marker="x", linestyle="--", alpha=0.6,
                    label="Without Agentic AI")]
    leg1 = ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Resource")
    ax.add_artist(leg1)
    ax.legend(handles=style, loc="upper left", bbox_to_anchor=(1.02, 0.45), title="Condition")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig2_machine_utilisation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig3_shopwide_queue(df, out_prefix):
    """Avg queue length averaged across all resources -- a shop-wide congestion
    indicator, robust to which particular machine was hit."""
    df = _drop_demo(df)
    kinds = _present(df)
    labels = _short_labels(kinds)
    df["qlen_eff"] = np.where(df["impact_decision"] == "reschedule",
                              df["des_reschedule_avg_qlen_shopwide"],
                              df["des_donothing_avg_qlen_shopwide"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, [df[df.scenario == k]["des_donothing_avg_qlen_shopwide"].mean() for k in kinds],
            marker="o", label="Without Agentic AI", color="crimson")
    ax.plot(labels, [df[df.scenario == k]["qlen_eff"].mean() for k in kinds],
            marker="o", label="With Agentic AI", color="seagreen")
    obj = _objective_note(df)
    ax.set_title("Shop-Wide Avg Queue Length Across Scenarios\n(averaged across all machines + ASM)"
                 + (f"\nRescheduling objective: {obj}" if obj else ""))
    ax.set_xlabel("Scenario"); ax.set_ylabel("Avg queue length (jobs)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig3_shopwide_queue.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig6_machine_waiting(df, out_prefix):
    """DES-measured average waiting time at every resource, same solid/dashed
    with-vs-without convention as fig2."""
    df = _drop_demo(df)
    kinds = _present(df)
    labels = _short_labels(kinds)
    resources = sorted([c[len("des_donothing_wait_"):] for c in df.columns
                        if c.startswith("des_donothing_wait_")],
                       key=lambda r: (r == core.ASM_STATION, r))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, res in enumerate(resources):
        dn_col, rs_col = f"des_donothing_wait_{res}", f"des_reschedule_wait_{res}"
        if rs_col not in df.columns:
            df[rs_col] = np.nan
        df["_eff"] = np.where((df["impact_decision"] == "reschedule") & df[rs_col].notna(),
                              df[rs_col], df[dn_col])
        ax.plot(x, [df[df.scenario == k]["_eff"].mean() for k in kinds],
                marker="o", label=res, color=colors[i % 10])
        ax.plot(x, [df[df.scenario == k][dn_col].mean() for k in kinds],
                marker="x", linestyle="--", alpha=0.6, color=colors[i % 10])

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Scenario"); ax.set_ylabel("Avg waiting time (hours)")
    obj = _objective_note(df)
    ax.set_title("Component Waiting Time at Each Machine Across Scenarios"
                 + (f"\nRescheduling objective: {obj}" if obj else ""))
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    style = [Line2D([0], [0], color="black", marker="o", label="With Agentic AI"),
             Line2D([0], [0], color="black", marker="x", linestyle="--", alpha=0.6,
                    label="Without Agentic AI")]
    leg1 = ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Resource")
    ax.add_artist(leg1)
    ax.legend(handles=style, loc="upper left", bbox_to_anchor=(1.02, 0.45), title="Condition")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig6_machine_waiting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig7_late_jobs(df, out_prefix):
    """Number of late jobs per scenario -- complements fig1's magnitude with a
    count of how many products actually miss their date."""
    df = _drop_demo(df)
    kinds = _present(df)
    labels = _short_labels(kinds)
    x = np.arange(len(labels)); w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, [df[df.scenario == k]["late_jobs_donothing"].mean() for k in kinds],
           w, label="Without Agentic AI", color="crimson")
    ax.bar(x + w/2, [df[df.scenario == k]["late_jobs_reschedule"].mean() for k in kinds],
           w, label="With Agentic AI", color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Scenario"); ax.set_ylabel("Number of late jobs (out of 20 products)")
    obj = _objective_note(df)
    ax.set_title("Number of Late Jobs Across Scenarios"
                 + (f"\nRescheduling objective: {obj}" if obj else ""))
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig7_late_jobs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- per-product/machine heatmaps ---
def _by_scenario(details):
    """{scenario: detail} -- one run per scenario now, so no representative
    selection is needed any more."""
    return {d["scenario"]: d for d in details}


def make_fig4_product_leadtime(ops, details, out_prefix):
    """Per-product total production time (hours), scenarios as columns."""
    reps = _by_scenario(details)
    jobs = sorted({o["job"] for o in ops})
    release = {o["job"]: o["release"] for o in ops if o["kind"] == "asm"}
    kinds = [k for k in SCENARIO_KINDS if k in reps]
    if not kinds:
        return

    for key, tag, title in [("job_completion_donothing", "donothing", "Without Agentic AI"),
                            ("job_completion_reschedule", "reschedule", "With Agentic AI")]:
        data = [[round((reps[k][key][j] - release[j]) / 60.0) for k in kinds] for j in jobs]
        dfh = pd.DataFrame(data, index=[f"Product {j}" for j in jobs],
                           columns=[PLOT_LABELS[k] for k in kinds])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dfh, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                    cbar_kws={"label": "Total Production Time (hours)"})
        note = ", ".join(f"{PLOT_LABELS[k]}: {reps[k].get('n_events', '?')} events"
                         for k in kinds if k != "baseline")
        ax.set_title(f"Product Total Production Times Across Scenarios ({title})\n"
                     f"end of run -- {note}", fontsize=11)
        ax.set_xlabel("Scenario"); ax.set_ylabel("Product")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_fig4_product_leadtime_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def make_fig5_product_waiting(ops, details, out_prefix):
    """Per-product waiting time at machines (hours), scenarios as columns."""
    reps = _by_scenario(details)
    jobs = sorted({o["job"] for o in ops})
    kinds = [k for k in SCENARIO_KINDS if k in reps]
    if not kinds:
        return

    for key, tag, title in [("job_wait_donothing", "donothing", "Without Agentic AI"),
                            ("job_wait_reschedule", "reschedule", "With Agentic AI")]:
        data = [[round(reps[k][key][j] / 60.0, 1) for k in kinds] for j in jobs]
        dfh = pd.DataFrame(data, index=[f"Product {j}" for j in jobs],
                           columns=[PLOT_LABELS[k] for k in kinds])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dfh, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Waiting Time (hours)"})
        ax.set_title(f"Product Waiting Time at Machines Across Scenarios ({title})\n"
                     f"end of run", fontsize=11)
        ax.set_xlabel("Scenario"); ax.set_ylabel("Product")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_fig5_product_waiting_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _machine_labels(ops):
    machines = sorted({o["machine"] for o in ops
                       if o["kind"] == "comp" and not o["outsourced"]})
    comp = {}
    for o in ops:
        if o["kind"] == "comp" and not o["outsourced"]:
            comp.setdefault(o["machine"], o["comp"])
    return machines, [f"{m} ({comp.get(m, '?')})" for m in machines]


def _condition_labels(kind):
    """Panel labels for the without/with pair. The baseline scenario has no
    disruptions and so never invokes the agent -- its comparison is the EDD
    list schedule against the CP-SAT plan, i.e. what the OPTIMISER contributes.
    Every other scenario holds the optimiser fixed and varies only whether the
    agent rescheduled."""
    if kind == "baseline":
        return "Without Optimisation", "With Optimisation"
    return "Without Agentic AI", "With Agentic AI"

def _machine_heatmap(ops, details, out_prefix, dn_key, rs_key, cmap, cbar, title, fname_part):
    """Shared Product x Machine heatmap builder for fig8 (waiting) and fig9
    (production time). One PNG per scenario, Without | With side by side. A
    scenario is skipped when no machine start time moved -- both panels would be
    identical by construction and the figure would say nothing."""
    reps = _by_scenario(details)
    jobs = sorted({o["job"] for o in ops})
    machines, mlabels = _machine_labels(ops)
    written = []
    for kind in SCENARIO_KINDS:
        rep = reps.get(kind)
        if not rep or rep.get("machine_start_times_changed", 0) < 3:
            continue
        lab_dn, lab_rs = _condition_labels(kind)
        # baseline has no rescheduling to compare against -- one panel only
        pairs = ([(rs_key, "")] if kind == "baseline"
                 else [(dn_key, lab_dn), (rs_key, lab_rs)])
        fig, axarr = plt.subplots(1, len(pairs), figsize=(7.6 * len(pairs), 8),
                                  squeeze=False)
        axes = axarr[0]
        vals = [v for key, _ in pairs for j in jobs
                for v in rep[key][j].values()]
        vmin, vmax = 0, max(vals) / 60.0
        for ax, (key, tag) in zip(axes, pairs):
            data = [[round(rep[key][j].get(m, 0.0) / 60.0, 1) for m in machines] for j in jobs]
            dfh = pd.DataFrame(data, index=[f"Product {j}" for j in jobs], columns=mlabels)
            sns.heatmap(dfh, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                        vmin=vmin, vmax=vmax, cbar_kws={"label": cbar})
            ax.set_title(tag); ax.set_xlabel("Machine (Component)")
            ax.set_ylabel("Product")
            
        if len(pairs) > 1:
            title_fontsize = axes[0].title.get_fontsize()
            for letter, ax in zip("ab", axes):
                ax.text(0.5, -0.14, f"({letter})", transform=ax.transAxes,
                       ha="center", va="top", fontsize=title_fontsize)
        
        fig.suptitle(f"{title} -- {PLOT_LABELS[kind]}"
                     + (f"\nRescheduling objective: {rep['objective_label']}"
                        if rep.get("objective_label") else ""), fontsize=12)
        fig.tight_layout()
        fn = f"{out_prefix}_{fname_part}_{kind}.png"
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(fn)
    return written


def make_fig8_product_machine_wait(ops, details, out_prefix):
    return _machine_heatmap(ops, details, out_prefix,
                            "job_machine_wait_donothing", "job_machine_wait_reschedule",
                            "YlOrRd", "Waiting Time (hours)",
                            "Component Queueing Time at Each Machine, by Product",
                            "fig8_product_machine_wait")


def make_fig9_product_machine_leadtime(ops, details, out_prefix):
    return _machine_heatmap(ops, details, out_prefix,
                            "job_machine_leadtime_donothing", "job_machine_leadtime_reschedule",
                            "YlGnBu", "Production Time (hours)",
                            "Component Production Time at Each Machine (queueing + processing), by Product",
                            "fig9_product_machine_leadtime")


def make_fig10_event_timeline(event_df, out_prefix):
    """NEW, specific to the sequential design: how total tardiness evolves as
    disruptions arrive. One line per scenario; a marker at each event, filled
    where the agent rescheduled and hollow where it did not. This is the figure
    that shows the rolling-horizon behaviour the old one-shot design could not."""
    if event_df is None or event_df.empty:
        return False
    kinds = [k for k in SCENARIO_KINDS if k in set(event_df["scenario"])]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, k in enumerate(kinds):
        sub = event_df[event_df.scenario == k].sort_values("now_wd")
        ax.plot(sub["now_wd"], sub["donothing_tardiness_wd"], marker="o",
                color=colors[i % 10], label=PLOT_LABELS[k])
        acted = sub[sub["decision"] == "reschedule"]
        ax.scatter(acted["now_wd"], acted["donothing_tardiness_wd"],
                   s=140, facecolors="none", edgecolors=colors[i % 10], linewidths=2)
    ax.set_xlabel("Clock at which the disruption became known (working-days)")
    ax.set_ylabel("Total tardiness if no action taken (working-days)")
    ax.set_title("Disruption Timeline: Tardiness Exposure as Events Arrive\n"
                 "(circled markers = the agent rescheduled)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_fig11_event_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def make_scenario_plots(ops, details, out_prefix):
    """Figures that describe ONE scenario in isolation -- the Product x Machine
    heatmaps. Produced by every single-scenario run."""
    print("\nGenerating per-scenario figures...")
    wrote = False
    for f in make_fig8_product_machine_wait(ops, details, out_prefix) or []:
        print(f"Wrote {f}"); wrote = True
    for f in make_fig9_product_machine_leadtime(ops, details, out_prefix) or []:
        print(f"Wrote {f}"); wrote = True
    if not wrote:
        print("  (fig8/fig9 skipped -- no machine start time moved in this run, so "
              "the Without and With panels would be identical)")


def make_combined_plots(ops, df, details, out_prefix, event_df=None):
    """Figures that compare scenarios against each other. These need every
    scenario's results together, so they are produced by --combine rather than
    by an individual run."""
    print("\nGenerating cross-scenario figures...")
    print(f"  scenarios present: {', '.join(_present(df))}")
    make_fig1_tardiness(df, out_prefix);            print(f"Wrote {out_prefix}_fig1_tardiness.png")
    make_fig2_machine_utilisation(df, out_prefix);  print(f"Wrote {out_prefix}_fig2_machine_utilisation.png")
    make_fig3_shopwide_queue(df, out_prefix);       print(f"Wrote {out_prefix}_fig3_shopwide_queue.png")
    if details:
        make_fig4_product_leadtime(ops, details, out_prefix)
        print(f"Wrote {out_prefix}_fig4_product_leadtime_{{donothing,reschedule}}.png")
        make_fig5_product_waiting(ops, details, out_prefix)
        print(f"Wrote {out_prefix}_fig5_product_waiting_{{donothing,reschedule}}.png")
    else:
        print("Skipped fig4/fig5 (no per-job detail available)")
    make_fig6_machine_waiting(df, out_prefix);      print(f"Wrote {out_prefix}_fig6_machine_waiting.png")
    make_fig7_late_jobs(df, out_prefix);            print(f"Wrote {out_prefix}_fig7_late_jobs.png")
    if make_fig10_event_timeline(event_df, out_prefix):
        print(f"Wrote {out_prefix}_fig10_event_timeline.png")


def load_run(path):
    """Read one scenario's result CSV plus its sidecar detail JSON.
    Returns (run_rows_df, event_rows_df_or_None, details_list)."""
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
        # JSON turns dict keys into strings; the heatmaps index by int job number
        for d in details:
            for key in ("job_completion_donothing", "job_completion_reschedule",
                        "job_wait_donothing", "job_wait_reschedule",
                        "job_machine_wait_donothing", "job_machine_wait_reschedule",
                        "job_machine_leadtime_donothing", "job_machine_leadtime_reschedule"):
                if key in d:
                    d[key] = {int(k): v for k, v in d[key].items()}
    else:
        print(f"(note: {det_path} not found -- fig4/5/8/9 need it)")
    return run_df, ev_df, details

# ======================================================================== main ====
def _details_path(out):
    """Only needed so --replot can rebuild the heatmaps; the CSV is the result."""
    return os.path.splitext(out)[0] + "_details.json"


def write_single_csv(path, summary, event_rows, product_rows):
    """Everything about ONE scenario run, in ONE file.

    Rows are stacked in three sections, distinguished by the `record` column:
        run      1 row  -- the end-of-run without-vs-with comparison
        event    1 row per disruption event -- what arrived, when, and what the
                 agent decided
        product  20 rows -- per-product metrics at each state of the run
    Columns not applicable to a section are left blank, so the file opens
    cleanly in Excel and filters cleanly in pandas (df[df.record == "product"])."""
    frames = []
    for name, rows in (("run", [summary]), ("event", event_rows), ("product", product_rows)):
        if not rows:
            continue
        f = pd.DataFrame(rows)
        f.insert(0, "record", name)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[[c for c in out.columns if not c.startswith("_")]]   # drop dict payloads
    # keep the identifying columns at the front, everything else after
    lead = [c for c in ("record", "scenario", "label", "event", "product") if c in out.columns]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(path, index=False)
    return out

def _products_path(out):
    return os.path.splitext(out)[0] + "_products.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="path to the job-shop .xlsx")
    p.add_argument("--scenario", default="late_1", choices=SCENARIO_KINDS,
                   help="which scenario to run (default late_1)")
    p.add_argument("--disruptions", type=int, default=0,
                   help="number of disruption EVENTS in the run. 0 (default) draws a "
                        "random number in [--min-disruptions, --max-disruptions].")
    p.add_argument("--min-disruptions", type=int, default=1)
    p.add_argument("--max-disruptions", type=int, default=5)
    p.add_argument("--objective", default="tardiness", choices=RESCHEDULE_OBJECTIVES,
                   help="which objective plan is committed on every reschedule "
                        "(default tardiness = fewest late deliveries). Use 'agent' via "
                        "--follow-recommendation to let the agent choose each time.")
    p.add_argument("--follow-recommendation", action="store_true",
                   help="commit whichever option the agent recommends at each event, "
                        "instead of holding the whole run to --objective")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--solver-time-limit", type=int, default=20,
                   help="CP-SAT seconds per solve (default 20)")
    p.add_argument("--out", default="adrs_experiment_results.csv")
    p.add_argument("--out-prefix", default="adrs", help="filename prefix for figures")
    p.add_argument("--plot", action="store_true", help="generate figures from this run")
    p.add_argument("--replot", action="store_true",
                   help="skip the pipeline entirely; reload --out and regenerate this "
                        "scenario's own figures (fig8/fig9). No Groq calls.")
    p.add_argument("--combine", nargs="+", metavar="CSV",
                   help="read several scenario result CSVs and draw the cross-scenario "
                        "figures (fig1-7 and fig11) from them together. Each file's "
                        "_details.json sidecar is picked up automatically. "
                        "Example: --combine s1.csv s2.csv s3.csv s4.csv s5.csv "
                        "--out-prefix all")
    args = p.parse_args()

    if args.combine:
        ops = core.load_ops(path=args.dataset)
        run_frames, ev_frames, details = [], [], []
        for path in args.combine:
            r, e, d = load_run(path)
            run_frames.append(r)
            if e is not None:
                ev_frames.append(e)
            details.extend(d)
            print(f"  loaded {path}: {list(r['scenario'])}")
        df = pd.concat(run_frames, ignore_index=True, sort=False)
        event_df = pd.concat(ev_frames, ignore_index=True, sort=False) if ev_frames else None
        dupes = df[df.duplicated("scenario", keep=False)]["scenario"].unique()
        if len(dupes):
            print(f"  WARNING: {list(dupes)} appears in more than one file; the figures "
                  f"will average those runs rather than show one per scenario.")
        make_combined_plots(ops, df, details, args.out_prefix, event_df)
        return df

    if args.replot:
        df, event_df, details = load_run(args.out)
        ops = core.load_ops(path=args.dataset)
        make_scenario_plots(ops, details, args.out_prefix)
        return df

    rng = random.Random(args.seed)
    core.DEFAULT_TIME_LIMIT = args.solver_time_limit
    ops = core.load_ops(path=args.dataset)
    nominal = core.build_schedule(ops, now=0)
    base_committed = nominal["plan"]
    print(f"Loaded {len(ops)} ops / {len({o['job'] for o in ops})} jobs from {args.dataset}")
    print(f"Nominal plan: status={nominal['status']} "
          f"tardiness={core.fmt_wd(nominal['total_tardiness'])} "
          f"makespan={core.fmt_wd(nominal['makespan'])}")

    kind = args.scenario
    pick = None if args.follow_recommendation else args.objective
    app = adrs_agents.build_app()
    nominal_des = sim.simulate(ops, base_committed, {})   # the load-up state

    if kind == "baseline":
        n = 0
    elif args.disruptions > 0:
        n = args.disruptions
    else:
        n = rng.randint(args.min_disruptions, args.max_disruptions)

    summary, event_rows, detail, products = run_scenario_run(
        app, ops, base_committed, kind, n, rng,
        pick_objective=pick, nominal_des=nominal_des)
    detail["n_events"] = summary["n_events"]

    combined = write_single_csv(args.out, summary, event_rows, products)
    with open(_details_path(args.out), "w") as f:
        json.dump([detail], f)
    print(f"\nWrote {len(combined)} rows to {args.out} "
          f"(1 run + {len(event_rows)} events + {len(products)} products)")
    print(f"Wrote heatmap detail to {_details_path(args.out)}")

    df = pd.DataFrame([summary])
    event_df = pd.DataFrame(event_rows) if event_rows else None
    details = [detail]

    print("\n" + "=" * 78)
    print("PER-SCENARIO SUMMARY")
    print("=" * 78)
    for _, r in df.iterrows():
        print(f"{r['label']}")
        print(f"    disruption events          : {r['n_events']} "
              f"({r['n_disrupted']} components disrupted in total)")
        print(f"    rescheduled at             : {r['n_reschedules']}/{r['n_events']} events")
        print(f"    tardiness without AI       : {r['donothing_tardiness_wd']:.2f}wd")
        rs = r["reschedule_tardiness_wd"]
        print(f"    tardiness with AI          : "
              f"{rs:.2f}wd" if pd.notna(rs) else "    tardiness with AI          : unchanged")
        print(f"    total reduction            : {r['tardiness_reduction_wd']:.2f}wd")
        print(f"    late jobs without -> with  : {r['late_jobs_donothing']} -> {r['late_jobs_reschedule']}")
        print(f"    machine start times changed: {r['machine_start_times_changed']}")

    if args.plot:
        make_scenario_plots(ops, details, args.out_prefix)
        print("\n(fig1-7 and fig11 compare scenarios and need every run's results -- "
              "produce them once all five CSVs exist:\n"
              "   python adrs_experiments.py --combine s1.csv s2.csv s3.csv s4.csv s5.csv "
              "--out-prefix all)")

    return df


if __name__ == "__main__":
    main()