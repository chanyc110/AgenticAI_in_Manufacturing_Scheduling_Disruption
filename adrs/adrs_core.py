"""
ADRS core scheduling module (no LLM — pure mechanics the agents wrap).

Model
-----
Job = Sr. No. Each job has component operations (in-house on M1..M6, or outsourced
off-site using no machine) plus one final ASSEMBLY operation on a single shared
station "ASM". Assembly cannot start until ALL of the job's components are ready,
including the returned outsourced part. A job completes when its assembly finishes.

This precedence is what lets a late outsource return reach a shared resource (ASM),
which is what gives rescheduling something to optimise.

Calendar: weekdays only, 09:00-17:00 (8h/day). Time unit = working minute from the
common release (2024-08-21 09:00). Objective: minimise total tardiness, then makespan.

Public API
----------
load_ops(path)                                  -> list of operation dicts
build_schedule(ops, now, frozen, actual_leads)  -> optimal plan for remaining problem
evaluate_donothing(ops, committed, actual)      -> tardiness if plan is kept (right-shift)
reschedule_from(ops, committed, now, actual)    -> frozen-horizon complete re-solve
Helpers: outsourced_ops, assembly_index, fmt_wd, asm_order
"""
import math
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

SEED = 42
LEAD_MIN_DAYS, LEAD_MAX_DAYS = 3, 10
MIN_PER_DAY = 480
ORIGIN = datetime(2024, 8, 21, 9, 0)
PATH = "job_shop_updated.xlsx"
ASM_STATION = "ASM"


# ---------- calendar ----------
def to_work_minutes(dt):
    if dt <= ORIGIN:
        return 0
    total, cur = 0, ORIGIN
    while cur.date() < dt.date():
        if cur.weekday() < 5:
            total += MIN_PER_DAY
        cur = (cur + timedelta(days=1)).replace(hour=9, minute=0)
    if dt.weekday() < 5:
        total += max(0, min(dt.hour, 17) * 60 - 540)
    return total


def fmt_wd(mins):
    return f"{mins / MIN_PER_DAY:.1f}wd"


# ---------- data ----------
def load_ops(path=PATH):
    """Return operations. Component/assembly durations come from Cycle Time (seconds)
    x Quantity. Outsource lead times come from a dedicated column (Assumed Lead Time
    (days)) — fixed and reproducible, no randomness. Columns are read by name, so
    reordering or adding spreadsheet columns won't silently break this."""
    import pandas as pd
    df = pd.read_excel(path)
    ops = []
    for _, r in df.iterrows():
        op_type = r["Operation"]
        proc = r["Process Type"]
        machine = r["Machine Number"]
        qty = r["Quantity Required"]
        cyc = r["Cycle Time (seconds)"]
        lead_days = r["Assumed Lead Time (days)"]
        job = int(r["Sr. No"])
        due = to_work_minutes(pd.Timestamp(r["Promised Delivery Date"]).to_pydatetime())

        if str(op_type) == "Assembly":
            kind, outsourced = "asm", False
            dur = math.ceil(qty * cyc / 60.0)
            machine = ASM_STATION
        elif proc == "Outsource":
            kind, outsourced = "comp", True
            dur = int(round(lead_days)) * MIN_PER_DAY   # dedicated column, in working-days
            machine = None
        else:
            kind, outsourced = "comp", False
            dur = math.ceil(qty * cyc / 60.0)

        ops.append(dict(idx=len(ops), job=job, comp=r["Components"],
                        kind=kind, machine=machine, outsourced=outsourced,
                        dur=dur, due=due))
    return ops


def outsourced_ops(ops):
    return [o for o in ops if o["outsourced"]]


def assembly_index(ops, job):
    return next(o["idx"] for o in ops if o["job"] == job and o["kind"] == "asm")


def _dur(o, actual_leads):
    if o["outsourced"] and o["idx"] in actual_leads:
        return actual_leads[o["idx"]]
    return o["dur"]


# ---------- scheduler ----------
def build_schedule(ops, now=0, frozen=None, actual_leads=None, objective="tardiness", reference_plan=None, time_limit=20):
    """Solve the (possibly partial) problem from `now`.
    frozen: {idx: start} operations pinned to a fixed start (already begun / known).
    actual_leads: {idx: minutes} real outsource durations that override the assumption.
    """
    now = int(now)
    frozen = {int(k): int(v) for k, v in (frozen or {}).items()}
    actual_leads = {int(k): int(v) for k, v in (actual_leads or {}).items()}
    horizon = sum(_dur(o, actual_leads) for o in ops) + max(o["due"] for o in ops) + MIN_PER_DAY * 40

    m = cp_model.CpModel()
    s, e = {}, {}
    machines = sorted({o["machine"] for o in ops if o["machine"]})
    mach_iv = {mc: [] for mc in machines}

    for o in ops:
        i = o["idx"]
        d = _dur(o, actual_leads)
        lo = 0 if i in frozen else now
        si = m.new_int_var(lo, horizon, f"s{i}")
        ei = m.new_int_var(lo, horizon, f"e{i}")
        iv = m.new_interval_var(si, d, ei, f"iv{i}")
        s[i], e[i] = si, ei
        if i in frozen:
            m.add(si == frozen[i])
            if o["outsourced"]:
                m.add(si == 0)        # dispatched at release; return time = lead time
        if o["machine"]:
            mach_iv[o["machine"]].append(iv)

    for mc in machines:
        m.add_no_overlap(mach_iv[mc])

    # assembly precedence: assembly starts only after all the job's components finish
    for a in ops:
        if a["kind"] == "asm":
            for c in ops:
                if c["job"] == a["job"] and c["kind"] == "comp":
                    m.add(s[a["idx"]] >= e[c["idx"]])

    jobs = sorted({o["job"] for o in ops})
    tard, comp = {}, {}
    for j in jobs:
        ai = assembly_index(ops, j)
        due = next(o["due"] for o in ops if o["job"] == j)
        cj = e[ai]                       # job completes when assembly finishes
        tj = m.new_int_var(0, horizon, f"T{j}")
        m.add_max_equality(tj, [cj - due, 0])
        comp[j], tard[j] = cj, tj

    mk = m.new_int_var(0, horizon, "mk")
    m.add_max_equality(mk, [e[o["idx"]] for o in ops])
    total_tard = sum(tard.values())

    if reference_plan is not None:                 # stability = total movement vs reference
        devs = []
        for o in ops:
            i = o["idx"]
            if i in reference_plan:
                dv = m.new_int_var(0, horizon, f"dev{i}")
                m.add_abs_equality(dv, s[i] - reference_plan[i]["start"])
                devs.append(dv)
        stability = sum(devs) if devs else 0
    else:
        stability = 0

    OBJECTIVES = {
        "tardiness": total_tard * 100000 + mk,
        "makespan":  mk * 100000 + total_tard,
        "stability": stability * 100000 + total_tard * 100 + mk,   # stability now dominates
        "balanced":  total_tard * 50 + stability * 30 + mk,        # genuine blend, tuned
    }
    m.minimize(OBJECTIVES.get(objective, OBJECTIVES["tardiness"]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = SEED
    status = solver.solve(m)

    plan = {o["idx"]: dict(start=solver.value(s[o["idx"]]),
                           end=solver.value(e[o["idx"]]),
                           machine=o["machine"]) for o in ops}
    return dict(status=solver.status_name(status), plan=plan,
                tardiness={j: solver.value(tard[j]) for j in jobs},
                completion={j: solver.value(comp[j]) for j in jobs},
                total_tardiness=sum(solver.value(tard[j]) for j in jobs),
                makespan=solver.value(mk))


# ---------- do-nothing (predictive-reactive right-shift) ----------
def evaluate_donothing(ops, committed, actual_leads, now=0):
    """Keep the committed plan's ASM order; right-shift in time to absorb the real
    outsource returns. Assembly ops already started before `now` are FROZEN at their
    committed times (they happened); only ops not yet started are right-shifted, and
    no earlier than `now`. This mirrors reschedule_from's frozen horizon so the two
    policies are compared on the same footing."""
    now = int(now)
    actual_leads = {int(k): int(v) for k, v in (actual_leads or {}).items()}

    comp_end = {}
    for o in ops:
        if o["kind"] == "comp":
            comp_end[o["idx"]] = committed[o["idx"]]["start"] + _dur(o, actual_leads)

    asm_ops = sorted([o for o in ops if o["kind"] == "asm"],
                     key=lambda o: committed[o["idx"]]["start"])
    clock, tard, comp = 0, {}, {}
    for o in asm_ops:
        cstart = committed[o["idx"]]["start"]
        ready = max(comp_end[c["idx"]] for c in ops
                    if c["job"] == o["job"] and c["kind"] == "comp")
        if cstart < now:                      # frozen: already started before now
            start = cstart
        else:                                 # not yet started: right-shift around now
            start = max(clock, ready, now)
        end = start + o["dur"]
        clock = max(clock, end)
        due = next(x["due"] for x in ops if x["job"] == o["job"])
        comp[o["job"]] = end
        tard[o["job"]] = max(0, end - due)
    return dict(tardiness=tard, completion=comp, total_tardiness=sum(tard.values()))


# ---------- reschedule (complete re-optimisation from now) ----------
def reschedule_from(ops, committed, now, actual_returns):
    """Frozen-horizon complete reschedule. Pin everything that has begun before
    `now`, enforce the known actual outsource return(s), and re-optimise the rest."""
    frozen = {}
    for o in ops:
        i = o["idx"]
        st = committed[i]["start"]
        if i in actual_returns:           # disrupted outsource: enforce real return
            frozen[i] = st
        elif st < now:                    # already started/finished in reality
            frozen[i] = st
    return build_schedule(ops, now=now, frozen=frozen, actual_leads=actual_returns)

OBJECTIVE_LABELS = {
    "tardiness": "Fewest late deliveries",
    "makespan":  "Finish batch soonest",
    "stability": "Least disruption to plan",
    "balanced":  "Balanced trade-off",
}

def plan_tardiness(ops, plan):
    """Per-job tardiness (working-minutes) for a given plan."""
    tard = {}
    for o in ops:
        if o["kind"] == "asm":
            due = next(x["due"] for x in ops if x["job"] == o["job"])
            tard[o["job"]] = max(0, plan[o["idx"]]["end"] - due)
    return tard

def generate_options(ops, committed, now, actual_returns,
                     objectives=("tardiness", "makespan", "stability", "balanced")):
    """Objectives layer: run the optimiser once per objective -> a set of option plans."""
    frozen = {}
    for o in ops:
        i = o["idx"]
        st = committed[i]["start"]
        if i in actual_returns or st < now:
            frozen[i] = st
    options = []
    for obj in objectives:
        res = build_schedule(ops, now=now, frozen=frozen, actual_leads=actual_returns,
                             objective=obj, reference_plan=committed)
        changed = sum(1 for o in ops
                      if res["plan"][o["idx"]]["start"] != committed[o["idx"]]["start"])
        options.append(dict(objective=obj, label=OBJECTIVE_LABELS[obj],
                            plan=res["plan"], total_tardiness=res["total_tardiness"],
                            makespan=res["makespan"], ops_changed=changed,
                            asm_order=asm_order(ops, res["plan"])))
    return options

def asm_order(ops, plan):
    """The sequence of jobs on the ASM station under a given plan (for explanations)."""
    asm = sorted([o for o in ops if o["kind"] == "asm"], key=lambda o: plan[o["idx"]]["start"])
    return [o["job"] for o in asm]

def compute_job_stats(ops, plan, actual_leads=None):
    """Per-job stats against a plan: due date, predicted completion, tardiness,
    and WHAT caused the lateness — either a specific component (the one whose
    finish time gated assembly) or ASM station queue congestion (assembly was
    ready but had to wait its turn on the shared station)."""
    actual_leads = actual_leads or {}
    stats = {}
    for j in sorted({o["job"] for o in ops}):
        comps = [o for o in ops if o["job"] == j and o["kind"] == "comp"]
        asm = next(o for o in ops if o["job"] == j and o["kind"] == "asm")
        due = asm["due"]
        comp_finish = {c["idx"]: plan[c["idx"]]["start"] + _dur(c, actual_leads) for c in comps}
        ready = max(comp_finish.values()) if comp_finish else 0
        asm_start, asm_end = plan[asm["idx"]]["start"], plan[asm["idx"]]["end"]
        tardiness = max(0, asm_end - due)

        if asm_start > ready + 1:                      # ready but station was busy
            cause, cause_kind = "ASM station queue delay", "congestion"
        else:                                           # gated by a specific component
            limiting = max(comps, key=lambda c: comp_finish[c["idx"]])
            cause = f"{limiting['comp']} ({'outsourced' if limiting['outsourced'] else limiting['machine']})"
            cause_kind = "outsourced" if limiting["outsourced"] else "in-house"

        stats[j] = dict(job=j, due=due, completion=asm_end, tardiness=tardiness,
                        on_time=(tardiness == 0), cause=cause, cause_kind=cause_kind,
                        asm_start=asm_start, ready=ready)
    return stats


def compute_machine_utilisation(ops, plan):
    """Busy time / overall makespan, per machine (incl. ASM)."""
    machines = sorted({o["machine"] for o in ops if o["machine"]})
    makespan = max(p["end"] for p in plan.values())
    util = {}
    for mc in machines:
        busy = sum(plan[o["idx"]]["end"] - plan[o["idx"]]["start"]
                   for o in ops if o["machine"] == mc)
        util[mc] = dict(busy=busy, makespan=makespan,
                        pct=(busy / makespan * 100) if makespan else 0)
    return util


def compute_waiting_times(ops, plan, actual_leads=None):
    """Queueing delay per operation: time between when a part COULD have started
    (release=0 for in-house components; all-components-ready for assembly) and
    when it actually started. Outsourced legs are excluded — they're dispatched
    immediately off-site, so there's no queueing concept for them."""
    actual_leads = actual_leads or {}
    rows = []
    for o in ops:
        if o["kind"] == "comp" and not o["outsourced"]:
            wait = max(0, plan[o["idx"]]["start"] - 0)
            rows.append(dict(job=o["job"], comp=o["comp"], kind="in-house", wait=wait))
        elif o["kind"] == "asm":
            comps = [c for c in ops if c["job"] == o["job"] and c["kind"] == "comp"]
            ready = max((plan[c["idx"]]["start"] + _dur(c, actual_leads) for c in comps), default=0)
            wait = max(0, plan[o["idx"]]["start"] - ready)
            rows.append(dict(job=o["job"], comp="Assembly", kind="assembly", wait=wait))
    return rows


# ---------- demo ----------
if __name__ == "__main__":
    ops = load_ops()
    nominal = build_schedule(ops, now=0)
    print(f"NOMINAL  status={nominal['status']}  "
          f"total tardiness={fmt_wd(nominal['total_tardiness'])}  makespan={fmt_wd(nominal['makespan'])}")
    print(f"ASM order (nominal): {asm_order(ops, nominal['plan'])}\n")

    dis = next(o['idx'] for o in ops if o['job'] == 8 and o['outsourced'])
    assumed = ops[dis]['dur']
    actual = {dis: 22 * MIN_PER_DAY}
    now = assumed
    print(f"DISRUPTION  Job 8 outsourced {ops[dis]['comp']}: assumed {fmt_wd(assumed)} -> "
          f"actual {fmt_wd(actual[dis])}, learned at now={fmt_wd(now)}\n")

    dn = evaluate_donothing(ops, nominal['plan'], actual)
    rs = reschedule_from(ops, nominal['plan'], now, actual)
    print(f"DO-NOTHING  total tardiness={fmt_wd(dn['total_tardiness'])}")
    print(f"RESCHEDULE  total tardiness={fmt_wd(rs['total_tardiness'])}  status={rs['status']}")
    print(f"GAP closed by rescheduling = {fmt_wd(dn['total_tardiness'] - rs['total_tardiness'])}\n")
    print(f"ASM order (reschedule): {asm_order(ops, rs['plan'])}  <- Job 8 pushed back; ready jobs first")
    print("\nPer-job tardiness (working-days):")
    print(f"  {'job':>3}{'do-nothing':>12}{'reschedule':>12}")
    for j in sorted(dn['tardiness']):
        print(f"  {j:>3}{dn['tardiness'][j] / MIN_PER_DAY:>12.1f}{rs['tardiness'][j] / MIN_PER_DAY:>12.1f}")