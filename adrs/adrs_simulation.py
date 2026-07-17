"""
ADRS discrete-event simulation & live animation (SimPy + standalone HTML/Canvas).

adrs_core solves a schedule with CP-SAT; this module does NOT trust that plan's
start/end times directly. Instead it RE-DERIVES what would actually happen by
simulating real resource contention: machines M1..Mn and the shared ASM station
are each modelled as a capacity-1 server, assembly genuinely waits on an AllOf of
its job's component processes (including a late-returning outsourced part), and
every KPI below is measured from the resulting event log rather than computed off
the plan. Dispatch order on each resource is pinned to match the given plan's
chosen sequence via an explicit predecessor-chain of SimPy events (see simulate()
docstring for why this isn't a PriorityResource) -- so the simulation is
validating/re-enacting THIS plan's sequencing decision rather than inventing its
own, but the actual times, waits and queue lengths are genuinely simulated, not
copied from the plan.

Public API
----------
simulate(ops, plan, actual_returns=None)   -> dict(events, makespan, resources,
                                                     outsourced, jobs, overall)
build_launch_widget(...)                    -> small HTML snippet for
                                                st.components.v1.html; its button
                                                opens the full animation in a new tab
build_animation_page(...)                   -> the full standalone HTML page
"""
import base64
import json

import simpy

import adrs_core as core

# Same hex values as plotly.express.colors.qualitative.Plotly, hard-coded so this
# module carries no plotly dependency of its own; keeps per-product colours
# consistent with the main dashboard's charts.
JOB_PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
               "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]


def _natural_key(label):
    """Sort key so component labels sort 'C2' < 'C10', not lexicographically."""
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    prefix = "".join(ch for ch in str(label) if not ch.isdigit())
    return (prefix, int(digits) if digits else 0)


# ============================================================ discrete-event sim ====
def simulate(ops, plan, actual_returns=None):
    """Runs the DES for one schedule (a committed plan, or one of the generated
    reschedule options) and returns the raw event log plus KPIs computed purely
    from observed simulated events (see module docstring).

    Sequencing model: each machine and the shared ASM station is a capacity-1
    server that processes its assigned operations in the STRICT order the given
    plan chose for it (ranked by that plan's start times). This is deliberately
    NOT implemented as SimPy priority-resource contention: a priority queue can
    only rank requests that have already arrived, so if the plan's intended next
    job happens to not be ready yet, a lower-priority-but-ready job would jump
    the queue -- silently reordering the schedule and breaking the "keep the
    committed order, right-shift in time" behaviour that adrs_core.evaluate_
    donothing() (and, for a reschedule option, the CP-SAT solve itself) relies
    on. Instead each operation explicitly waits on an event fired by its
    immediate predecessor on that resource, so the server genuinely idles if the
    correct next job isn't ready -- it never reorders around a hold-up. This is
    what lets a late outsourced return generate real collateral delay for other
    jobs queued behind it on ASM, which is the mechanism the whole reschedule
    decision is trying to protect against."""
    actual_returns = actual_returns or {}
    env = simpy.Environment()

    machines = sorted({o["machine"] for o in ops
                        if o["machine"] and o["machine"] != core.ASM_STATION})
    resource_names = machines + [core.ASM_STATION]

    order_by_resource = {name: [] for name in resource_names}
    for o in ops:
        if o["kind"] == "asm":
            order_by_resource[core.ASM_STATION].append(o)
        elif o["kind"] == "comp" and not o["outsourced"]:
            order_by_resource[o["machine"]].append(o)
    for name in order_by_resource:
        order_by_resource[name].sort(key=lambda o: plan[o["idx"]]["start"])

    # idx -> event fired when that op FINISHES and frees the resource, used by the
    # next op in that resource's chain as its "wait for my turn" event. (Firing on
    # start rather than finish was an earlier bug here: it let every op queued
    # behind a delay begin at the same instant instead of queueing properly.)
    my_turn = {}        # idx -> predecessor's finished-event, or None if first up
    finished_event = {}  # idx -> this op's own finished-event (for the next op to await)
    for name, seq in order_by_resource.items():
        prev_finished = None
        for o in seq:
            my_turn[o["idx"]] = prev_finished
            finished_event[o["idx"]] = env.event()
            prev_finished = finished_event[o["idx"]]

    events = []

    def component_proc(o):
        release = o.get("release", 0) or 0
        if env.now < release:
            yield env.timeout(release - env.now)
        arrival = env.now
        if o["outsourced"]:
            # No shared capacity off-site: dispatched immediately, elapses the
            # ACTUAL lead time if one was reported, else the assumed duration.
            dur = actual_returns.get(o["idx"], o["dur"])
            yield env.timeout(dur)
            events.append(dict(idx=o["idx"], job=o["job"], comp=o["comp"], kind="comp",
                                resource="OutSrc", outsourced=True,
                                arrival=arrival, start=arrival, end=env.now,
                                wait=0, dur=dur))
        else:
            pred = my_turn[o["idx"]]
            if pred is not None:
                yield pred                      # server may idle here, on purpose
            start = env.now
            dur = o["dur"]
            yield env.timeout(dur)
            finished_event[o["idx"]].succeed()  # NOW frees the next op in line
            events.append(dict(idx=o["idx"], job=o["job"], comp=o["comp"], kind="comp",
                                resource=o["machine"], outsourced=False,
                                arrival=arrival, start=start, end=env.now,
                                wait=start - arrival, dur=dur))

    def assembly_proc(o, comp_procs):
        # Genuine precedence: this yields on the actual component processes, so a
        # late outsourced return really does hold up assembly here, not just on
        # paper.
        yield simpy.AllOf(env, comp_procs)
        arrival = env.now
        pred = my_turn[o["idx"]]
        if pred is not None:
            yield pred
        start = env.now
        dur = o["dur"]
        yield env.timeout(dur)
        finished_event[o["idx"]].succeed()
        events.append(dict(idx=o["idx"], job=o["job"], comp="Assembly", kind="asm",
                            resource=core.ASM_STATION, outsourced=False,
                            arrival=arrival, start=start, end=env.now,
                            wait=start - arrival, dur=dur))

    jobs = sorted({o["job"] for o in ops})
    comp_procs = {j: [] for j in jobs}
    for o in ops:
        if o["kind"] == "comp":
            comp_procs[o["job"]].append(env.process(component_proc(o)))
    for o in ops:
        if o["kind"] == "asm":
            env.process(assembly_proc(o, comp_procs[o["job"]]))

    env.run()

    return _summarise(ops, jobs, machines, resource_names, events)


def _time_weighted_avg(samples, horizon):
    """Integrates a (time, length) step function over [0, horizon] and divides by
    horizon -- the standard L_bar = (1/T) * integral(L(t) dt) queueing metric."""
    if horizon <= 0 or not samples:
        return 0.0
    area = 0.0
    for (t0, l0), (t1, _l1) in zip(samples, samples[1:]):
        area += l0 * (t1 - t0)
    last_t, last_l = samples[-1]
    if last_t < horizon:
        area += last_l * (horizon - last_t)
    return area / horizon


def _queue_length_series(evs):
    """Reconstructs the (time, queue_length) step function for one resource
    directly from its events' arrival/start timestamps -- an op counts toward the
    queue for exactly [arrival, start), i.e. ready-and-waiting-its-turn but not
    yet in service. (Outsourced legs have arrival == start, so they contribute
    nothing, correctly reflecting that they never queue for shared capacity.)"""
    marks = []
    for e in evs:
        marks.append((e["arrival"], 1))
        marks.append((e["start"], -1))
    marks.sort(key=lambda x: x[0])
    samples, running = [(0, 0)], 0
    for t, d in marks:
        running += d
        samples.append((t, running))
    return samples


def _summarise(ops, jobs, machines, resource_names, events):
    makespan = max((e["end"] for e in events), default=0)

    resource_kpis = {}
    for name in resource_names:
        evs = [e for e in events if e["resource"] == name]
        busy = sum(e["end"] - e["start"] for e in evs)
        waits = [e["wait"] for e in evs]
        q_samples = _queue_length_series(evs)
        resource_kpis[name] = dict(
            n_ops=len(evs),
            avg_wait=(sum(waits) / len(waits)) if waits else 0.0,
            avg_queue_length=_time_weighted_avg(q_samples, makespan),
            max_queue_length=max((l for _, l in q_samples), default=0),
            utilisation_pct=(busy / makespan * 100) if makespan else 0.0,
            total_production_time=busy,
            idle_time=max(0, makespan - busy),
        )

    out_evs = [e for e in events if e.get("outsourced")]
    outsourced_kpi = dict(
        n_ops=len(out_evs),
        avg_duration=(sum(e["dur"] for e in out_evs) / len(out_evs)) if out_evs else 0.0,
    )

    due_by_job = {o["job"]: o["due"] for o in ops if o["kind"] == "asm"}
    jobs_kpi = {}
    for j in jobs:
        ev = next(e for e in events if e["kind"] == "asm" and e["job"] == j)
        tardiness = max(0, ev["end"] - due_by_job[j])
        jobs_kpi[j] = dict(completion=ev["end"], due=due_by_job[j],
                            tardiness=tardiness, on_time=(tardiness == 0))

    n_res = len(resource_kpis) or 1
    overall = dict(
        total_production_time=makespan,
        avg_waiting_time=(sum(e["wait"] for e in events) / len(events)) if events else 0.0,
        avg_queue_length=sum(k["avg_queue_length"] for k in resource_kpis.values()) / n_res,
        avg_utilisation_pct=sum(k["utilisation_pct"] for k in resource_kpis.values()) / n_res,
        avg_idle_time=sum(k["idle_time"] for k in resource_kpis.values()) / n_res,
        total_tardiness=sum(v["tardiness"] for v in jobs_kpi.values()),
        jobs_on_time=sum(1 for v in jobs_kpi.values() if v["on_time"]),
        jobs_total=len(jobs),
    )

    return dict(events=events, makespan=makespan, machines=machines,
                resources=resource_kpis, outsourced=outsourced_kpi,
                jobs=jobs_kpi, overall=overall)
    
    
    
    

# ================================================================ animation page ====
def _prep_animation_data(ops, sim_result):
    jobs = sorted({o["job"] for o in ops})
    machines = sim_result["machines"]
    comp_labels = sorted({o["comp"] for o in ops if o["kind"] == "comp"}, key=_natural_key)
    # Which component labels each job actually has -- the staging grid should only
    # draw a slot for these, not one for every label that exists anywhere across
    # the whole dataset (a product with 3 components was showing 6 slots before).
    comps_by_job = {
        j: sorted({o["comp"] for o in ops if o["job"] == j and o["kind"] == "comp"}, key=_natural_key)
        for j in jobs
    }
    due_by_job = {o["job"]: o["due"] for o in ops if o["kind"] == "asm"}
    color_by_job = {j: JOB_PALETTE[(j - 1) % len(JOB_PALETTE)] for j in jobs}
    return dict(
        events=sim_result["events"], makespan=sim_result["makespan"],
        machines=machines, asmName=core.ASM_STATION, jobs=jobs,
        compLabels=comp_labels, compsByJob=comps_by_job,
        dueByJob=due_by_job, colorByJob=color_by_job,
        minPerDay=core.MIN_PER_DAY,
    )


def _kpi_skeleton_html(sim_result, ops):
    """Static skeleton only -- labels, tips, table structure, and one <span id=...>
    per numeric value. No numbers are computed here. Every value is filled in and
    continuously recomputed by the JS liveSummary()/renderKpis() functions as the
    simulation plays, restricted to events up to the current playhead time -- same
    principle as the small 't=... completed X/Y' readout, just extended to cover
    every KPI so the whole panel genuinely reflects "so far", not the final answer
    shown immediately."""
    resource_names = sim_result["machines"] + [core.ASM_STATION]
    rows_html = "".join(
        f"<tr><td class='rname'>{name}</td>"
        f"<td id='kpi-r-{name}-ops'>&ndash;</td>"
        f"<td id='kpi-r-{name}-wait'>&ndash;</td>"
        f"<td id='kpi-r-{name}-qlen'>&ndash;</td>"
        f"<td id='kpi-r-{name}-maxq'>&ndash;</td>"
        f"<td id='kpi-r-{name}-util'>&ndash;</td>"
        f"<td id='kpi-r-{name}-prod'>&ndash;</td>"
        f"<td id='kpi-r-{name}-idle'>&ndash;</td></tr>"
        for name in resource_names
    )
    cards = [
        ("elapsed", "Production time (elapsed)", "wall-clock progress through the simulation so far"),
        ("avgwait", "Avg waiting time (so far)", "mean queue delay across operations that have started"),
        ("avgqlen", "Avg queue length (so far)", "mean of each resource's time-weighted queue length, up to now"),
        ("avgutil", "Avg utilisation (so far)", "mean busy-time share across machines + ASM, up to now"),
        ("avgidle", "Avg idle time (so far)", "mean idle time across machines + ASM, up to now"),
        ("tardiness", "Total tardiness (so far)", "accrued only from products that have actually completed"),
    ]
    cards_html = "".join(
        f"<div class='kcard'><div class='kval' id='kpi-{key}'>&ndash;</div>"
        f"<div class='klabel'>{label}</div><div class='ktip'>{tip}</div></div>"
        for key, label, tip in cards
    )
    return f"""
    <div class="kpi-cards">{cards_html}</div>
    <table class="kpi-table">
      <thead><tr><th>Resource</th><th>Ops (done/total)</th><th>Avg wait</th><th>Avg queue len</th>
      <th>Max queue len</th><th>Utilisation</th><th>Production time</th><th>Idle time</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="kpi-note">Outsourced legs returned so far: <span id="kpi-out-returned">&ndash;</span>
    &mdash; not shown in the table above as they hold no shared capacity and never queue.
    Every figure on this panel is recomputed live at the current simulation time directly from
    the discrete-event log (arrival/start/end of each operation) &mdash; nothing here is a final
    number shown early. Scrub or play to watch it evolve.</p>
    """


_PAGE_CSS = """
:root{
  --bg:#0f1420; --panel:#161d2c; --panel2:#1b2334; --border:#2a3549;
  --text:#e7ecf5; --muted:#8a96ad; --idle:#4a5568; --busy:#38bdf8;
  --waiting:#f5b942; --done:#4ade80; --danger:#ef5350;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.wrap{max-width:1600px;margin:0 auto;padding:18px 22px 40px;}
.hdr{display:flex;align-items:baseline;justify-content:space-between;
  border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:14px;flex-wrap:wrap;gap:8px;}
.hdr h1{font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0;font-family:ui-monospace,monospace;}
.hdr .opt{font-size:20px;color:var(--text);font-weight:600;margin-top:2px;}
.hdr .sub{color:var(--muted);font-size:12px;}
.controls{display:flex;align-items:center;gap:10px;background:var(--panel);
  border:1px solid var(--border);border-radius:6px;padding:10px 14px;margin-bottom:14px;flex-wrap:wrap;}
button.ctl{background:var(--panel2);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:6px 12px;font-family:inherit;font-size:12px;cursor:pointer;}
button.ctl:hover{border-color:var(--busy);color:var(--busy);}
button.ctl.active{border-color:var(--busy);color:var(--busy);background:#132433;}
.controls input[type=range]{accent-color:var(--busy);}
.controls .readout{margin-left:auto;font-size:12px;color:var(--muted);white-space:nowrap;}
.controls .readout b{color:var(--text);}
.scrubrow{display:flex;align-items:center;gap:10px;width:100%;margin-top:2px;}
.scrubrow input[type=range]{flex:1;}
.stage-panel{background:var(--panel);border:1px solid var(--border);border-radius:6px;
  padding:10px;overflow-x:auto;}
canvas{display:block;}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--muted);
  padding:10px 4px 2px;}
.legend span{display:inline-flex;align-items:center;gap:5px;}
.legend i{width:9px;height:9px;border-radius:2px;display:inline-block;}
.kpi-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin:18px 0 12px;}
.kcard{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:12px 14px;}
.kval{font-size:19px;font-weight:600;color:var(--busy);}
.klabel{font-size:11px;color:var(--text);margin-top:2px;}
.ktip{font-size:10px;color:var(--muted);margin-top:4px;line-height:1.35;}
table.kpi-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;}
table.kpi-table th{text-align:left;color:var(--muted);font-weight:500;
  border-bottom:1px solid var(--border);padding:6px 10px;}
table.kpi-table td{padding:6px 10px;border-bottom:1px solid #1e2739;}
table.kpi-table td.rname{color:var(--busy);font-weight:600;}
.kpi-note{font-size:11px;color:var(--muted);margin-top:10px;line-height:1.5;}
.section-title{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin:22px 0 8px;}
"""

_PAGE_JS = r"""
const ENTER_MS = 40, EXIT_MS = 40, MAX_QUEUE_SHOWN = 6, MAX_CONCURRENT_SHOWN = 5;

function buildIndex(DATA){
  const byResource = {};
  const rows = ['OutSrc'].concat(DATA.machines);
  rows.concat([DATA.asmName]).forEach(r => { byResource[r] = []; });
  DATA.events.forEach(e => { (byResource[e.resource] = byResource[e.resource] || []).push(e); });
  Object.keys(byResource).forEach(r => byResource[r].sort((a,b) => a.start - b.start));
  const compEventsByJob = {}, asmEventByJob = {};
  DATA.jobs.forEach(j => { compEventsByJob[j] = []; });
  DATA.events.forEach(e => {
    if (e.kind === 'comp') compEventsByJob[e.job].push(e);
    else if (e.kind === 'asm') asmEventByJob[e.job] = e;
  });
  return { byResource, rows, compEventsByJob, asmEventByJob };
}

function rowStateAt(evs, t){
  const current = evs.filter(e => e.start <= t && t < e.end);
  // Full future queue for this resource is already known (the sim already ran to
  // completion before playback starts) -- show ALL of it from t=0 as placeholders,
  // not just the operations that have already arrived. Arrived-but-not-yet-started
  // ops render bright (genuinely waiting their turn); not-yet-arrived ops render
  // faint/outline, same convention the staging grid uses for "not ready yet".
  const upcoming = evs.filter(e => t < e.start).sort((a,b) => a.start - b.start);
  const arrived = upcoming.filter(e => e.arrival <= t);
  const notArrived = upcoming.filter(e => e.arrival > t);
  const allDone = evs.length > 0 && evs.every(e => e.end <= t);
  let boxState;
  if (current.length > 0) boxState = 'busy';
  else if (allDone) boxState = 'done';
  else if (arrived.length > 0) boxState = 'waiting';
  else boxState = 'idle';
  return { boxState, current,
           arrived: arrived.slice(0, MAX_QUEUE_SHOWN),
           arrivedOverflow: Math.max(0, arrived.length - MAX_QUEUE_SHOWN),
           notArrived: notArrived.slice(0, MAX_QUEUE_SHOWN),
           notArrivedOverflow: Math.max(0, notArrived.length - MAX_QUEUE_SHOWN) };
}

function stateAt(DATA, idx, t){
  const state = { simTime:t, rows:{}, asm:null, staging:[], completed:[], entering:[], exiting:[] };
  idx.rows.forEach(name => { state.rows[name] = rowStateAt(idx.byResource[name] || [], t); });
  state.asm = rowStateAt(idx.byResource[DATA.asmName] || [], t);

  let completedCount=0, onTimeCount=0, tardinessSoFar=0;
  DATA.jobs.forEach(job => {
    const asmEv = idx.asmEventByJob[job];
    idx.compEventsByJob[job].forEach(ce => {
      const ready = t >= ce.end;
      const consumed = asmEv ? t >= asmEv.start : false;
      if (ready && !consumed) state.staging.push({ job, comp: ce.comp });
      if (ready && !consumed && t < ce.end + ENTER_MS)
        state.entering.push({ job, comp: ce.comp, frac: Math.min(1,(t-ce.end)/ENTER_MS), from: ce.resource });
    });
    if (asmEv && asmEv.end <= t){
      completedCount++;
      const due = DATA.dueByJob[job];
      const tardy = Math.max(0, asmEv.end - due);
      if (tardy === 0) onTimeCount++;
      tardinessSoFar += tardy;
      if (t < asmEv.end + EXIT_MS) state.exiting.push({ job, frac: Math.min(1,(t-asmEv.end)/EXIT_MS) });
      else state.completed.push({ job });
    }
  });
  state.live = { completed:completedCount, onTime:onTimeCount, total:DATA.jobs.length, tardinessSoFar };
  return state;
}

function computeLayout(DATA){
  const rows = ['OutSrc'].concat(DATA.machines);
  const ROW_H = 150, ROWS_TOP = 44, N_ROWS = rows.length;
  // Same sizing principle as the ASM gap below: must fit 2*MAX_QUEUE_SHOWN triangles.
  const QUEUE_STEP = 24;
  const QUEUE_X_END = 2 * MAX_QUEUE_SHOWN * QUEUE_STEP + 34;
  const BOX_X0 = QUEUE_X_END + 12, BOX_X1 = BOX_X0 + 220, BOX_H = 56;
  // OutSrc can have many concurrent items in transit (no shared capacity there,
  // unlike a machine which only ever processes one) -- give it a taller box so
  // stacked markers don't spill outside it. ROW_H above was increased to match,
  // so OutSrc's taller box doesn't collide with the row below it.
  const OUTSRC_BOX_H = 130;
  const STAGE_X0 = BOX_X1 + 64, STAGE_LABEL_W = 60, STAGE_COL_W = 48;
  const nCols = DATA.compLabels.length;
  const STAGE_X1 = STAGE_X0 + STAGE_LABEL_W + nCols * STAGE_COL_W;
  const rowsBottom = ROWS_TOP + N_ROWS * ROW_H;
  // Staging rows scale to fill the SAME total vertical space as the machine-row
  // column, instead of being capped at a fixed pixel budget -- previously this
  // left a lot of unused canvas below a short product list.
  const STAGE_ROW_H = Math.max(32, Math.min(100, (rowsBottom - ROWS_TOP) / DATA.jobs.length));
  const stageBottom = ROWS_TOP + DATA.jobs.length * STAGE_ROW_H;
  // Gap must fit up to 2*MAX_QUEUE_SHOWN queue triangles (arrived + not-yet-arrived)
  // plus margins on both sides -- sized explicitly rather than guessed, since the
  // ASM queue can legitimately run into double digits under a disruption.
  const ASM_GAP = 2 * MAX_QUEUE_SHOWN * QUEUE_STEP + 50;
  const ASM_X0 = STAGE_X1 + ASM_GAP, ASM_X1 = ASM_X0 + 190, ASM_W = ASM_X1 - ASM_X0, ASM_H = 150;
  const ASM_Y = ROWS_TOP + (N_ROWS * ROW_H) / 2;
  const DONE_X0 = ASM_X1 + 70, DONE_COLS = 4, DONE_COL_W = 64, DONE_ROW_H = 64;
  const doneRows = Math.ceil(DATA.jobs.length / DONE_COLS);
  const W = DONE_X0 + DONE_COLS * DONE_COL_W + 30;
  const H = Math.max(rowsBottom, stageBottom, ASM_Y + ASM_H, ROWS_TOP + doneRows * DONE_ROW_H) + 40;
  return { rows, ROW_H, ROWS_TOP, N_ROWS, QUEUE_X_END, QUEUE_STEP, BOX_X0, BOX_X1, BOX_H, OUTSRC_BOX_H,
           STAGE_X0, STAGE_LABEL_W, STAGE_COL_W, STAGE_X1, STAGE_ROW_H, nCols,
           ASM_X0, ASM_X1, ASM_W, ASM_H, ASM_Y, DONE_X0, DONE_COLS, DONE_COL_W, DONE_ROW_H, W, H };
}

const STATE_COLOR = { idle:'#4a5568', busy:'#38bdf8', waiting:'#f5b942', done:'#4ade80' };

function draw(ctx, DATA, idx, geo, t){
  const state = stateAt(DATA, idx, t);
  const W = geo.W, H = geo.H;
  ctx.clearRect(0,0,W,H);
  ctx.font = '13px ui-monospace, monospace';

  // ---- machine + outsourced rows ----
  geo.rows.forEach((name, ri) => {
    const y = geo.ROWS_TOP + ri * geo.ROW_H + geo.ROW_H/2;
    const rs = state.rows[name];
    rs.arrived.forEach((e, k) => {
      const x = geo.QUEUE_X_END - k * geo.QUEUE_STEP;
      drawTriangle(ctx, x, y, 9, DATA.colorByJob[e.job]);
      ctx.fillStyle = '#0f1420'; ctx.font = 'bold 9px ui-monospace, monospace';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('P'+e.job, x, y+1);
    });
    const notArrivedBase = geo.QUEUE_X_END - rs.arrived.length * geo.QUEUE_STEP;
    rs.notArrived.forEach((e, k) => {
      const x = notArrivedBase - k * geo.QUEUE_STEP;
      drawTriangle(ctx, x, y, 6.5, 'transparent', '#33405c');
    });
    const overflowN = rs.arrivedOverflow + rs.notArrivedOverflow;
    if (overflowN > 0){
      const x = notArrivedBase - rs.notArrived.length * geo.QUEUE_STEP - 22;
      ctx.fillStyle = '#8a96ad'; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign='left';
      ctx.fillText('+' + overflowN, Math.max(2, x), y+4);
    }
    const boxH = name === 'OutSrc' ? geo.OUTSRC_BOX_H : geo.BOX_H;
    const col = STATE_COLOR[rs.boxState];
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    roundRect(ctx, geo.BOX_X0, y-boxH/2, geo.BOX_X1-geo.BOX_X0, boxH, 7);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#e7ecf5'; ctx.font='bold 16px ui-monospace, monospace';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(name === 'OutSrc' ? 'Outsourced' : name, geo.BOX_X0+12, y-boxH/2+18);
    ctx.fillStyle = col; ctx.font='13px ui-monospace, monospace';
    ctx.fillText(rs.boxState, geo.BOX_X0+12, y+boxH/2-14);
    const shownCurrent = rs.current.slice(0, MAX_CONCURRENT_SHOWN);
    const curOverflow = Math.max(0, rs.current.length - MAX_CONCURRENT_SHOWN);
    const spacing = shownCurrent.length > 1 ? Math.min(22, (boxH - 40) / (shownCurrent.length - 1)) : 0;
    shownCurrent.forEach((e, k) => {
      const frac = Math.min(1, Math.max(0, (t - e.start) / Math.max(1, e.end - e.start)));
      const bx = geo.BOX_X0 + 46 + frac * (geo.BOX_X1 - geo.BOX_X0 - 92);
      const off = shownCurrent.length > 1 ? (k - (shownCurrent.length-1)/2) * spacing : 0;
      drawDiamond(ctx, bx, y+off, 12, DATA.colorByJob[e.job]);
      ctx.fillStyle='#0f1420'; ctx.font='bold 10px ui-monospace, monospace';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('P'+e.job+e.comp.replace(/^C?/,'C').slice(0,3), bx, y+off);
    });
    if (curOverflow > 0){
      ctx.fillStyle = col; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign='center';
      ctx.fillText('+' + curOverflow + ' more in transit', geo.BOX_X0 + (geo.BOX_X1-geo.BOX_X0)/2, y);
    }
    drawArrow(ctx, geo.BOX_X1+8, y, geo.STAGE_X0-8, y);
  });

  // ---- staging grid ----
  ctx.textAlign='left'; ctx.fillStyle='#8a96ad'; ctx.font='12px ui-monospace, monospace';
  ctx.fillText('WAITING FOR ASSEMBLY', geo.STAGE_X0, geo.ROWS_TOP - 14);
  DATA.compLabels.forEach((label, ci) => {
    const x = geo.STAGE_X0 + geo.STAGE_LABEL_W + ci*geo.STAGE_COL_W + geo.STAGE_COL_W/2;
    ctx.textAlign='center'; ctx.fillStyle='#7c8aa3'; ctx.font='11px ui-monospace, monospace';
    ctx.fillText(label, x, geo.ROWS_TOP - 2);
  });
  const stagingSet = {};
  state.staging.forEach(s => { stagingSet[s.job+'|'+s.comp] = true; });
  DATA.jobs.forEach((job, ji) => {
    const y = geo.ROWS_TOP + ji*geo.STAGE_ROW_H + geo.STAGE_ROW_H/2;
    ctx.fillStyle = DATA.colorByJob[job]; ctx.font='bold 13px ui-monospace, monospace';
    ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText('P'+job, geo.STAGE_X0, y);
    const jobComps = DATA.compsByJob[job] || [];
    DATA.compLabels.forEach((label, ci) => {
      if (!jobComps.includes(label)) return;
      const x = geo.STAGE_X0 + geo.STAGE_LABEL_W + ci*geo.STAGE_COL_W + geo.STAGE_COL_W/2;
      const ready = stagingSet[job+'|'+label];
      drawTriangle(ctx, x, y, ready ? 10 : 7, ready ? DATA.colorByJob[job] : 'transparent',
                   ready ? null : '#33405c');
      if (ready){
        ctx.fillStyle='#0f1420'; ctx.font='bold 8px ui-monospace, monospace';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(label, x, y+1);
      }
    });
  });
  state.entering.forEach(en => {
    const ci = DATA.compLabels.indexOf(en.comp);
    if (ci < 0) return;
    const ji = DATA.jobs.indexOf(en.job);
    const toX = geo.STAGE_X0 + geo.STAGE_LABEL_W + ci*geo.STAGE_COL_W + geo.STAGE_COL_W/2;
    const toY = geo.ROWS_TOP + ji*geo.STAGE_ROW_H + geo.STAGE_ROW_H/2;
    const ri = Math.max(0, geo.rows.indexOf(en.from));
    const fromX = geo.BOX_X1, fromY = geo.ROWS_TOP + ri*geo.ROW_H + geo.ROW_H/2;
    const x = fromX + en.frac*(toX-fromX), y = fromY + en.frac*(toY-fromY);
    drawTriangle(ctx, x, y, 9, DATA.colorByJob[en.job]);
  });

  // ---- ASM ----
  const asmQueueXEnd = geo.ASM_X0 - 26;
  state.asm.arrived.forEach((e, k) => {
    const x = asmQueueXEnd - k * geo.QUEUE_STEP;
    drawTriangle(ctx, x, geo.ASM_Y, 9, DATA.colorByJob[e.job]);
    ctx.fillStyle = '#0f1420'; ctx.font = 'bold 9px ui-monospace, monospace';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('P'+e.job, x, geo.ASM_Y+1);
  });
  const asmNotArrivedBase = asmQueueXEnd - state.asm.arrived.length * geo.QUEUE_STEP;
  state.asm.notArrived.forEach((e, k) => {
    const x = asmNotArrivedBase - k * geo.QUEUE_STEP;
    drawTriangle(ctx, x, geo.ASM_Y, 6.5, 'transparent', '#33405c');
  });
  const asmOverflowN = state.asm.arrivedOverflow + state.asm.notArrivedOverflow;
  if (asmOverflowN > 0){
    const x = asmNotArrivedBase - state.asm.notArrived.length * geo.QUEUE_STEP - 22;
    ctx.fillStyle = '#8a96ad'; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign='left';
    ctx.fillText('+' + asmOverflowN, Math.max(geo.STAGE_X1+48, x), geo.ASM_Y+4);
  }
  drawArrow(ctx, geo.STAGE_X1+8, geo.ASM_Y, geo.STAGE_X1+48, geo.ASM_Y);
  const acol = STATE_COLOR[state.asm.boxState];
  ctx.strokeStyle = acol; ctx.lineWidth = 2.5; ctx.fillStyle='rgba(255,255,255,0.03)';
  roundRect(ctx, geo.ASM_X0, geo.ASM_Y-geo.ASM_H/2, geo.ASM_W, geo.ASM_H, 9);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle='#e7ecf5'; ctx.font='bold 19px ui-monospace, monospace';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('ASM', geo.ASM_X0+geo.ASM_W/2, geo.ASM_Y-geo.ASM_H/2+22);
  ctx.fillStyle=acol; ctx.font='13px ui-monospace, monospace';
  ctx.fillText(state.asm.boxState, geo.ASM_X0+geo.ASM_W/2, geo.ASM_Y+geo.ASM_H/2-16);
  state.asm.current.forEach(e => {
    const frac = Math.min(1, Math.max(0, (t - e.start)/Math.max(1, e.end-e.start)));
    const ax = geo.ASM_X0 + 28 + frac*(geo.ASM_W-56);
    drawDiamond(ctx, ax, geo.ASM_Y, 14, DATA.colorByJob[e.job]);
    ctx.fillStyle='#0f1420'; ctx.font='bold 11px ui-monospace, monospace';
    ctx.fillText('P'+e.job, ax, geo.ASM_Y);
  });

  // ---- completed grid ----
  drawArrow(ctx, geo.ASM_X1+8, geo.ASM_Y, geo.DONE_X0-8, geo.ASM_Y);
  ctx.fillStyle='#8a96ad'; ctx.font='12px ui-monospace, monospace'; ctx.textAlign='left';
  ctx.fillText('COMPLETED', geo.DONE_X0, geo.ROWS_TOP - 14);
  state.completed.forEach((c, i) => {
    const col = i % geo.DONE_COLS, row = Math.floor(i/geo.DONE_COLS);
    const x = geo.DONE_X0 + col*geo.DONE_COL_W + geo.DONE_COL_W/2;
    const y = geo.ROWS_TOP + row*geo.DONE_ROW_H + geo.DONE_ROW_H/2;
    drawCircle(ctx, x, y, 18, DATA.colorByJob[c.job]);
    ctx.fillStyle='#0f1420'; ctx.font='bold 12px ui-monospace, monospace';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('P'+c.job, x, y);
  });
  state.exiting.forEach((ex, i) => {
    const slotIndex = state.completed.length + i;
    const col = slotIndex % geo.DONE_COLS, row = Math.floor(slotIndex/geo.DONE_COLS);
    const toX = geo.DONE_X0 + col*geo.DONE_COL_W + geo.DONE_COL_W/2;
    const toY = geo.ROWS_TOP + row*geo.DONE_ROW_H + geo.DONE_ROW_H/2;
    const fromX = geo.ASM_X1, fromY = geo.ASM_Y;
    const x = fromX + ex.frac*(toX-fromX), y = fromY + ex.frac*(toY-fromY);
    drawCircle(ctx, x, y, 18, DATA.colorByJob[ex.job]);
    ctx.fillStyle='#0f1420'; ctx.font='bold 12px ui-monospace, monospace';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('P'+ex.job, x, y);
  });

  return state.live;
}

function drawTriangle(ctx, x, y, r, fill, strokeOverride){
  ctx.beginPath();
  ctx.moveTo(x, y-r); ctx.lineTo(x+r, y+r*0.8); ctx.lineTo(x-r, y+r*0.8); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.strokeStyle = strokeOverride || '#0f1420'; ctx.lineWidth = 1; ctx.stroke();
}
function drawDiamond(ctx, x, y, r, fill){
  ctx.beginPath();
  ctx.moveTo(x, y-r); ctx.lineTo(x+r, y); ctx.lineTo(x, y+r); ctx.lineTo(x-r, y); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle='#0f1420'; ctx.lineWidth=1; ctx.stroke();
}
function drawCircle(ctx, x, y, r, fill){
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
  ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle='#0f1420'; ctx.lineWidth=1; ctx.stroke();
}
function roundRect(ctx, x, y, w, h, r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}
function drawArrow(ctx, x0, y, x1, y2){
  ctx.strokeStyle = '#3a4763'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1-5,y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1,y); ctx.lineTo(x1-6,y-4); ctx.lineTo(x1-6,y+4); ctx.closePath();
  ctx.fillStyle='#3a4763'; ctx.fill();
}

// ---- live KPIs: every number here is restricted to events with start/end/arrival
// <= t, i.e. only things that have actually happened by the current playhead. This
// mirrors adrs_simulation.py's _summarise()/_time_weighted_avg() logic exactly, but
// with the integration horizon capped at t instead of the full makespan -- so at
// t=0 everything reads zero/blank, and the numbers only ever grow in as playback
// reaches the point where each operation genuinely occurs.
function timeWeightedAvgUpTo(marks, t){
  const relevant = marks.filter(m => m[0] <= t).sort((a,b) => a[0]-b[0]);
  let area = 0, running = 0, last = 0;
  for (const [mt, d] of relevant){ area += running * (mt - last); running += d; last = mt; }
  area += running * (t - last);
  return t > 0 ? area / t : 0;
}
function maxQueueUpTo(marks, t){
  const relevant = marks.filter(m => m[0] <= t).sort((a,b) => a[0]-b[0]);
  let running = 0, max = 0;
  for (const [, d] of relevant){ running += d; if (running > max) max = running; }
  return max;
}
function queueMarksForResource(evs){
  const marks = [];
  evs.forEach(e => { marks.push([e.arrival, 1]); marks.push([e.start, -1]); });
  return marks;
}

function liveSummary(DATA, idx, t){
  const resourceNames = DATA.machines.concat([DATA.asmName]);
  const resources = {};
  let sumQ = 0, sumUtil = 0, sumIdle = 0, nRes = 0;
  resourceNames.forEach(name => {
    const evs = idx.byResource[name] || [];
    const started = evs.filter(e => e.start <= t);
    const completed = evs.filter(e => e.end <= t);
    const busy = started.reduce((s, e) => s + (Math.min(t, e.end) - e.start), 0);
    const waits = started.map(e => e.wait);
    const marks = queueMarksForResource(evs);
    const avgQ = timeWeightedAvgUpTo(marks, t);
    const maxQ = maxQueueUpTo(marks, t);
    const util = t > 0 ? (busy / t * 100) : 0;
    const idle = Math.max(0, t - busy);
    resources[name] = { nOps: evs.length, nCompleted: completed.length,
      avgWait: waits.length ? waits.reduce((a,b)=>a+b,0)/waits.length : 0,
      avgQueueLength: avgQ, maxQueueLength: maxQ,
      utilisationPct: util, productionTime: busy, idleTime: idle };
    sumQ += avgQ; sumUtil += util; sumIdle += idle; nRes++;
  });

  let completedJobs = 0, onTimeJobs = 0, tardSoFar = 0;
  DATA.jobs.forEach(job => {
    const asmEv = idx.asmEventByJob[job];
    if (asmEv && asmEv.end <= t){
      completedJobs++;
      const tardy = Math.max(0, asmEv.end - DATA.dueByJob[job]);
      if (tardy === 0) onTimeJobs++;
      tardSoFar += tardy;
    }
  });

  // overall avg waiting time = mean wait across EVERY started operation, including
  // outsourced legs (which always wait 0, no shared capacity) -- matches Python's
  // _summarise() definition exactly, so this is NOT just an average of the
  // per-resource averages above.
  const allStarted = DATA.events.filter(e => e.start <= t);
  const overallAvgWait = allStarted.length
    ? allStarted.reduce((s, e) => s + e.wait, 0) / allStarted.length : 0;

  const outEvs = DATA.events.filter(e => e.outsourced && e.end <= t);

  return {
    resources,
    outReturned: outEvs.length,
    overall: {
      elapsed: t,
      avgWaitingTime: overallAvgWait,
      avgQueueLength: nRes ? sumQ / nRes : 0,
      avgUtilisationPct: nRes ? sumUtil / nRes : 0,
      avgIdleTime: nRes ? sumIdle / nRes : 0,
      totalTardiness: tardSoFar,
      jobsOnTime: onTimeJobs, jobsCompleted: completedJobs, jobsTotal: DATA.jobs.length,
    },
  };
}

function initSim(rootId, DATA){
  const root = document.getElementById(rootId);
  const canvas = root.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const idx = buildIndex(DATA);
  const geo = computeLayout(DATA);
  canvas.width = geo.W; canvas.height = geo.H;
  canvas.style.width = geo.W + 'px'; canvas.style.height = geo.H + 'px';

  let simTime = 0, playing = false, lastTs = null;
  const baseSpeed = Math.max(1, DATA.makespan / 90); // sim-minutes per real-second, ~90s full run at 1x
  let speedMult = 0.5; // matches the .btn-speed marked 'active' in the HTML below -- was
                        // hardcoded to 1 before, silently ignoring which button looked selected

  const playBtn = root.querySelector('.btn-play');
  const pauseBtn = root.querySelector('.btn-pause');
  const resetBtn = root.querySelector('.btn-reset');
  const speedBtns = root.querySelectorAll('.btn-speed');
  const scrub = root.querySelector('.scrub');
  const readout = root.querySelector('.readout');

  function fmtWd(mins){ return (mins / DATA.minPerDay).toFixed(1) + 'wd'; }

  function renderKpis(summary){
    const o = summary.overall;
    const setText = (id, val) => { const el = root.querySelector('#'+id); if (el) el.textContent = val; };
    setText('kpi-elapsed', fmtWd(o.elapsed));
    setText('kpi-avgwait', fmtWd(o.avgWaitingTime));
    setText('kpi-avgqlen', o.avgQueueLength.toFixed(2));
    setText('kpi-avgutil', o.avgUtilisationPct.toFixed(0) + '%');
    setText('kpi-avgidle', fmtWd(o.avgIdleTime));
    setText('kpi-tardiness', fmtWd(o.totalTardiness) + ' (' + o.jobsCompleted + '/' + o.jobsTotal + ' done, ' +
      o.jobsOnTime + ' on-time)');
    setText('kpi-out-returned', summary.outReturned);
    Object.keys(summary.resources).forEach(name => {
      const r = summary.resources[name];
      setText('kpi-r-'+name+'-ops', r.nCompleted + '/' + r.nOps);
      setText('kpi-r-'+name+'-wait', fmtWd(r.avgWait));
      setText('kpi-r-'+name+'-qlen', r.avgQueueLength.toFixed(2));
      setText('kpi-r-'+name+'-maxq', r.maxQueueLength);
      setText('kpi-r-'+name+'-util', r.utilisationPct.toFixed(0) + '%');
      setText('kpi-r-'+name+'-prod', fmtWd(r.productionTime));
      setText('kpi-r-'+name+'-idle', fmtWd(r.idleTime));
    });
  }

  function render(){
    const live = draw(ctx, DATA, idx, geo, simTime);
    scrub.value = simTime;
    readout.innerHTML = 't=<b>' + fmtWd(simTime) + '</b> &nbsp; completed <b>' +
      live.completed + '/' + live.total + '</b> (' + live.onTime + ' on-time) &nbsp; tardiness so far <b>' +
      fmtWd(live.tardinessSoFar) + '</b>';
    renderKpis(liveSummary(DATA, idx, simTime));
  }

  function frame(ts){
    if (playing){
      if (lastTs != null){
        const dtReal = (ts - lastTs) / 1000;
        simTime = Math.min(DATA.makespan, simTime + dtReal * baseSpeed * speedMult);
        if (simTime >= DATA.makespan) playing = false;
      }
      lastTs = ts;
    } else { lastTs = ts; }
    render();
    requestAnimationFrame(frame);
  }

  playBtn.onclick = () => { playing = true; };
  pauseBtn.onclick = () => { playing = false; };
  resetBtn.onclick = () => { playing = false; simTime = 0; };
  speedBtns.forEach(b => b.onclick = () => {
    speedBtns.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    speedMult = parseFloat(b.dataset.speed);
  });
  scrub.max = DATA.makespan;
  scrub.oninput = () => { playing = false; simTime = parseFloat(scrub.value); render(); };

  render();
  requestAnimationFrame(frame);
}
"""


def build_animation_page(ops, plan, actual_returns=None, option_label="", sim_result=None):
    """The full standalone HTML page: canvas animation + KPI tables. Self-contained
    (no external requests, no CDN) so it works when opened as a blob: URL with no
    server behind it."""
    actual_returns = actual_returns or {}
    if sim_result is None:
        sim_result = simulate(ops, plan, actual_returns)

    data = _prep_animation_data(ops, sim_result)
    data_json = json.dumps(data)
    kpi_html = _kpi_skeleton_html(sim_result, ops)
    n_disrupted = sum(1 for i, d in actual_returns.items() if d != ops[i]["dur"])
    subtitle = (f"{n_disrupted} outsourced return(s) deviating from plan"
                if n_disrupted else "no disruption applied — nominal plan")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ADRS Simulation — {option_label}</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="wrap" id="sim-root">
  <div class="hdr">
    <div><h1>ADRS &middot; discrete-event simulation</h1>
      <div class="opt">{option_label}</div>
      <div class="sub">{subtitle} &middot; makespan {core.fmt_wd(data['makespan'])}</div>
    </div>
  </div>

  <div class="controls">
    <button class="ctl btn-play">&#9654; Play</button>
    <button class="ctl btn-pause">&#9208; Pause</button>
    <button class="ctl btn-reset">&#8634; Reset</button>
    <button class="ctl btn-speed" data-speed="0.25">0.25&times;</button>
    <button class="ctl btn-speed active" data-speed="0.5">0.5&times;</button>
    <button class="ctl btn-speed" data-speed="1">1&times;</button>
    <button class="ctl btn-speed" data-speed="2">2&times;</button>
    <button class="ctl btn-speed" data-speed="5">5&times;</button>
    <span class="readout"></span>
    <div class="scrubrow"><input class="scrub" type="range" min="0" value="0" step="1"></div>
  </div>

  <div class="stage-panel">
    <canvas></canvas>
    <div class="legend">
      <span><i style="background:#4a5568"></i> idle</span>
      <span><i style="background:#38bdf8"></i> busy</span>
      <span><i style="background:#f5b942"></i> waiting</span>
      <span><i style="background:#4ade80"></i> done</span>
      <span>&#9650; component &nbsp; &#9670; in service &nbsp; &#9679; completed product</span>
    </div>
  </div>

  <div class="section-title">Simulation KPIs (measured, not read off the plan)</div>
  {kpi_html}
</div>
<script>
{_PAGE_JS}
initSim('sim-root', {data_json});
</script>
</body></html>"""


def build_launch_widget(ops, plan, actual_returns=None, option_label="",
                          sim_result=None, button_label="\U0001F3AC Launch Simulation"):
    """A small HTML/JS snippet (a single button) suitable for
    st.components.v1.html(...). The full animation page is base64-embedded and
    opened in a NEW browser tab via a Blob URL when the button is clicked --
    this keeps the animation's own render loop completely isolated from
    Streamlit's rerun cycle, and the Blob approach (rather than a data: URL)
    avoids the top-level data: navigation block some browsers apply to link
    clicks."""
    page = build_animation_page(ops, plan, actual_returns, option_label, sim_result)
    b64 = base64.b64encode(page.encode("utf-8")).decode("ascii")
    uid = f"simlaunch_{abs(hash(option_label)) % 100000}"
    return f"""
    <div style="font-family:ui-sans-serif,system-ui,sans-serif;">
    <button id="{uid}" style="background:#161d2c;color:#e7ecf5;border:1px solid #2a3549;
      border-radius:6px;padding:9px 16px;font-size:14px;cursor:pointer;">
      {button_label}
    </button>
    </div>
    <script>
    (function(){{
      function b64DecodeUnicode(str) {{
        return decodeURIComponent(Array.prototype.map.call(atob(str), function(c) {{
          return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }}).join(''));
      }}
      document.getElementById('{uid}').addEventListener('click', function(){{
        const html = b64DecodeUnicode('{b64}');
        const blob = new Blob([html], {{type: 'text/html'}});
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
      }});
    }})();
    </script>
    """