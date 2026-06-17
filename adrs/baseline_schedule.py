"""
ADRS baseline (nominal) scheduler.
Job = Sr. No. Job done when all its components finish.
Each component -> one operation -> one dedicated machine (M1..M6), unit capacity,
non-preemptive. Outsourced components have a RANDOM assumed lead time.
Calendar: weekdays only, 09:00-17:00 (8h/day). Objective: min total tardiness
(primary), then makespan (secondary).
"""
import math, random
import pandas as pd
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

# ---------- config ----------
SEED = 42
LEAD_MIN_DAYS, LEAD_MAX_DAYS = 3, 10   # random outsource lead time range (working days)
WORK_START_H, WORK_END_H = 9, 17
MIN_PER_DAY = (WORK_END_H - WORK_START_H) * 60   # 480
random.seed(SEED)

# ---------- working-time calendar (weekdays only) ----------
ORIGIN = datetime(2024, 8, 21, 9, 0)   # release of all orders (a Wednesday)

def to_work_minutes(dt):
    """Working minutes (Mon-Fri, 09:00-17:00) from ORIGIN to dt."""
    if dt <= ORIGIN:
        return 0
    total = 0
    cur = ORIGIN
    while cur.date() < dt.date():
        if cur.weekday() < 5:                      # Mon-Fri
            total += MIN_PER_DAY
        cur += timedelta(days=1)
        cur = cur.replace(hour=9, minute=0)
    # same day as dt
    if dt.weekday() < 5:
        end_min = min(dt.hour, WORK_END_H) * 60 + (dt.minute if dt.hour < WORK_END_H else 0)
        start_min = WORK_START_H * 60
        total += max(0, end_min - start_min)
    return total

# ---------- load data ----------
df = pd.read_excel('adrs\job_shop_clean_1.xlsx')

ops = []   # one dict per row/component
for _, r in df.iterrows():
    outsourced = (r['Process Type'] == 'Outsource')
    if outsourced:
        dur = random.randint(LEAD_MIN_DAYS, LEAD_MAX_DAYS) * MIN_PER_DAY
    else:
        dur = math.ceil(r['Quantity Required'] * r['Cycle Time (seconds)'] / 60.0)  # minutes
    ops.append(dict(
        job=int(r['Sr. No']), comp=r['Components'], machine=r['Machine Number'],
        outsourced=outsourced, dur=dur,
        due=to_work_minutes(r['Promised Delivery Date'].to_pydatetime()),
        lead_days=(dur // MIN_PER_DAY) if outsourced else None,
    ))

jobs = sorted({o['job'] for o in ops})
machines = sorted({o['machine'] for o in ops if not o['outsourced']})
HORIZON = sum(o['dur'] for o in ops) + max(o['due'] for o in ops)

# ---------- model ----------
m = cp_model.CpModel()
starts, ends, intervals = {}, {}, {}
mach_intervals = {mc: [] for mc in machines}

for i, o in enumerate(ops):
    s = m.new_int_var(0, HORIZON, f's{i}')
    e = m.new_int_var(0, HORIZON, f'e{i}')
    iv = m.new_interval_var(s, o['dur'], e, f'iv{i}')
    starts[i], ends[i], intervals[i] = s, e, iv
    if not o['outsourced']:
        mach_intervals[o['machine']].append(iv)

for mc in machines:
    m.add_no_overlap(mach_intervals[mc])   # unit capacity per machine

# job completion = max end over its components
job_done, tard = {}, {}
for j in jobs:
    idxs = [i for i, o in enumerate(ops) if o['job'] == j]
    due = ops[idxs[0]]['due']
    cj = m.new_int_var(0, HORIZON, f'C{j}')
    m.add_max_equality(cj, [ends[i] for i in idxs])
    job_done[j] = cj
    t = m.new_int_var(0, HORIZON, f'T{j}')
    m.add_max_equality(t, [cj - due, 0])
    tard[j] = t

makespan = m.new_int_var(0, HORIZON, 'makespan')
m.add_max_equality(makespan, [ends[i] for i in range(len(ops))])

total_tard = sum(tard[j] for j in jobs)
m.minimize(total_tard * 100000 + makespan)   # tardiness primary, makespan secondary

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8
st = solver.solve(m)

# ---------- report ----------
def fmt(mins):  # working-minutes -> "Dd HH:MM" working-day index
    d, r = divmod(mins, MIN_PER_DAY)
    return f"d{d}+{r//60:02d}:{r%60:02d}"

print("STATUS:", solver.status_name(st))
print(f"TOTAL TARDINESS: {solver.value(total_tard)} min "
      f"({solver.value(total_tard)/MIN_PER_DAY:.2f} working-days)")
print(f"MAKESPAN: {solver.value(makespan)} min ({solver.value(makespan)/MIN_PER_DAY:.2f} working-days)")
print()
print(f"{'Job':<4}{'Due(min)':>9}{'Done(min)':>10}{'Slack':>8}{'Tardy':>7}  status")
for j in jobs:
    due = next(o['due'] for o in ops if o['job'] == j)
    c = solver.value(job_done[j]); t = solver.value(tard[j])
    slack = due - c
    print(f"{j:<4}{due:>9}{c:>10}{slack:>8}{t:>7}  {'LATE' if t>0 else 'ok'}")

print("\nRandom outsource lead times (working days):")
for o in ops:
    if o['outsourced']:
        print(f"  Job {o['job']} {o['comp']} (outsourced): {o['lead_days']}d = {o['dur']} min")

print("\nPer-machine schedule (start-end in working-minutes):")
for mc in machines:
    rows = sorted([(solver.value(starts[i]), solver.value(ends[i]), ops[i])
                   for i, o in enumerate(ops) if o['machine'] == mc])
    print(f" {mc}:")
    for s, e, o in rows:
        print(f"    J{o['job']:<2} {o['comp']}  {s:>6}-{e:<6}  ({fmt(s)}->{fmt(e)})  dur={o['dur']}")