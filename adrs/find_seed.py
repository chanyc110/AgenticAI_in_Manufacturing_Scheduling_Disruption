"""
Find a seed that produces a visible without/with difference for a scenario.

Why this has to run on YOUR machine
-----------------------------------
The tight dataset does not solve to proven optimality, so the nominal plan CP-SAT
returns depends on the solver time limit -- and every disruption's return day is
measured from that plan's dispatch times. A seed found under one time limit
therefore samples different disruptions under another, and the run diverges
completely. This script builds the nominal plan exactly as adrs_experiments.py
does, with the same --solver-time-limit, so what it reports is what you will get.

It replicates the rolling-horizon loop but replaces the LLM with the same
measured-gain rule the safety guard applies (reschedule when the best of the
four objective plans beats doing nothing by more than the threshold). The agent
can still decide otherwise, so treat the output as a shortlist, not a promise.

    python find_seed.py --scenario early_2 --disruptions 4 --seeds 1 12
    python find_seed.py --scenario late_1 --disruptions 5 --seeds 1 8 --quick
"""
import argparse
import random

import adrs_core as core
from adrs_experiments import (EVENT_SHAPE, DELAY_MIN_DAYS, DELAY_MAX_DAYS,
                              EARLY_MIN_SLACK_DAYS, SCENARIO_KINDS,
                              leads_from_return_days, event_clock)

MPD = core.MIN_PER_DAY


def sample_run(ops, base, rng, kind, n_events):
    """Mirror of plan_events + sample_event, kept in step by using the same
    constants. Draws the whole run's events, ordered by knowable time."""
    used, evs = set(), []
    for _ in range(n_events):
        rds, parts = {}, []
        for direction in EVENT_SHAPE[kind]:
            pool = [o for o in ops if o["outsourced"] and o["idx"] not in used
                    and (o["dur"] >= EARLY_MIN_SLACK_DAYS * MPD
                         if direction == "early" else True)]
            if not pool:
                break
            o = rng.choice(pool)
            used.add(o["idx"])
            if direction == "late":
                d = rng.randint(DELAY_MIN_DAYS, DELAY_MAX_DAYS)
                rds[o["idx"]] = (base[o["idx"]]["start"] + o["dur"]) / MPD + d
                parts.append(f"P{o['job']}{o['comp']}+{d}wd")
            else:
                assumed = int(o["dur"] / MPD)
                act = rng.randint(0, max(1, assumed - 1))
                rds[o["idx"]] = base[o["idx"]]["start"] / MPD + act
                parts.append(f"P{o['job']}{o['comp']} {assumed}->{act}wd")
        if not rds:
            break
        evs.append((rds, parts, event_clock(ops, base, rds)))
    evs.sort(key=lambda e: e[2])
    return evs


def trial(ops, base, seed, kind, n_events, threshold_wd, use_all_objectives, objective="tardiness"):
    rng = random.Random(seed)
    committed, known, floor = base, {}, 0
    n_res, gains = 0, []
    for rds, parts, _ in sample_run(ops, base, rng, kind, n_events):
        known.update(rds)
        ar = leads_from_return_days(committed, known)
        now = max(floor, event_clock(ops, committed, rds))
        dn = core.evaluate_donothing(ops, committed, ar, now=now)
        if use_all_objectives:
            # Commit the SAME objective the real run is held to, not the best of
            # the four. Taking best-of-four here reports a plan that will never
            # be applied and makes the predicted tardiness diverge from the run.
            opts = core.generate_options(ops, committed, now, ar)
            chosen = next(o for o in opts if o["objective"] == objective)
            plan, tard = chosen["plan"], chosen["total_tardiness"]
        else:
            rs = core.reschedule_from(ops, committed, now, ar)
            plan, tard = rs["plan"], rs["total_tardiness"]
        gain = (dn["total_tardiness"] - tard) / MPD
        gains.append(round(gain, 1))
        if gain > threshold_wd:
            committed = plan
            n_res += 1
        floor = now

    moved = {(o["job"], o["machine"]) for o in ops
             if o["kind"] == "comp" and not o["outsourced"]
             and committed[o["idx"]]["start"] != base[o["idx"]]["start"]}
    ar_b = leads_from_return_days(base, known)
    dn_f = core.evaluate_donothing(ops, base, ar_b, now=floor)["total_tardiness"] / MPD
    rs_f = sum(core.plan_tardiness(ops, committed).values()) / MPD
    return dict(machines=sorted({m for _, m in moved}), n_moved=len(moved),
                n_res=n_res, gains=gains, dn=dn_f, rs=rs_f)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="job_shop_20products_tight.xlsx")
    p.add_argument("--scenario", default="early_2", choices=SCENARIO_KINDS)
    p.add_argument("--disruptions", type=int, default=4)
    p.add_argument("--seeds", nargs=2, type=int, default=[1, 11], metavar=("FROM", "TO"))
    p.add_argument("--solver-time-limit", type=int, default=20,
                   help="MUST match the --solver-time-limit of the real run")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="working-days of gain below which the agent is assumed to decline")
    p.add_argument("--objective", default="tardiness",
                   choices=("tardiness", "makespan", "stability", "balanced"),
                   help="must match the --objective of the real run")
    p.add_argument("--quick", action="store_true",
                   help="one solve per event instead of all four objectives -- ~4x faster, "
                        "slightly less faithful to the pipeline")
    a = p.parse_args()

    core.DEFAULT_TIME_LIMIT = a.solver_time_limit
    ops = core.load_ops(path=a.dataset)
    nominal = core.build_schedule(ops, now=0)
    base = nominal["plan"]
    print(f"dataset {a.dataset} | solver {a.solver_time_limit}s")
    print(f"nominal: tardiness {nominal['total_tardiness']/MPD:.2f}wd  {nominal['status']}")
    if nominal["status"] != "OPTIMAL":
        print("  NOTE: nominal is not proven optimal, so this plan -- and therefore every\n"
              "  seed below -- is specific to this time limit. Use the SAME value when you\n"
              "  run the experiment, or the results will not reproduce.")
    print(f"\nsearching {a.scenario}, {a.disruptions} events, "
          f"seeds {a.seeds[0]}..{a.seeds[1]-1}\n")

    hits = []
    for s in range(a.seeds[0], a.seeds[1]):
        r = trial(ops, base, s, a.scenario, a.disruptions, a.threshold, not a.quick, a.objective)
        flag = "  <-- usable" if r["n_moved"] >= 3 else ""
        print(f"seed {s:>3}: {r['n_moved']:>2} machine starts moved {str(r['machines']):22s} "
              f"{r['n_res']}/{a.disruptions} resched  tard {r['dn']:.1f}->{r['rs']:.1f}wd  "
              f"gains={r['gains']}{flag}", flush=True)
        if r["n_moved"] >= 3:
            hits.append((r["n_moved"], s, r))

    if hits:
        hits.sort(reverse=True)
        n, s, r = hits[0]
        print(f"\nBest: --seed {s}  ({n} machine start times moved across "
              f"{', '.join(r['machines'])}, {r['n_res']}/{a.disruptions} events rescheduled)")
    else:
        print("\nNo seed in this range moved 3+ machine start times. Either widen --seeds, "
              "or accept that this scenario does not perturb the machine layer -- which for "
              "the early scenarios is a legitimate finding rather than a failed search.")


if __name__ == "__main__":
    main()