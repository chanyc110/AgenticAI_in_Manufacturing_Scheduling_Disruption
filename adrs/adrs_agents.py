"""
ADRS agent layer — five LangGraph nodes over adrs_core, reasoning with Groq/LLaMA 3.

Orchestrator  -> sets up the committed (nominal) plan, controls the loop
Detection     -> is actual != assumed, and how bad (LLM classifies severity)
Impact        -> computes do-nothing consequences, LLM decides reschedule or not
Rescheduling  -> calls OR-Tools (core.reschedule_from) — the only optimisation step
Explainability-> LLM narrates what changed and why, for a human planner

Division of labour: every number comes from adrs_core; the LLM never does the maths.
"""
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

import adrs_core as core

load_dotenv()  # reads GROQ_API_KEY

# one model powers every agent; temperature 0 for reproducible reasoning
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ---------- structured outputs ----------
class DetectionOut(BaseModel):
    severity: Literal["minor", "moderate", "severe"] = Field(description="overall disruption severity")
    reasoning: str = Field(description="one sentence justifying the severity")

class ImpactOut(BaseModel):
    decision: Literal["reschedule", "no_action"]
    reasoning: str = Field(description="why this decision, referencing the at-risk jobs")

# method="json_schema" constrains decoding to the schema directly instead of
# relying on the model choosing to emit a tool call. gpt-oss-120b (Groq's
# replacement for the deprecated llama-3.3-70b-versatile) intermittently
# answers in prose under the default function-calling method, which raises
# "Tool choice is required, but model did not call a tool".
det_llm = llm.with_structured_output(DetectionOut, method="json_schema")
imp_llm = llm.with_structured_output(ImpactOut, method="json_schema")

class OptionLine(BaseModel):
    objective: str = Field(description="the option's objective key")
    explanation: str = Field(description="one plain sentence: what this option prioritises and its trade-off vs the others")


## have to decide whether you want the AI agent to reccomend or use a deterministic formula to reccomend.
## currently its deterministic and agent explains
class OptionsAdvice(BaseModel):
    lines: list[OptionLine]
    reason: str = Field(description="one sentence justifying the already-chosen recommended option, "
                                    "referencing its tardiness and how many operations it moves")

adv_llm = llm.with_structured_output(OptionsAdvice, method="json_schema")


# ---------- shared state ----------
class State(TypedDict, total=False):
    ops: list                      # operations (from core.load_ops)
    committed_plan: dict           # current committed schedule (idx -> {start,end,machine})
    now: int                       # rescheduling point (working minutes)
    actual_returns: dict           # {op_idx: actual_lead_minutes} — the disruption
    disruption: dict               # detection output
    donothing: dict                # impact: do-nothing evaluation
    collateral: dict               # impact: late jobs that are NOT the disrupted one
    dis_jobs: list                 # jobs whose own part deviated (late OR early), union of below
    late_dis_jobs: list             # jobs whose own part is physically late (unrecoverable)
    early_dis_jobs: list            # jobs whose own part returned early (not a problem)
    decision: str                  # "reschedule" | "no_action"
    impact_reasoning: str
    reschedule: dict               # rescheduling output
    explanation: str               # explainability output
    chosen: dict                   # the option the user picked (set on resume)
    options: list                  # the 4 option plans (from objectives layer)
    advice: dict                   # options advisor output
    achievable_gain: float         # best reschedule improvement vs do-nothing (minutes)
    best_objective: str            # which objective achieved it


# ---------- 1. Orchestrator ----------
def orchestrator(state: State) -> dict:
    if not state.get("committed_plan"):
        nominal = core.build_schedule(state["ops"], now=0)
        return {"committed_plan": nominal["plan"]}
    return {}


# ---------- 2. Disruption Detection ----------
def detection(state: State) -> dict:
    ops, ar = state["ops"], state["actual_returns"]
    facts = []
    any_late, any_early = False, False
    for idx, actual in ar.items():
        o = ops[idx]
        diff = actual - o["dur"]
        if diff == 0:
            continue
        if diff > 0:
            any_late = True
            facts.append(f"Job {o['job']} outsourced {o['comp']}: assumed "
                         f"{core.fmt_wd(o['dur'])}, actual {core.fmt_wd(actual)}, "
                         f"LATE by {core.fmt_wd(diff)}")
        else:
            any_early = True
            facts.append(f"Job {o['job']} outsourced {o['comp']}: assumed "
                         f"{core.fmt_wd(o['dur'])}, actual {core.fmt_wd(actual)}, "
                         f"EARLY by {core.fmt_wd(-diff)}")
    direction = ("later than planned" if any_late and not any_early else
                 "earlier than planned" if any_early and not any_late else
                 "earlier or later than planned")
    prompt = (
        "You are the Disruption Detection agent in a manufacturing scheduler. "
        f"One or more outsourced components returned {direction} compared with the "
        "assumed lead time used to build the schedule. A component returning EARLY is "
        "not a problem in itself -- it cannot make anything late. Only genuinely LATE "
        "returns are a potential source of lateness, though an early return may still be "
        "worth acting on if it frees capacity that other jobs need; that is assessed "
        "separately downstream. Classify the "
        "overall severity (minor/moderate/severe) and justify in one sentence.\n\n"
        "Deviations:\n" + "\n".join(facts)
    )
    out = det_llm.invoke(prompt)
    return {"disruption": {"facts": facts, "severity": out.severity, "reasoning": out.reasoning}}


# Below this, an improvement isn't worth the disruption of republishing the plan.
# Reported in the methodology as the reschedule trigger threshold.
RESCHEDULE_GAIN_THRESHOLD_WD = 0.5


# ---------- 3. Impact Assessment ----------
def impact(state: State) -> dict:
    ops, committed, ar = state["ops"], state["committed_plan"], state["actual_returns"]
    dn = core.evaluate_donothing(ops, committed, ar, now=state["now"])
    # Jobs whose actual return differs from the assumed lead -- split by direction.
    dis_idx = {i for i in ar if ar[i] != ops[i]["dur"]}
    late_dis_jobs = {ops[i]["job"] for i in dis_idx if ar[i] > ops[i]["dur"]}
    early_dis_jobs = {ops[i]["job"] for i in dis_idx if ar[i] < ops[i]["dur"]}
    dis_jobs = late_dis_jobs | early_dis_jobs
    late = {j: t for j, t in dn["tardiness"].items() if t > 0}
    collateral = {j: t for j, t in late.items() if j not in dis_jobs}
    coll_txt = ", ".join(f"Job {j} late by {core.fmt_wd_words(t)}" for j, t in collateral.items()) or "none"

    own_part_lines = []
    if late_dis_jobs:
        own_part_lines.append(
            f"Job(s) {sorted(late_dis_jobs)} had their OWN outsourced part return LATE and "
            "cannot be recovered by rescheduling -- no schedule change can undo a delay "
            "that has already happened.")
    if early_dis_jobs:
        own_part_lines.append(
            f"Job(s) {sorted(early_dis_jobs)} had their OWN outsourced part return EARLY. "
            "An early return cannot make anything late, but it may free capacity sooner; "
            "whether that is worth acting on is captured by the achievable improvement below.")
    own_part_txt = " ".join(own_part_lines) if own_part_lines else \
        "No job's own outsourced part changed unfavourably."

    # MEASURE, don't assume. Solve all four objectives now and take the best
    # achievable outcome, so an EARLY return is assessed by exactly the same
    # criterion as a late one rather than being excluded by a hardcoded rule.
    # These plans are reused by the objectives node, so this costs no extra solving.
    opts = core.generate_options(ops, committed, state["now"], ar)
    best = min(opts, key=lambda o: o["total_tardiness"])
    gain = dn["total_tardiness"] - best["total_tardiness"]

    # Safety guard: CP-SAT returns FEASIBLE rather than OPTIMAL on hard instances
    # (notably under compressed due dates), and a FEASIBLE plan can be WORSE than
    # simply right-shifting. Never commit one.
    if gain <= 0:
        return {"donothing": dn, "collateral": collateral, "dis_jobs": sorted(dis_jobs),
                "late_dis_jobs": sorted(late_dis_jobs), "early_dis_jobs": sorted(early_dis_jobs),
                "options": opts, "achievable_gain": gain, "best_objective": None,
                "decision": "no_action",
                "impact_reasoning": (
                    f"No rescheduling option improves on the current plan "
                    f"(best available is {core.fmt_wd_words(-gain)} worse), so the "
                    f"committed plan is retained.")}
    
    prompt = (
        "You are the Impact Assessment agent for a production scheduler. Decide whether "
        "to trigger a full reschedule, based on these facts:\n\n"
        f"If we DO NOTHING (keep the current plan), total tardiness across all jobs is "
        f"{core.fmt_wd_words(dn['total_tardiness'])}.\n"
        f"If we DO RESCHEDULE, the best of the four objective plans achieves "
        f"{core.fmt_wd_words(best['total_tardiness'])} (objective '{best['objective']}').\n"
        f"ACHIEVABLE IMPROVEMENT: {core.fmt_wd_words(gain)}.\n"
        f"{own_part_txt}\n"
        f"Collateral damage (OTHER jobs dragged late only because the plan wasn't "
        f"re-sequenced): {coll_txt}.\n\n"
        "Base your decision on the ACHIEVABLE IMPROVEMENT above. It already accounts for "
        "what can and cannot be recovered in either direction: rescheduling cannot undo a "
        "late return that already happened, and an early return only helps if the optimiser "
        "can actually exploit the freed capacity. Rescheduling has a real cost -- it "
        "disrupts the current plan and consumes planner attention -- so trigger it only "
        f"when the improvement justifies that. Guide: below "
        f"{RESCHEDULE_GAIN_THRESHOLD_WD} working-days is usually not worth acting on; "
        "several working-days clearly is. Decide 'reschedule' or 'no_action', and justify "
        "in one sentence citing the achievable improvement and any specific collateral jobs."
    )
    
    out = imp_llm.invoke(prompt)
    return {"donothing": dn, "collateral": collateral, "dis_jobs": sorted(dis_jobs),
            "late_dis_jobs": sorted(late_dis_jobs), "early_dis_jobs": sorted(early_dis_jobs),
            "options": opts, "achievable_gain": gain, "best_objective": best["objective"],
            "decision": out.decision, "impact_reasoning": out.reasoning}


# ---------- 4. Objectives layer ----------
def objectives(state: State) -> dict:
    # impact() already solved all four while measuring the achievable gain
    if state.get("options"):
        return {}
    return {"options": core.generate_options(state["ops"], state["committed_plan"],
                                             state["now"], state["actual_returns"])}



def _recommendation_score(option):
    """Lower is better: fewest late deliveries first, then fewest changed start
    times (schedule stability) as tiebreak. Deterministic — the LLM never
    chooses this, only explains it."""
    return (option["total_tardiness"], option["start_times_changed"])


# ---------- 5. Options Advisor (compares options, recommends one) ----------
def options_advisor(state: State) -> dict:
    opts = state["options"]
    recommended = min(opts, key=_recommendation_score)["objective"]

    summary = "\n".join(
        f"{o['objective']} ({o['label']}): tardiness {core.fmt_wd(o['total_tardiness'])}, "
        f"makespan {core.fmt_wd(o['makespan'])}, {o['start_times_changed']} start times changed"
        for o in opts)
    facts = "; ".join(state["disruption"]["facts"])
    prompt = (
        "You are the Options Advisor agent for a production planner. Four rescheduling "
        "options were produced, each optimising a different objective. For EACH option, "
        "write one plain sentence on what it prioritises and its trade-off compared with "
        "the others.\n\n"
        f"The recommended option has ALREADY been chosen (fewest late deliveries, then "
        f"fewest changed start times as tiebreak): it is '{recommended}'. Write one sentence "
        "justifying why this is the sensible default.\n\n"
        f"Disruption: {facts}\n\nOptions:\n{summary}\n\n"
        "Respond ONLY with a JSON object, no prose, of the form: "
        '{"lines": [{"objective": "<key>", "explanation": "<one sentence>"}, ...], '
        '"reason": "<one sentence>"}'
    )
    out = adv_llm.invoke(prompt)
    return {"advice": dict(recommended=recommended, reason=out.reason,
                           explanations={l.objective: l.explanation for l in out.lines})}
    
    
# ---------- 6. Human pick (suspends the graph until the user chooses) ----------
def human_pick(state: State) -> dict:
    # interrupt() halts the run here; the value is surfaced to the app.
    # On resume, Command(resume=...) supplies the chosen objective key.
    chosen_objective = interrupt({
        "options": [{"objective": o["objective"], "label": o["label"],
                     "total_tardiness": o["total_tardiness"], "makespan": o["makespan"],
                     "start_times_changed": o["start_times_changed"]} for o in state["options"]],
        "advice": state["advice"],
    })
    chosen = next(o for o in state["options"] if o["objective"] == chosen_objective)
    return {"chosen": chosen,
            "reschedule": {"plan": chosen["plan"],
                           "tardiness": core.plan_tardiness(state["ops"], chosen["plan"]),
                           "total_tardiness": chosen["total_tardiness"]}}


# ---------- 7. Rescheduling (the only OR-Tools call) ----------
def rescheduling(state: State) -> dict:
    rs = core.reschedule_from(state["ops"], state["committed_plan"],
                              state["now"], state["actual_returns"])
    return {"reschedule": rs}


# ---------- 8. Explainability ----------
def explainability(state: State) -> dict:
    ops, committed = state["ops"], state["committed_plan"]
    dn = state["donothing"]
    before = core.asm_order(ops, committed)
    late_dis_jobs = state.get("late_dis_jobs", state.get("dis_jobs", []))
    early_dis_jobs = state.get("early_dis_jobs", [])

    if state["decision"] == "reschedule":
        rs = state["reschedule"]; after = core.asm_order(ops, rs["plan"])
        saved = {j: dn["tardiness"][j] - rs["tardiness"][j]
                for j in dn["tardiness"] if dn["tardiness"][j] - rs["tardiness"][j] > 0}
        saved_txt = ", ".join(f"Job {j} (saved {core.fmt_wd(v)})" for j, v in saved.items()) or "none"
        instruction = ("explain what happened, what was decided, and why — if there are "
                       "unrecoverable late job(s) listed below, note they cannot be saved "
                       "but other jobs were protected by re-sequencing the shared assembly "
                       "station; do not invent a late job if none is listed.")
        facts = (f"Disruption: {state['disruption']['facts']}\n"
                 f"Unrecoverable (late-returning) job(s): {late_dis_jobs or 'none'}\n"
                 f"Early-returning job(s) (not a problem): {early_dis_jobs or 'none'}\n"
                 f"Do-nothing tardiness: {core.fmt_wd_words(dn['total_tardiness'])}\n"
                 f"Reschedule tardiness: {core.fmt_wd_words(rs['total_tardiness'])}\n"
                 f"ASM order before: {before}\nASM order after: {after}\n"
                 f"Jobs protected: {saved_txt}")
    else:
        # The instruction must match what actually happened -- a return that was only
        # ever EARLY must never be narrated as an unrecoverable late job (this was the
        # actual bug: this branch used to hardcode "the late job cannot be saved"
        # regardless of whether anything was late at all).
        if late_dis_jobs and not early_dis_jobs:
            instruction = ("explain what happened and why NO rescheduling was done — the "
                           "late job cannot be saved, but no other jobs were at risk, so "
                           "the plan was left unchanged. Do NOT claim any re-sequencing "
                           "occurred.")
        elif early_dis_jobs and not late_dis_jobs:
            instruction = ("explain what happened: a component returned EARLIER than "
                           "assumed. Make clear this is NOT a problem -- it cannot make "
                           "anything late and required no schedule change. Do NOT describe "
                           "this as a late or unrecoverable job, and do NOT claim any "
                           "re-sequencing occurred.")
        else:
            instruction = ("explain what happened and why NO rescheduling was done — "
                           "describe each deviation accurately as either early (harmless) "
                           "or late (unrecoverable for that job) using the lists below, and "
                           "note no other jobs were at risk, so the plan was left unchanged. "
                           "Do NOT claim any re-sequencing occurred.")
        facts = (f"Disruption: {state['disruption']['facts']}\n"
                 f"Late-returning job(s): {late_dis_jobs or 'none'}\n"
                 f"Early-returning job(s): {early_dis_jobs or 'none'}\n"
                 f"Do-nothing tardiness: {core.fmt_wd(dn['total_tardiness'])}\n"
                 "No collateral damage; rescheduling could not improve outcomes.")

    prompt = ("You are the Explainability agent. In 3-4 plain sentences for a production "
              "planner (no jargon), " + instruction + "\n\n" + facts)
    return {"explanation": llm.invoke(prompt).content}


# ---------- routing (orchestrator's control decision) ----------
def route(state: State) -> str:
    return state["decision"]


def build_app():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("detection", detection)
    g.add_node("impact", impact)
    g.add_node("objectives", objectives)
    g.add_node("options_advisor", options_advisor) 
    g.add_node("human_pick", human_pick)
    g.add_node("explainability", explainability)   # used on the no_action branch

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "detection")
    g.add_edge("detection", "impact")
    g.add_conditional_edges("impact", route,
                            {"reschedule": "objectives", "no_action": "explainability"})
    g.add_edge("objectives", "options_advisor")
    g.add_edge("options_advisor", "human_pick")
    g.add_edge("human_pick", "explainability")
    g.add_edge("explainability", END)
    return g.compile(checkpointer=MemorySaver())   # checkpointer enables suspend/resume


# ---------- run ----------
if __name__ == "__main__":
    ops = core.load_ops()
    nominal = core.build_schedule(ops, now=0)

    # inject the Job 8 disruption (assumed ~6wd, actually 22wd)
    dis = next(o["idx"] for o in ops if o["job"] == 8 and o["outsourced"])
    init: State = {
        "ops": ops,
        "committed_plan": nominal["plan"],
        "now": ops[dis]["dur"],
        "actual_returns": {dis: 22 * core.MIN_PER_DAY},
    }

    app = build_app()
    final = app.invoke(init)

    print("SEVERITY :", final["disruption"]["severity"], "-", final["disruption"]["reasoning"])
    print("DECISION :", final["decision"], "-", final["impact_reasoning"])
    print("DO-NOTHING:", core.fmt_wd(final["donothing"]["total_tardiness"]))
    if "reschedule" in final:
        print("RESCHEDULE:", core.fmt_wd(final["reschedule"]["total_tardiness"]))
    print("\nEXPLANATION:\n" + final["explanation"])