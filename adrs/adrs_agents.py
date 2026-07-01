"""
ADRS agent layer — five LangGraph nodes over adrs_core, reasoning with Groq/LLaMA 3.

Orchestrator  -> sets up the committed (nominal) plan, controls the loop
Detection     -> is actual != assumed, and how bad (LLM classifies severity)
Impact        -> computes do-nothing consequences, LLM decides reschedule or not
Rescheduling  -> calls OR-Tools (core.reschedule_from) — the only optimisation step
Explainability-> LLM narrates what changed and why, for a human planner

Division of labour: every number comes from adrs_core; the LLM never does the maths.
"""
from typing import TypedDict, Literal, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

import adrs_core as core

load_dotenv()  # reads GROQ_API_KEY

# one model powers every agent; temperature 0 for reproducible reasoning
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ---------- structured outputs ----------
class DetectionOut(BaseModel):
    severity: Literal["minor", "moderate", "severe"] = Field(description="overall disruption severity")
    reasoning: str = Field(description="one sentence justifying the severity")

class ImpactOut(BaseModel):
    decision: Literal["reschedule", "no_action"]
    reasoning: str = Field(description="why this decision, referencing the at-risk jobs")

det_llm = llm.with_structured_output(DetectionOut)
imp_llm = llm.with_structured_output(ImpactOut)

class OptionLine(BaseModel):
    objective: str = Field(description="the option's objective key")
    explanation: str = Field(description="one plain sentence: what it prioritises and its trade-off vs the others")

class OptionsAdvice(BaseModel):
    lines: list[OptionLine]
    recommended: Literal["tardiness", "makespan", "stability", "balanced"]
    reason: str = Field(description="one sentence: why the recommended option is the sensible default")

adv_llm = llm.with_structured_output(OptionsAdvice)


# ---------- shared state ----------
class State(TypedDict, total=False):
    ops: list                      # operations (from core.load_ops)
    committed_plan: dict           # current committed schedule (idx -> {start,end,machine})
    now: int                       # rescheduling point (working minutes)
    actual_returns: dict           # {op_idx: actual_lead_minutes} — the disruption
    disruption: dict               # detection output
    donothing: dict                # impact: do-nothing evaluation
    collateral: dict               # impact: late jobs that are NOT the disrupted one
    dis_jobs: list                 # jobs whose own part is physically late (unrecoverable)
    decision: str                  # "reschedule" | "no_action"
    impact_reasoning: str
    reschedule: dict               # rescheduling output
    explanation: str               # explainability output
    chosen: dict                   # the option the user picked (set on resume)
    options: list                  # the 4 option plans (from objectives layer)
    advice: dict                   # options advisor output


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
    for idx, actual in ar.items():
        o = ops[idx]
        if actual == o["dur"]:
            continue                      # unchanged field — not a disruption
        facts.append(f"Job {o['job']} outsourced {o['comp']}: assumed "
                     f"{core.fmt_wd(o['dur'])}, actual {core.fmt_wd(actual)}, "
                     f"overrun {core.fmt_wd(actual - o['dur'])}")
    prompt = (
        "You are the Disruption Detection agent in a manufacturing scheduler. "
        "An outsourced component returned later than planned. Classify the overall "
        "severity (minor/moderate/severe) and justify in one sentence.\n\n"
        "Deviations:\n" + "\n".join(facts)
    )
    out = det_llm.invoke(prompt)
    return {"disruption": {"facts": facts, "severity": out.severity, "reasoning": out.reasoning}}


# ---------- 3. Impact Assessment ----------
def impact(state: State) -> dict:
    ops, committed, ar = state["ops"], state["committed_plan"], state["actual_returns"]
    dn = core.evaluate_donothing(ops, committed, ar, now=state["now"])             # numbers from core
    # only jobs whose actual return actually differs from the assumed lead are "disrupted"
    dis_idx = {i for i in ar if ar[i] != ops[i]["dur"]}
    dis_jobs = {ops[i]["job"] for i in dis_idx}
    late = {j: t for j, t in dn["tardiness"].items() if t > 0}
    collateral = {j: t for j, t in late.items() if j not in dis_jobs}

    coll_txt = ", ".join(f"Job {j} late by {core.fmt_wd(t)}" for j, t in collateral.items()) or "none"
    prompt = (
        "You are the Impact Assessment agent. If we DO NOTHING (keep the current "
        f"plan), total tardiness is {core.fmt_wd(dn['total_tardiness'])}.\n"
        f"The disrupted job(s) {sorted(dis_jobs)} cannot be recovered — their parts "
        "are physically late, so no schedule change can save them.\n"
        f"Collateral damage (OTHER jobs dragged late only because of the rigid plan): {coll_txt}.\n\n"
        "Rescheduling re-optimises the shared assembly queue and can only help the "
        "collateral jobs. Choose 'reschedule' if there is collateral damage worth "
        "recovering, else 'no_action'. Justify briefly."
    )
    out = imp_llm.invoke(prompt)
    decision = "reschedule" if collateral else "no_action"   # deterministic gate
    return {"donothing": dn, "collateral": collateral, "dis_jobs": sorted(dis_jobs),
            "decision": decision, "impact_reasoning": out.reasoning}
    
    
# ---------- 4. Objectives layer (runs optimiser once per objective) ----------
def objectives(state: State) -> dict:
    opts = core.generate_options(state["ops"], state["committed_plan"],
                                 state["now"], state["actual_returns"])
    return {"options": opts}


# ---------- 5. Options Advisor (compares options, recommends one) ----------
def options_advisor(state: State) -> dict:
    opts = state["options"]
    summary = "\n".join(
        f"{o['objective']} ({o['label']}): tardiness {core.fmt_wd(o['total_tardiness'])}, "
        f"makespan {core.fmt_wd(o['makespan'])}, {o['ops_changed']} operations moved"
        for o in opts)
    facts = "; ".join(state["disruption"]["facts"])
    prompt = (
        "You are the Options Advisor agent for a production planner. Four rescheduling "
        "options were produced, each optimising a different objective. For EACH option, "
        "write one plain sentence on what it prioritises and its trade-off compared with "
        "the others. Then recommend ONE option as the sensible default and justify in one "
        "sentence. Prefer the option with lowest tardiness; break ties by fewest operations "
        f"moved (less disruption).\n\nDisruption: {facts}\n\nOptions:\n{summary}")
    out = adv_llm.invoke(prompt)
    return {"advice": dict(recommended=out.recommended, reason=out.reason,
                           explanations={l.objective: l.explanation for l in out.lines})}
    
    
# ---------- 6. Human pick (suspends the graph until the user chooses) ----------
def human_pick(state: State) -> dict:
    # interrupt() halts the run here; the value is surfaced to the app.
    # On resume, Command(resume=...) supplies the chosen objective key.
    chosen_objective = interrupt({
        "options": [{"objective": o["objective"], "label": o["label"],
                     "total_tardiness": o["total_tardiness"], "makespan": o["makespan"],
                     "ops_changed": o["ops_changed"]} for o in state["options"]],
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

    if state["decision"] == "reschedule":
        rs = state["reschedule"]; after = core.asm_order(ops, rs["plan"])
        saved = {j: dn["tardiness"][j] - rs["tardiness"][j]
                for j in dn["tardiness"] if dn["tardiness"][j] - rs["tardiness"][j] > 0}
        saved_txt = ", ".join(f"Job {j} (saved {core.fmt_wd(v)})" for j, v in saved.items()) or "none"
        instruction = ("explain what happened, what was decided, and why — note the late job "
                       "cannot be saved, but other jobs were protected by re-sequencing the "
                       "shared assembly station.")
        facts = (f"Disruption: {state['disruption']['facts']}\n"
                 f"Unrecoverable job(s): {state['dis_jobs']}\n"
                 f"Do-nothing tardiness: {core.fmt_wd(dn['total_tardiness'])}\n"
                 f"Reschedule tardiness: {core.fmt_wd(rs['total_tardiness'])}\n"
                 f"ASM order before: {before}\nASM order after: {after}\n"
                 f"Jobs protected: {saved_txt}")
    else:
        instruction = ("explain what happened and why NO rescheduling was done — the late job "
                       "cannot be saved, but no other jobs were at risk, so the plan was left "
                       "unchanged. Do NOT claim any re-sequencing occurred.")
        facts = (f"Disruption: {state['disruption']['facts']}\n"
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