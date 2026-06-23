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

import adrs_core as core

load_dotenv()  # reads GROQ_API_KEY from .env

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
    dn = core.evaluate_donothing(ops, committed, ar)             # numbers from core
    dis_jobs = {ops[i]["job"] for i in ar}
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
    return {"donothing": dn, "collateral": collateral, "dis_jobs": sorted(dis_jobs),
            "decision": out.decision, "impact_reasoning": out.reasoning}


# ---------- 4. Rescheduling (the only OR-Tools call) ----------
def rescheduling(state: State) -> dict:
    rs = core.reschedule_from(state["ops"], state["committed_plan"],
                              state["now"], state["actual_returns"])
    return {"reschedule": rs}


# ---------- 5. Explainability ----------
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
    g.add_node("rescheduling", rescheduling)
    g.add_node("explainability", explainability)

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "detection")
    g.add_edge("detection", "impact")
    g.add_conditional_edges("impact", route,
                            {"reschedule": "rescheduling", "no_action": "explainability"})
    g.add_edge("rescheduling", "explainability")
    g.add_edge("explainability", END)
    return g.compile()


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