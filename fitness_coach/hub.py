from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from fitness_coach.config import get_fast_llm
from fitness_coach.state import AgentState

_CONFIDENCE_THRESHOLD = 0.75

# ── Router ─────────────────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    route: Literal["COACH", "WORKOUT_GENERATE", "WORKOUT_LOG", "CLARIFY", "NEW_TOPIC"] = Field(
        description="Most appropriate agent for this request"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the routing decision (0 = uncertain, 1 = certain)"
    )

_ROUTER_SYSTEM = SystemMessage(
    content=(
        "You are a routing agent for a fitness coaching system. "
        "Classify the user's request into exactly one route:\n\n"
        "  COACH            — fitness questions, exercise education, form advice, anatomy\n"
        "  WORKOUT_GENERATE — requests to create, plan, or build a new workout\n"
        "  WORKOUT_LOG      — logging or recording a completed exercise or workout\n"
        "  CLARIFY          — genuinely ambiguous; intent cannot be determined\n"
        "  NEW_TOPIC        — the user is clearly starting an unrelated new conversation\n"
        "                     (only use when prior history exists AND the new message has\n"
        "                     no connection to it; corrections and follow-ups are NOT new topics)\n\n"
        "Provide an honest confidence score.\n\n"
        "NEW_TOPIC examples (prior history present, genuinely unrelated new request):\n"
        "  'what should I eat for breakfast?' after a completed log exchange\n"
        "  'can you build me a chest workout?' after a fully resolved coach Q&A\n\n"
        "NOT new topics — route these normally:\n"
        "  'oh wait, I did 4 sets not 3' after logging     → WORKOUT_LOG (correction)\n"
        "  'it was awesome' after workout generation        → CLARIFY (follow-up)\n"
        "  'actually make it 45 minutes' after generate     → WORKOUT_GENERATE (revision)\n\n"
        "Use CLARIFY whenever confidence would be below 0.75.\n"
        "Use NEW_TOPIC only when history is non-empty and the shift is unambiguous."
    )
)

def _router(state: AgentState) -> dict:
    decision: RouteDecision = (
        get_fast_llm().with_structured_output(RouteDecision).invoke(
            [_ROUTER_SYSTEM, *state["messages"]]
        )
    )
    if decision.confidence < _CONFIDENCE_THRESHOLD:
        decision.route = "CLARIFY"
    return {"route": decision.route, "confidence": decision.confidence}


# ── Sub-agent nodes ────────────────────────────────────────────────────────────

def _clarify(state: AgentState) -> dict:
    response = get_fast_llm().invoke([
        SystemMessage(
            content=(
                "You are a friendly fitness coach responding to an ambiguous message. "
                "Read the full conversation history before replying.\n\n"
                "If the user just reacted positively to a workout plan that was generated "
                "earlier in the conversation (e.g. 'it was awesome', 'looks great', 'perfect', "
                "'I loved it'), acknowledge the reaction warmly and ask if they would like to "
                "log that workout.\n\n"
                "Otherwise, ask one brief, friendly question to clarify whether they want "
                "fitness advice, a new workout plan, or to log a completed workout."
            )
        ),
        *state["messages"],
    ])
    return {"response": response.content}


def _coach(state: AgentState) -> dict:
    from fitness_coach.agents import coach
    return coach.run(state)


def _generator(state: AgentState, config: RunnableConfig) -> dict:
    from fitness_coach.agents import generator
    return generator.run(state, config)


def _logger(state: AgentState) -> dict:
    from fitness_coach.agents import logger
    return logger.run(state)


# ── Resolution check ───────────────────────────────────────────────────────────
# Runs after every sub-agent. Single consistent place that determines whether
# the conversation is resolved or the agent is waiting for more input from the user.

class _Resolution(BaseModel):
    needs_input: bool = Field(
        description=(
            "True if the response asks the user a question or requires them to provide "
            "more information before anything useful can happen. "
            "False if the response fully addresses the request."
        )
    )

_RESOLUTION_SYSTEM = SystemMessage(
    content=(
        "You are evaluating a fitness coach's response to determine whether it needs "
        "a follow-up from the user. Answer True if the response asks a question or "
        "is waiting on the user for more detail. Answer False if it is a complete, "
        "self-contained response that does not require a reply."
    )
)

def _resolution_check(state: AgentState) -> dict:
    response = state.get("response") or ""
    if not response:
        return {"needs_input": False}
    check: _Resolution = get_fast_llm().with_structured_output(_Resolution).invoke([
        _RESOLUTION_SYSTEM,
        HumanMessage(content=f"Response to evaluate:\n\n{response}"),
    ])
    return {"needs_input": check.needs_input}


# ── Graph ──────────────────────────────────────────────────────────────────────

def _confirm_topic_change(state: AgentState) -> dict:
    response = get_fast_llm().invoke([
        SystemMessage(
            content=(
                "The user appears to be switching to a new topic. "
                "Ask exactly one short, friendly question: should we start a fresh conversation "
                "or continue from where we left off? "
                "Do NOT ask what they want to do next — only ask start-fresh-or-continue."
            )
        ),
        *state["messages"],
    ])
    return {"response": response.content}


def _pick_route(state: AgentState) -> str:
    route = state.get("route", "CLARIFY")
    # NEW_TOPIC requires prior context — fall back to CLARIFY on a cold start
    if route == "NEW_TOPIC" and len(state.get("messages", [])) <= 1:
        return "CLARIFY"
    return route


@lru_cache(maxsize=1)
def build():
    g = StateGraph(AgentState)

    for name, fn in {
        "router": _router, "clarify": _clarify, "coach": _coach,
        "generator": _generator, "logger": _logger,
        "confirm_topic_change": _confirm_topic_change,
        "resolution_check": _resolution_check,
    }.items():
        g.add_node(name, fn)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _pick_route,
        {
            "COACH":            "coach",
            "WORKOUT_GENERATE": "generator",
            "WORKOUT_LOG":      "logger",
            "CLARIFY":          "clarify",
            "NEW_TOPIC":        "confirm_topic_change",
        },
    )
    for node in ("clarify", "coach", "generator", "logger", "confirm_topic_change"):
        g.add_edge(node, "resolution_check")
    g.add_edge("resolution_check", END)

    return g.compile()
