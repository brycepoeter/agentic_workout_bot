from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from fitness_coach.config import get_llm
from fitness_coach.state import AgentState

_SYSTEM = SystemMessage(
    content=(
        "You are an expert fitness coach. Answer questions about exercise technique, "
        "anatomy, training programming, and general fitness clearly and accurately. "
        "Be concise but thorough."
    )
)


def _run(state: AgentState) -> dict:
    response = get_llm().invoke([_SYSTEM, *state["messages"]])
    return {"response": response.content}


def run(state: AgentState) -> dict:
    return _run(state)


@lru_cache(maxsize=1)
def build():
    g = StateGraph(AgentState)
    g.add_node("coach", _run)
    g.set_entry_point("coach")
    g.add_edge("coach", END)
    return g.compile()
