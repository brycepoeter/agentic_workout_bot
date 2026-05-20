from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from fitness_coach import data
from fitness_coach.config import get_fast_llm
from fitness_coach.state import AgentState

# ── Schemas ────────────────────────────────────────────────────────────────────

class LogAssessment(BaseModel):
    has_exercise_data: bool = Field(
        description=(
            "True only if the conversation explicitly names at least one specific exercise "
            "the user actually performed — not just a muscle group or vague intent to log."
        )
    )
    context_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Muscle groups, body parts, or exercise types mentioned "
            "(e.g. 'biceps', 'chest', 'pull ups'). Used to suggest relevant exercises."
        ),
    )


class LogEntry(BaseModel):
    exercise_name_raw: str = Field(description="Exercise name as the user stated it")
    matched_id: str | None = Field(None, description="Matched exercise ID from the database")
    matched_name: str | None = Field(None, description="Matched exercise name from the database")
    sets: int | None = Field(None, description="Number of sets performed")
    reps: int | None = Field(None, description="Reps per set")
    weight_lbs: float | None = Field(None, description="Weight in pounds")
    duration_seconds: int | None = Field(None, description="Duration in seconds for timed exercises")
    notes: str | None = Field(None, description="Any additional context")


class WorkoutLog(BaseModel):
    entries: list[LogEntry] = Field(description="One entry per exercise mentioned")


# ── Node ───────────────────────────────────────────────────────────────────────

_ASSESS_SYSTEM = SystemMessage(
    content=(
        "Analyze the conversation. Decide whether the user has named a specific exercise "
        "they actually performed (not just expressed intent to log or mentioned a muscle group). "
        "Also extract any muscle groups, body parts, or exercise types mentioned."
    )
)

_EXTRACT_SYSTEM = SystemMessage(
    content=(
        "Extract all workout log entries from the conversation.\n\n"
        "For each exercise, capture: name as stated, sets, reps, weight, and duration.\n\n"
        "IMPORTANT — infer from context before using null:\n"
        "- If the user says 'all of them', 'as you said', 'the workout you made', or similar, "
        "they are referring to a workout plan that appears earlier in the conversation. "
        "Find that plan and use the sets/reps/rest values from it.\n"
        "- If a prior assistant message lists a workout with sets and reps, treat those as the "
        "values for any exercise the user confirms they performed.\n"
        "- Only use null when the information is genuinely absent from the entire conversation."
    )
)


def _format_response(entries: list[LogEntry]) -> str:
    import json
    if not entries:
        return json.dumps({"logged": [], "message": "No exercises identified in the conversation."})
    return json.dumps({
        "logged": [entry.model_dump(exclude_none=True) for entry in entries]
    }, indent=2)


def _run(state: AgentState) -> dict:
    llm = get_fast_llm()

    # Step 1 — enough data to log?
    assessment: LogAssessment = llm.with_structured_output(LogAssessment).invoke(
        [_ASSESS_SYSTEM, *state["messages"]]
    )

    if not assessment.has_exercise_data:
        candidates = data.search(muscle_groups=assessment.context_terms, limit=6)
        if not candidates:
            seen: set[str] = set()
            for term in assessment.context_terms:
                match = data.fuzzy_match(term)
                if match and match.id not in seen:
                    candidates.append(match)
                    seen.add(match.id)

        if candidates:
            options = "\n".join(f"  • {e.name}" for e in candidates)
            return {
                "response": (
                    "I'd love to log that! Which exercise did you do?\n\n"
                    f"Based on our conversation, here are some options:\n{options}\n\n"
                    "Please also share your sets, reps, and weight (if applicable)."
                )
            }

        return {
            "response": (
                "I'd love to log that! What exercise did you do? "
                "Please include sets, reps, and weight if applicable."
            )
        }

    # Step 2 — extract structured entries
    result: WorkoutLog = llm.with_structured_output(WorkoutLog).invoke(
        [_EXTRACT_SYSTEM, *state["messages"]]
    )
    for entry in result.entries:
        match = data.fuzzy_match(entry.exercise_name_raw)
        if match:
            entry.matched_id = match.id
            entry.matched_name = match.name

    return {"response": _format_response(result.entries)}


def run(state: AgentState) -> dict:
    return _run(state)


@lru_cache(maxsize=1)
def build():
    g = StateGraph(AgentState)
    g.add_node("logger", _run)
    g.set_entry_point("logger")
    g.add_edge("logger", END)
    return g.compile()
