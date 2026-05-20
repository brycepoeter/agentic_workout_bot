import json
from functools import lru_cache

from pydantic import ValidationError

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from fitness_coach import data
from fitness_coach.config import get_fast_llm, get_llm
from fitness_coach.state import AgentState

# ── Constants ──────────────────────────────────────────────────────────────────

# Muscle groups that together cover the whole body; used when "full body" is requested.
_FULL_BODY_MUSCLE_GROUPS = [
    "chest", "lats", "deltoids", "triceps", "biceps",  # upper
    "quads", "hamstrings", "glutes", "calves",          # lower
    "core",                                             # core
]

# Phrases a user might use to indicate bodyweight-only intent.
_BODYWEIGHT_TERMS = frozenset({
    "bodyweight", "no equipment", "none", "nothing",
    "no gear", "bodyweight only", "no weights",
})

# ── Tool schemas ───────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    muscle_groups: list[str] | None = Field(
        None,
        description=(
            "Target muscle groups, e.g. ['chest', 'triceps']. "
            "Use 'full body' as a single entry to cover upper body, lower body, and core."
        ),
    )
    equipment: list[str] | None = Field(
        None,
        description=(
            "Available equipment exactly as named in the database, e.g. ['Dumbbell', 'Flat Bench']. "
            "Pass ['bodyweight'] or ['no equipment'] for bodyweight-only exercises. "
            "Omit entirely to search all exercises regardless of equipment."
        ),
    )
    movement_patterns: list[str] | None = Field(
        None, description="Movement patterns, e.g. ['upper push - horizontal', 'lower push - squat']"
    )
    limit: int = Field(10, description="Maximum number of exercises to return")


class WorkoutExercise(BaseModel):
    exercise_id: str = Field(description="Exercise UUID from search_exercises results")
    sets: int = Field(description="Number of sets")
    reps: int | None = Field(None, description="Reps per set; null for duration-based exercises")
    duration_seconds: int | None = Field(None, description="Duration in seconds for timed exercises")
    rest_seconds: int = Field(60, description="Rest between sets in seconds")


class BuildInput(BaseModel):
    warmup: list[WorkoutExercise] = Field(description="2–3 warmup exercises")
    main: list[WorkoutExercise] = Field(description="Main workout block")
    cooldown: list[WorkoutExercise] = Field(description="1–2 cooldown exercises")


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool(args_schema=SearchInput)
def search_exercises(
    muscle_groups: list[str] | None = None,
    equipment: list[str] | None = None,
    movement_patterns: list[str] | None = None,
    limit: int = 10,
) -> str:
    """Search the exercise database by muscle group, available equipment, or movement pattern.
    Always call this before build_workout to get valid exercise IDs."""

    # Expand "full body" into representative muscle groups covering upper, lower, and core.
    if muscle_groups:
        expanded: list[str] = []
        for mg in muscle_groups:
            if mg.lower().strip() == "full body":
                expanded.extend(_FULL_BODY_MUSCLE_GROUPS)
            else:
                expanded.append(mg)
        muscle_groups = expanded or None

    # Detect bodyweight-only intent so we never fire the unrecognized-equipment
    # warning for phrases like "no equipment" or "bodyweight".
    bodyweight_only = bool(
        equipment and all(e.lower().strip() in _BODYWEIGHT_TERMS for e in equipment)
    )
    search_equipment = None if bodyweight_only else equipment

    # For bodyweight-only searches, fetch the full corpus before filtering so the
    # limit cap doesn't inadvertently exclude bodyweight exercises that appear later.
    search_limit = len(data.EXERCISES) if bodyweight_only else limit

    results = data.search(
        muscle_groups=muscle_groups,
        equipment=search_equipment,
        movement_patterns=movement_patterns,
        limit=search_limit,
    )

    if bodyweight_only:
        results = [e for e in results if not e.equipment_required][:limit]

    # Detect when the caller requested named equipment but none of it exists in
    # the database — search() silently falls back to bodyweight, so we surface it.
    if search_equipment:
        eq_lower = {e.lower() for e in search_equipment}
        equipment_matched = [
            ex for ex in results
            if ex.equipment_required
            and {r.lower() for r in ex.equipment_required} & eq_lower
        ]
        if not equipment_matched:
            available = sorted(data.KNOWN_EQUIPMENT)
            return json.dumps({
                "found": len(results),
                "equipment_warning": True,
                "unrecognized_equipment": equipment,
                "available_equipment": available,
                "bodyweight_exercises_available": len(results),
                "message": (
                    f"None of the requested equipment ({', '.join(equipment)}) exists in the "
                    "exercise database. Only bodyweight exercises are shown as fallback. "
                    "See 'available_equipment' for what the database supports."
                ),
            })

    if not results:
        return json.dumps({
            "found": 0,
            "message": (
                "No exercises matched those criteria. "
                "Try broader muscle groups, different equipment names, or remove filters."
            ),
        })

    return json.dumps({
        "found": len(results),
        "exercises": [
            {
                "id": e.id,
                "name": e.name,
                "muscle_groups": e.muscle_groups,
                "equipment_required": e.equipment_required,
                "is_reps": e.is_reps,
                "is_duration": e.is_duration,
                "supports_weight": e.supports_weight,
            }
            for e in results
        ],
    })


@tool(args_schema=BuildInput)
def build_workout(warmup: list, main: list, cooldown: list) -> str:
    """Assemble a structured workout (warmup / main / cooldown) from exercises
    returned by search_exercises. Pass exercise IDs exactly as returned."""
    sections: dict[str, list] = {}
    for label, items in [("warmup", warmup), ("main", main), ("cooldown", cooldown)]:
        resolved = []
        for raw in items:
            try:
                item = WorkoutExercise.model_validate(raw)
            except ValidationError as exc:
                return json.dumps({
                    "error": f"Invalid exercise schema in '{label}': {exc.error_count()} field error(s). "
                             "Each exercise needs exercise_id (string) and sets (int). "
                             f"Details: {exc.errors()[0]['msg']} on field '{exc.errors()[0]['loc'][0]}'."
                })
            ex = data.EXERCISE_BY_ID.get(item.exercise_id)
            if not ex:
                return json.dumps({
                    "error": (
                        f"Unknown exercise ID '{item.exercise_id}'. "
                        "Use search_exercises to obtain valid IDs."
                    )
                })
            resolved.append({
                "name": ex.name,
                "muscle_groups": ex.muscle_groups,
                "equipment": ex.equipment_required or ["bodyweight"],
                "sets": item.sets,
                "reps": item.reps,
                "duration_seconds": item.duration_seconds,
                "rest_seconds": item.rest_seconds,
            })
        sections[label] = resolved
    return json.dumps(sections, indent=2)


# ── Tool-calling graph (smart model) ──────────────────────────────────────────

_TOOLS = [search_exercises, build_workout]

_SYSTEM = SystemMessage(
    content=(
        "You are a fitness programming expert. Build workouts exclusively from exercises "
        "in the database — never invent exercises not returned by search_exercises.\n\n"
        "Workflow:\n"
        "1. Call search_exercises to find suitable exercises.\n"
        "2. If search_exercises returns 'equipment_warning: true', do NOT build a workout. "
        "Instead, tell the user that the equipment they specified is not in the database, "
        "give 4–6 representative examples from 'available_equipment', and ask them:\n"
        "   a) whether they have any of the listed equipment available, and\n"
        "   b) whether a bodyweight-only workout would work for them.\n"
        "Wait for their answer before proceeding.\n"
        "3. If search_exercises returns no results, tell the user what is unavailable and "
        "suggest alternatives (different equipment, broader muscle groups).\n"
        "4. Call build_workout with the selected exercise IDs to assemble the plan.\n"
        "5. Present the final workout clearly to the user."
    )
)


def _call_model(state: MessagesState) -> dict:
    response = get_llm().bind_tools(_TOOLS).invoke([_SYSTEM, *state["messages"]])
    return {"messages": [response]}


def _should_continue(state: MessagesState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


@lru_cache(maxsize=1)
def build():
    g = StateGraph(MessagesState)
    g.add_node("agent", _call_model)
    g.add_node("tools", ToolNode(_TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _should_continue)
    g.add_edge("tools", "agent")
    return g.compile()


# ── Pre-check + dispatch (fast model) ─────────────────────────────────────────

class _WorkoutContext(BaseModel):
    has_sufficient_context: bool = Field(
        description=(
            "True if the conversation gives enough context to search for and build a workout. "
            "The user must indicate at minimum a target muscle group, body area (e.g. upper body, "
            "legs, full body), or workout type. Equipment is NOT required — bodyweight is always valid."
        )
    )

_CONTEXT_SYSTEM = SystemMessage(
    content=(
        "Determine whether the conversation contains enough information to build a workout. "
        "The minimum requirement is some indication of target muscles or body area. "
        "Equipment is optional — a bodyweight workout is always possible."
    )
)


def run(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Pre-check (fast model) then tool-calling loop (smart model)."""
    assessment: _WorkoutContext = get_fast_llm().with_structured_output(_WorkoutContext).invoke(
        [_CONTEXT_SYSTEM, *state["messages"]]
    )

    if not assessment.has_sufficient_context:
        return {
            "response": (
                "I'd love to build a workout for you! I just need a couple of details:\n\n"
                "1. **Target muscles or body area** — e.g. chest, upper body, full body, legs\n"
                "2. **Available equipment** — or 'no equipment' for a bodyweight workout\n"
                "3. **Duration** *(optional)* — e.g. 30 minutes\n"
                "4. **Fitness level** *(optional)* — beginner, intermediate, or advanced"
            )
        }

    result = build().invoke(
        {"messages": state["messages"]},
        {**(config or {}), "recursion_limit": 12},
    )
    return {"response": result["messages"][-1].content}
