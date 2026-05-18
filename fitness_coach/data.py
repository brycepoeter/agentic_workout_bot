import json
from pathlib import Path

from pydantic import BaseModel
from rapidfuzz import fuzz, process

_DATA_PATH = Path(__file__).parent.parent / "exercises.json"


class Exercise(BaseModel):
    id: str
    name: str
    muscle_groups: list[str]
    joints_loaded: list[str]
    movement_patterns: list[str]
    equipment_required: list[str]
    is_bilateral: bool
    side: str | None
    priority_tier: int
    is_reps: bool
    is_duration: bool
    supports_weight: bool
    estimated_rep_duration: float
    bilateral_pair_id: str | None


def _load() -> list[Exercise]:
    with open(_DATA_PATH) as f:
        return [Exercise(**e) for e in json.load(f)]


EXERCISES: list[Exercise] = _load()
EXERCISE_BY_ID: dict[str, Exercise] = {e.id: e for e in EXERCISES}

# Precomputed at load time — used by fuzzy_match and search tooling
_EXERCISE_NAMES: list[str] = [e.name for e in EXERCISES]
KNOWN_EQUIPMENT: frozenset[str] = frozenset(
    item for e in EXERCISES for item in e.equipment_required
)


def search(
    muscle_groups: list[str] | None = None,
    equipment: list[str] | None = None,
    movement_patterns: list[str] | None = None,
    limit: int = 10,
) -> list[Exercise]:
    """Filter exercises by any combination of muscle groups, equipment, or movement patterns.

    Each filter is an OR match within its category — an exercise passes if it matches
    at least one of the supplied values. Bodyweight exercises (empty equipment_required)
    are always included when filtering by equipment.
    """
    results = list(EXERCISES)

    if muscle_groups:
        mg = {m.lower() for m in muscle_groups}
        results = [e for e in results if {m.lower() for m in e.muscle_groups} & mg]

    if equipment:
        eq = {e.lower() for e in equipment}
        results = [
            e for e in results
            if not e.equipment_required  # bodyweight always included
            or {r.lower() for r in e.equipment_required} & eq
        ]

    if movement_patterns:
        mp = {m.lower() for m in movement_patterns}
        results = [e for e in results if {m.lower() for m in e.movement_patterns} & mp]

    return results[:limit]


def fuzzy_match(name: str, threshold: int = 65) -> Exercise | None:
    """Return the best-matching Exercise for a user-supplied name, or None if below threshold."""
    result = process.extractOne(name, _EXERCISE_NAMES, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return EXERCISES[_EXERCISE_NAMES.index(result[0])]
    return None
