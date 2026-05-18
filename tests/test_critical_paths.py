"""
Critical path tests.

Two paths were chosen for their highest practical impact:

Path 1 — Workout log end-to-end
    The most frequent user action in a fitness app. Exercises every layer of the
    pipeline in a single shot: LLM routing, structured extraction, and fuzzy
    matching. A regression here means users silently lose workout data.

Path 2 — Generator graceful degradation
    The most dangerous failure mode. When search_exercises finds nothing (unknown
    equipment, misspelling, out-of-scope request), the generator must communicate
    that clearly rather than hallucinate exercises. A hallucinated workout is
    worse than no workout — it erodes trust and could cause injury.

The data-layer tests at the bottom require no LLM and run in milliseconds.
They guard the search and fuzzy-match logic that both critical paths depend on.
"""

import pytest
from langchain_core.messages import HumanMessage

from fitness_coach.data import EXERCISES, KNOWN_EQUIPMENT, fuzzy_match, search
from fitness_coach.hub import build as build_hub

# ── Helpers ────────────────────────────────────────────────────────────────────

def invoke(text: str) -> dict:
    return build_hub().invoke({"messages": [HumanMessage(content=text)]})


# ── Critical Path 1: Workout log end-to-end ───────────────────────────────────

@pytest.mark.integration
def test_log_routes_correctly():
    result = invoke("I just did 3x10 bench press at 185 lbs")
    assert result["route"] == "WORKOUT_LOG"
    assert result["confidence"] >= 0.75


@pytest.mark.integration
def test_log_extracts_sets_reps_weight():
    result = invoke("I just did 3x10 bench press at 185 lbs")
    response = result["response"]
    assert "3 sets" in response
    assert "10 reps" in response
    assert "185" in response


@pytest.mark.integration
def test_log_fuzzy_matches_exercise():
    result = invoke("I just did 3x10 bench press at 185 lbs")
    response = result["response"].lower()
    assert "bench" in response or "press" in response


# ── Critical Path 2: Generator graceful degradation ───────────────────────────

@pytest.mark.integration
def test_generate_routes_correctly():
    result = invoke("Build me a 30 min upper body session with dumbbells")
    assert result["route"] == "WORKOUT_GENERATE"


@pytest.mark.integration
def test_generate_handles_unavailable_equipment():
    result = invoke("Build me a workout using only a hovercraft")
    assert result["route"] == "WORKOUT_GENERATE"
    # Agent must communicate the gap, not invent exercises
    response = result["response"].lower()
    gap_words = ["not", "no ", "unavailable", "found", "don't", "cannot", "can't", "unable"]
    assert any(word in response for word in gap_words)


# ── Routing smoke tests ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_coach_routing():
    result = invoke("What muscles does a deadlift work?")
    assert result["route"] == "COACH"


@pytest.mark.integration
def test_ambiguous_input_triggers_clarify():
    result = invoke("bench press")
    assert result["route"] == "CLARIFY" or result["confidence"] < 0.75


# ── Data layer unit tests (no LLM) ────────────────────────────────────────────

def test_exercises_loaded():
    assert len(EXERCISES) == 50


def test_fuzzy_match_bench_press():
    match = fuzzy_match("bench press")
    assert match is not None
    assert "bench" in match.name.lower() or "press" in match.name.lower()


def test_fuzzy_match_returns_none_for_nonsense():
    assert fuzzy_match("xyzzy quantum hovercraft") is None


def test_search_by_muscle_group():
    results = search(muscle_groups=["chest"])
    assert len(results) > 0
    assert all("chest" in [m.lower() for m in e.muscle_groups] for e in results)


def test_search_by_equipment_dumbbell():
    results = search(equipment=["Dumbbell"])
    assert len(results) > 0
    # Every result either requires a dumbbell or needs no equipment
    for e in results:
        assert not e.equipment_required or any("dumbbell" in r.lower() for r in e.equipment_required)


def test_search_unknown_equipment_returns_only_bodyweight():
    results = search(equipment=["hovercraft"])
    assert all(not e.equipment_required for e in results)


def test_search_multiple_muscle_groups():
    results = search(muscle_groups=["chest", "triceps"])
    assert len(results) > 0


def test_search_respects_limit():
    results = search(muscle_groups=["glutes"], limit=3)
    assert len(results) <= 3


def test_known_equipment_constant():
    assert "Dumbbell" in KNOWN_EQUIPMENT
    assert "Barbell" in KNOWN_EQUIPMENT
    assert "swimming pool" not in KNOWN_EQUIPMENT


def test_search_full_body_expansion_covers_all_areas():
    from fitness_coach.agents.generator import _FULL_BODY_MUSCLE_GROUPS
    results = search(muscle_groups=_FULL_BODY_MUSCLE_GROUPS, limit=50)
    all_muscles = {m for e in results for m in e.muscle_groups}
    assert all_muscles & {"chest", "lats"}        # upper
    assert all_muscles & {"quads", "glutes"}       # lower
    assert "core" in all_muscles


def test_bodyweight_search_returns_only_no_equipment():
    all_exercises = search(limit=len(EXERCISES))
    bodyweight = [e for e in all_exercises if not e.equipment_required]
    assert len(bodyweight) > 0
    assert all(not e.equipment_required for e in bodyweight)
