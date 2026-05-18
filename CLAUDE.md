# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a take-home assignment to build a **fitness coaching multi-agent system** using Python, LangGraph, and LangChain. The package lives in `fitness_coach/`. Exercise seed data is in `exercises.json`.

## LLM Provider

`fitness_coach/config.py` is the single place that selects the LLM. Call `get_llm()` everywhere — never import a provider directly in agent/tool code.

- **Default (no config needed):** Ollama running `qwen2.5:7b` locally.
- **OpenAI:** set `OPENAI_API_KEY` in `.env` and the app switches automatically.

`provider_label()` returns a human-readable string (`"Ollama (qwen2.5:7b)"` or `"OpenAI (gpt-4o)"`) useful for startup logging.

## Commands

```bash
make install      # uv sync --all-extras
make pull-model   # ollama pull qwen2.5:7b  (~4.4 GB, one-time)
make demo         # python -m fitness_coach.demo
make serve        # uvicorn fitness_coach.web:app --reload
make test         # pytest -v
make lint         # ruff check
make format       # ruff check --fix && ruff format
```

## Required Architecture

### Hub-and-Spoke Graph
- A LangGraph `StateGraph` with typed state acts as the hub/router
- The hub routes to three sub-agents: `COACH`, `WORKOUT_GENERATE`, `WORKOUT_LOG`
- **Routing must use LLM structured output** (`with_structured_output()`) — not regex or keyword matching
- Router must emit a confidence score; low-confidence inputs should trigger clarification or a fallback route (the approach must be explicit, not silent)
- Sub-agents are separate graphs composed into the hub — not inlined functions

### Sub-Agents

**Workout Generator** (tool-calling agent):
- `search_exercises` tool — queries `exercises.json` by muscle groups, equipment, or movement patterns
- `build_workout` tool — assembles warmup/main/cooldown with sets, reps, and rest from selected exercises
- If `search_exercises` returns no results (e.g. unavailable equipment), the agent must recover gracefully — no crashes or hallucinated exercises

**Workout Logger**:
- Parses exercise name, sets, reps, and weight from natural language
- Fuzzy-matches user input against `exercises.json` names (e.g. "bench press" → "Barbell Flat Bench Press")
- Returns structured JSON log entries

**Coach**:
- Answers general fitness questions (e.g. "What muscles does a deadlift work?")

### Pydantic Schemas
All tool inputs must have Pydantic schemas with field descriptions.

## Exercise Data (`exercises.json`)

50 exercises. Key fields:
- `id` — UUID, used as the canonical exercise identifier
- `muscle_groups` — list of strings (e.g. `["chest", "triceps"]`)
- `joints_loaded` — joints stressed; relevant for injury avoidance
- `movement_patterns` — movement taxonomy (e.g. `"upper push - horizontal"`, `"lower push - squat"`)
- `equipment_required` — list of equipment strings; empty list means bodyweight
- `priority_tier` — all currently `2`
- `is_bilateral` — if `true`, `side` is set (e.g. `"left_arm"`) and `bilateral_pair_id` points to the paired exercise UUID
- `is_reps` / `is_duration` — whether the exercise supports rep-based or time-based tracking
- `supports_weight` — whether external load applies
- `estimated_rep_duration` — seconds per rep (0 for duration-only exercises)

## Requirements Checklist (from assignment)

1. Hub is a LangGraph `StateGraph` with typed state and explicit edges
2. Sub-agents are separate composed graphs
3. Tools have Pydantic input schemas with field descriptions
4. At least 2 critical path tests — document why those paths were chosen
5. Runnable demo or transcript (simple web view acceptable)
6. README section: "How I would evaluate this system in production" (metrics, failure modes, health signals)
7. Submit as a GitHub repo

## Stretch Goals (optional, noted in assignment)
- Streaming support
- Multi-turn conversation memory
- Injury avoidance using `joints_loaded`
- Bilateral exercise auto-pairing via `bilateral_pair_id`
- Observability (Langfuse, OpenTelemetry, or structured logging)
