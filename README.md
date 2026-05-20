# Fitness Coach — Multi-Agent System

A hub-and-spoke multi-agent fitness coaching system built with LangGraph and LangChain. A router agent uses LLM structured output and confidence scoring to dispatch user requests to one of three specialized sub-agents.

---

## Demo

Check out a few of the routes in action!


https://github.com/user-attachments/assets/97b2d227-d310-49cb-b2bd-e87ac58fd85f




---

## Quick Start

**Prerequisite:** [Ollama](https://ollama.com/download) for local inference. `make install` will prompt you to install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it isn't found.

```bash
make install      # install dependencies
make pull-model   # download local model (~1 GB, one-time)
make demo         # interactive CLI
```

Or run the web UI:

```bash
make serve        # http://localhost:8000
```

Run `make` with no arguments to see all available commands.

### LLM Providers

The system is designed for OpenAI, which enables the two-tier model split (`gpt-4o` for reasoning, `gpt-4o-mini` for fast classification). Ollama is available as a local fallback if you prefer not to use an API key.

| Provider | Setup |
|---|---|
| **OpenAI (recommended)** | Add `OPENAI_API_KEY` to `.env`. The app detects it automatically and uses `gpt-4o` (smart tasks) and `gpt-4o-mini` (fast tasks). |
| **Ollama (local fallback)** | Run `make pull-model`. Default model: `qwen2.5:1.5b` (~1 GB). For better quality on Apple Silicon, set `LOCAL_MODEL=qwen2.5:7b` in `.env` before pulling. Both smart and fast roles use the same local model. |

> **Intel CPU without a GPU:** Any local model will be slow (60–120 sec per response). The OpenAI path is strongly recommended.

Copy `.env.example` to `.env` to see all configuration options.

---

## Architecture

```
User message
      │
      ▼
┌───────────────────────────────────────────┐
│  Router  (LLM structured output)          │
│  Emits route + confidence score.          │
│  Falls back to CLARIFY if score < 0.75.   │
└──┬──────────┬──────────┬──────────┬───────┘
   │          │          │          │
 COACH    GENERATE      LOG     CLARIFY /
                                NEW_TOPIC
   │          │          │          │
   └──────────┴──────────┴──────────┘
                    │
            Resolution check
            (does the response need
             a follow-up from the user?)
                    │
                   END
```

### Router

The router classifies every message using LLM structured output and emits both a route and a confidence score. Five possible routes:

| Route | Triggered by |
|---|---|
| `COACH` | Fitness questions, form advice, exercise education |
| `WORKOUT_GENERATE` | Requests to build or plan a workout |
| `WORKOUT_LOG` | Logging a completed exercise or workout |
| `CLARIFY` | Ambiguous input, or confidence below 0.75 |
| `NEW_TOPIC` | Clear shift to an unrelated subject mid-conversation |

### Sub-Agents

**Coach** — answers general fitness questions using the smart LLM.

**Workout Generator** — a tool-calling agent with two tools:
- `search_exercises` — queries the 50-exercise database by muscle group, equipment, or movement pattern. Expands "full body" automatically, handles bodyweight-only requests, and returns a structured warning when requested equipment isn't in the database so the agent can ask the user rather than hallucinate.
- `build_workout` — assembles a structured warmup / main / cooldown plan from exercise IDs returned by `search_exercises`. Returns a clean JSON error if the LLM provides an unknown ID or a malformed schema.

**Workout Logger** — extracts exercise name, sets, reps, and weight from natural language using the fast LLM, fuzzy-matches the exercise name against the database (e.g. "bench press" → "Barbell Flat Bench Press"), and returns a structured JSON log entry.

### Two-Tier LLM

Tasks that need reasoning use `get_llm()` (smart model). Tasks that need only classification or extraction use `get_fast_llm()` (fast model). Both are configured in `fitness_coach/config.py` and never imported directly in agent code.

| Role | Tasks |
|---|---|
| Smart | Coach, Workout Generator tool loop |
| Fast | Router, Logger, resolution check, generator pre-check |

### Observability (optional)

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`, every LLM call, tool call, and routing decision is traced automatically via [Langfuse](https://langfuse.com). All turns in a single CLI or web session are grouped under one session ID. See `.env.example` for configuration.

---

## Testing

```bash
make test-unit    # unit tests only — no LLM required, runs in < 1 second
make test         # full suite including integration tests (requires Ollama or OPENAI_API_KEY)
```

The unit tests cover the router's decision logic (mocked LLM), all data-layer search and fuzzy-match behavior, the logger's JSON output format, and `build_workout` error handling for bad exercise IDs and invalid schemas. Integration tests run the full pipeline end-to-end.

---

## How I Would Evaluate This System in Production

I would make heavy use of the Langfuse integration. That's where we'll see what routes users are hitting most often, what our costs and latency are, which types of questions get asked the most. This gives us insight into what type of answer caching strategy we might want to employ (e.g. if many users are asking for the same workouts, do we need to use the LLM each time?). This will also allow us to evaluate the two-tiered LLM solution. Is the speed and cost tradeoff worth it? Do we want to get more granular with how we choose models? Perhaps some models perform better in different agentic tasks and we could get even more intentional about which ones we use. 

This would also give us the chance to evaluate how many LLM calls we're making. My gut sense from looking at the Langfuse dashboard is that my app is making too many calls and is too expensive. It's only a few cents per session, but that adds up. Perhaps there are improvements to be made in the new topic recognition. Or perhaps there are other ways to streamline the call sequencing. But having our call traces there is what will empower us to analyze data about the user flows and make changes to improve the user experience and the cost to us as providers.

I would also evaluate the system along classic software engineering dimensions. How many requests is our API gateway handling? How are our containers performing? What are our cloud costs? What kind of latency are we experiencing in the non-AI portions of the app? These impact the user experience greatly, and if we only focus on the core AI portion, we will miss some of the opportunities to make it better. 
