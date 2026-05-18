# Fitness Coach — Multi-Agent System

A hub-and-spoke multi-agent fitness coaching system built with LangGraph. A router agent uses structured LLM output to dispatch user requests to one of three specialized sub-agents: a general coach, a workout generator, and a workout logger.

## Quick Start

**Prerequisite:** [Ollama](https://ollama.com/download) — for running the model locally. `make install` will prompt you to install [uv](https://docs.astral.sh/uv/getting-started/installation/) automatically if it's not found.

```bash
# 1. Install Python dependencies
make install

# 2. Download the local model (~4.4 GB, one-time)
make pull-model

# 3. Run the demo
make demo
```

No API key required. The system runs fully locally using [qwen2.5:1.5b](https://ollama.com/library/qwen2.5) via Ollama (~1 GB download).

> **Apple Silicon / GPU:** For better response quality, set `LOCAL_MODEL=qwen2.5:7b` in your `.env` before running `make pull-model`.
>
> **OpenAI:** Copy `.env.example` to `.env` and add `OPENAI_API_KEY`. The app detects it automatically and switches to GPT-4o.
>
> **Intel CPU without a GPU:** Response times with any local model will be slow (60–120 sec). Using the OpenAI path is strongly recommended.

Run `make` to see all available commands.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Hub (LangGraph StateGraph)                         │
│  • LLM-based router with structured output          │
│  • Confidence scoring + fallback handling           │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐  ┌──────────────┐  ┌───────────┐
      │  Coach  │  │   Workout    │  │  Workout  │
      │         │  │  Generator   │  │   Logger  │
      └─────────┘  └──────────────┘  └───────────┘
                         │
                   ┌─────┴──────┐
                   ▼            ▼
            search_exercises  build_workout
```

**Three routes:**

| Route | Example input |
|---|---|
| `COACH` | "What muscles does a deadlift work?" |
| `WORKOUT_GENERATE` | "Build me a 30 min upper body session with dumbbells" |
| `WORKOUT_LOG` | "I just did 3x10 bench press at 185 lbs" |

The router handles ambiguous inputs by emitting a confidence score. Low-confidence requests ask for clarification rather than silently misrouting.

## How I Would Evaluate This System in Production

**The router is the highest-leverage thing to monitor.** A misroute is the worst failure mode in this system — the user gets a response, so there's no error to catch, but it's from the wrong agent. I'd track the confidence score distribution on every request (p50, p95) and alert when the low-confidence rate climbs above ~10%. A spike there usually means users are phrasing requests in ways the router wasn't tested on, and it's a leading indicator of misroutes before users start complaining. Periodic spot-checks of a random sample of routed requests against their intended route is the only reliable way to catch the silent failures.

**For the workout generator, empty search results are the critical failure path.** If `search_exercises` returns nothing — because a user asked for equipment not in the dataset — the agent should degrade gracefully rather than hallucinate exercises. I'd monitor the empty-result rate as its own metric. A sustained rise means the exercise dataset has a coverage gap worth addressing. I'd also track the rate of invalid tool calls (malformed schemas from the LLM), which Pydantic catches at runtime but should never silently disappear — those exceptions belong in your error budget.

**For the workout logger**, fuzzy match confidence is the signal to watch. The matcher will always return *something*, so a false positive (matching "squat" to the wrong exercise) is harder to detect than a failure. Logging the match score alongside the matched exercise name lets you audit low-confidence matches and tune the threshold over time. If the unmatch rate is high, it means users are describing exercises in ways the dataset names don't anticipate — a content problem, not a code problem.

**On infrastructure:** this system runs locally via Ollama for development and evaluation, but in production I'd route through a hosted API (OpenAI, Anthropic, or AWS Bedrock) — the `OPENAI_API_KEY` environment variable already switches the app into that mode with no code changes. For observability, I'd add structured logging with a trace ID on every request so you can correlate the router's decision with each downstream tool call in a single query. LangSmith or Langfuse bolt onto LangGraph with minimal setup and give you full LLM call tracing for free, which is the fastest way to diagnose a misroute or a bad tool call in production.

## Development

```bash
make test      # run the test suite
make lint      # check code style
make format    # auto-fix formatting
```
