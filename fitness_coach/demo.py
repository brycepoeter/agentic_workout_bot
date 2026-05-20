"""Interactive CLI demo — run with: python -m fitness_coach.demo"""
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from fitness_coach.config import get_trace_config, provider_label
from fitness_coach.hub import build

console = Console()

_EXAMPLES = [
    ("Coach",    "What muscles does a deadlift work?"),
    ("Generate", "Build me a 30 min upper body session with dumbbells"),
    ("Log",      "I just did 3x10 bench press at 185 lbs"),
    ("Ambiguous","bench press"),
]


_AFFIRMATIVE   = {"yes", "y", "yeah", "yep", "sure", "yup", "ok", "okay", "new", "fresh", "start fresh", "start over"}
_CONTINUATION  = {"no", "n", "nope", "nah", "continue", "same", "keep going", "same topic", "not new"}


def main() -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold green]Fitness Coach[/bold green] — Multi-Agent System\n"
        f"[dim]Provider:[/dim] [cyan]{provider_label()}[/cyan]\n\n"
        f"[dim]Type[/dim] [bold]examples[/bold] [dim]to see sample inputs,[/dim] "
        f"[bold]reset[/bold] [dim]to start a new conversation, or[/dim] [bold]exit[/bold] [dim]to quit.[/dim]",
        border_style="green",
    ))

    hub = build()
    history: list = []
    session_id = str(uuid4())  # groups all turns in this CLI session in Langfuse
    # When the hub detects a probable topic change it asks the user to confirm.
    # We hold the original message here until they answer.
    pending_message: str | None = None

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)

        if user_input.lower() == "examples":
            for label, text in _EXAMPLES:
                console.print(f"  [dim]{label}:[/dim] {text}")
            continue

        if user_input.lower() == "reset":
            history.clear()
            pending_message = None
            console.print("[dim]Conversation reset.[/dim]")
            continue

        # ── Topic-change confirmation ──────────────────────────────────────────
        # The previous turn flagged a likely topic change and asked the user.
        # Three cases for their reply:
        #   affirmative  → clear history, re-process the original held message
        #   continuation → keep history, re-process the original held message
        #   direct intent (e.g. "plan", "log it") → clear history, process their
        #                  reply directly as the new message
        if pending_message is not None:
            answer = user_input.lower().strip()
            if answer in _CONTINUATION:
                console.print("[dim]Continuing current conversation.[/dim]")
                user_input = pending_message
            elif answer in _AFFIRMATIVE:
                history.clear()
                console.print("[dim]Starting fresh.[/dim]")
                user_input = pending_message
            else:
                # User answered with direct intent — treat it as a fresh request
                history.clear()
                console.print("[dim]Starting fresh.[/dim]")
                # user_input stays as-is so their intent is processed directly
            pending_message = None

        history.append(HumanMessage(content=user_input))

        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            try:
                result = hub.invoke({"messages": history}, config=get_trace_config(session_id))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Error:[/red] {exc}")
                history.pop()  # don't keep a message that produced an error
                continue

        route      = result.get("route", "—")
        confidence = result.get("confidence")
        response   = result.get("response") or "[dim]No response generated.[/dim]"

        if route == "NEW_TOPIC":
            # Don't commit this exchange to history — it's a meta-conversation.
            # Hold the original message and re-process it after the user confirms.
            history.pop()
            pending_message = user_input
        else:
            history.append(AIMessage(content=response))

        meta = f"Route: [cyan]{route}[/cyan]"
        if confidence is not None:
            meta += f"  ·  Confidence: [cyan]{confidence:.0%}[/cyan]"
        console.print(f"\n[dim]{meta}[/dim]")
        console.print(Panel(response, border_style="green", title="[green]Coach[/green]"))


if __name__ == "__main__":
    main()
