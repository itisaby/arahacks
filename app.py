"""
AraFlow — Self-Optimizing AI Personal Assistant
================================================
A daily-life assistant that:
  1. Manages tasks, reminders, notes, and daily summaries
  2. Saves tokens via recursive hierarchical summarization
  3. Instruments every action with OpenTelemetry for workflow visualization

Deploy: ara deploy app.py --cron "0 8 * * *"   (daily 8 AM run)
Run:    ara run app.py
"""

import sys
import os
import json
from datetime import datetime, timezone

# Ensure the script's directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ara_sdk as ara

from telemetry import traced_tool, trace_chat_turn, get_span_log, get_token_usage
from recursive_summarizer import RecursiveSummarizer
from council import council_engine, PERSONAS

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

summarizer = RecursiveSummarizer(chunk_size=6, max_summary_tokens=200)

# Simple in-memory stores (Ara sandbox persists across turns)
_tasks: list[dict] = []
_notes: list[dict] = []
_reminders: list[dict] = []

# ---------------------------------------------------------------------------
# Tools: Daily Life Management
# ---------------------------------------------------------------------------

@ara.tool
@traced_tool
def add_task(title: str, priority: str = "medium", due: str = "") -> dict:
    """Add a task to your personal task list."""
    task = {
        "id": len(_tasks) + 1,
        "title": title,
        "priority": priority,
        "due": due,
        "done": False,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _tasks.append(task)
    summarizer.add_message("system", f"Task added: {title} (priority={priority})")
    return {"status": "added", "task": task}


@ara.tool
@traced_tool
def list_tasks(show_done: bool = False) -> dict:
    """List all tasks, optionally including completed ones."""
    visible = _tasks if show_done else [t for t in _tasks if not t["done"]]
    return {"tasks": visible, "total": len(_tasks), "pending": sum(1 for t in _tasks if not t["done"])}


@ara.tool
@traced_tool
def complete_task(task_id: int) -> dict:
    """Mark a task as done."""
    for t in _tasks:
        if t["id"] == task_id:
            t["done"] = True
            summarizer.add_message("system", f"Task completed: {t['title']}")
            return {"status": "completed", "task": t}
    return {"status": "not_found"}


@ara.tool
@traced_tool
def add_note(content: str, tags: str = "") -> dict:
    """Save a quick note with optional comma-separated tags."""
    note = {
        "id": len(_notes) + 1,
        "content": content,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _notes.append(note)
    summarizer.add_message("system", f"Note saved: {content[:50]}...")
    return {"status": "saved", "note": note}


@ara.tool
@traced_tool
def search_notes(query: str) -> dict:
    """Search notes by keyword or tag."""
    results = []
    q = query.lower()
    for n in _notes:
        if q in n["content"].lower() or q in [t.lower() for t in n["tags"]]:
            results.append(n)
    return {"results": results, "count": len(results)}


@ara.tool
@traced_tool
def set_reminder(message: str, when: str) -> dict:
    """Set a reminder. 'when' can be a natural description like 'tomorrow 9am'."""
    reminder = {
        "id": len(_reminders) + 1,
        "message": message,
        "when": when,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _reminders.append(reminder)
    summarizer.add_message("system", f"Reminder set: {message} at {when}")
    return {"status": "set", "reminder": reminder}


@ara.tool
@traced_tool
def daily_summary() -> dict:
    """Generate a summary of your day — tasks, notes, reminders, and token savings."""
    pending = [t for t in _tasks if not t["done"]]
    completed = [t for t in _tasks if t["done"]]
    upcoming = _reminders[-5:]  # most recent reminders
    tokens = get_token_usage()
    ctx_stats = summarizer.get_stats()

    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "tasks_pending": len(pending),
        "tasks_completed_today": len(completed),
        "top_pending": [t["title"] for t in pending[:5]],
        "recent_notes": len(_notes),
        "upcoming_reminders": [r["message"] for r in upcoming],
        "token_usage": tokens,
        "context_stats": ctx_stats,
    }


# ---------------------------------------------------------------------------
# Tools: Token Optimization & Observability
# ---------------------------------------------------------------------------

@ara.tool
@traced_tool
def get_context_stats() -> dict:
    """Show how many tokens the recursive summarizer has saved."""
    stats = summarizer.get_stats()
    stats["token_usage"] = get_token_usage()
    return stats


@ara.tool
@traced_tool
def get_workflow_traces() -> dict:
    """Return OTel span data for workflow visualization and optimization."""
    spans = get_span_log()
    # compute per-tool stats
    tool_stats: dict[str, list[float]] = {}
    for s in spans:
        name = s["name"]
        tool_stats.setdefault(name, []).append(s["duration_ms"])

    analysis = {}
    for name, durations in tool_stats.items():
        analysis[name] = {
            "call_count": len(durations),
            "avg_ms": round(sum(durations) / len(durations), 2),
            "max_ms": round(max(durations), 2),
            "total_ms": round(sum(durations), 2),
        }

    return {
        "total_spans": len(spans),
        "tool_analysis": analysis,
        "recent_spans": spans[-10:],
        "optimization_hints": _generate_hints(analysis),
    }


def _generate_hints(analysis: dict) -> list[str]:
    """Auto-generate optimization suggestions from trace data."""
    hints = []
    for name, stats in analysis.items():
        if stats["avg_ms"] > 500:
            hints.append(f"'{name}' averages {stats['avg_ms']}ms — consider caching results")
        if stats["call_count"] > 10:
            hints.append(f"'{name}' called {stats['call_count']} times — consider batching")
    if not hints:
        hints.append("Workflow looks efficient! No bottlenecks detected.")
    return hints


@ara.tool
@traced_tool
def export_otel_dashboard() -> dict:
    """Export a JSON snapshot of all telemetry data for external visualization."""
    return {
        "spans": get_span_log(),
        "token_usage": get_token_usage(),
        "context_stats": summarizer.get_stats(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Tools: LLM Council — Multi-Persona Debate
# ---------------------------------------------------------------------------

@ara.tool
@traced_tool
def start_council(topic: str, context: str = "", personas: str = "", rounds: int = 3) -> dict:
    """Start an LLM Council debate on a topic.

    Multiple AI personas debate the topic across rounds to find the best solution.
    Args:
        topic: The question or problem to debate
        context: Optional background info from the user
        personas: Comma-separated persona IDs (pragmatist,critic,visionary,synthesizer).
                  Leave empty for all four.
        rounds: Number of debate rounds (default 3)
    """
    persona_ids = [p.strip() for p in personas.split(",") if p.strip()] if personas else None
    result = council_engine.create_debate_prompt(
        topic=topic,
        persona_ids=persona_ids,
        num_rounds=min(rounds, 5),
        user_context=context,
    )
    summarizer.add_message("system", f"Council started on: {topic}")
    return result


@ara.tool
@traced_tool
def list_council_personas() -> dict:
    """Show all available council personas and their thinking styles."""
    return {
        "personas": {
            pid: {"name": p["name"], "emoji": p["emoji"], "style": p["style"]}
            for pid, p in PERSONAS.items()
        },
        "usage": "Pass persona IDs (pragmatist, critic, visionary, synthesizer) to start_council",
    }


@ara.tool
@traced_tool
def get_council_history() -> dict:
    """List all past council debate sessions."""
    return {"sessions": council_engine.list_sessions()}


# ---------------------------------------------------------------------------
# Automation definition
# ---------------------------------------------------------------------------

ara.Automation(
    "araflow-assistant",
    system_instructions="""You are AraFlow, a self-optimizing personal AI assistant.

Your capabilities:
- **Task Management**: Add, list, complete tasks with priorities and due dates
- **Notes**: Save and search notes with tags
- **Reminders**: Set reminders for future events
- **Daily Summary**: Provide end-of-day overviews
- **Token Optimization**: You use recursive summarization to keep conversation
  context compact. Mention token savings when relevant.
- **Workflow Analytics**: You can show OTel trace data so the user can see
  which tools are slow, frequently called, or could be optimized.
- **LLM Council**: When the user wants a deep discussion or needs to weigh
  multiple perspectives on a hard problem, use start_council. This activates
  a panel of AI personas — The Pragmatist, The Critic, The Visionary, and
  The Synthesizer — who debate the topic across multiple rounds. You must
  role-play each persona in order, staying in character for each one.
  After all rounds, The Synthesizer delivers a final verdict with concrete
  next steps. Format the debate clearly with headers and persona labels.
- **GitHub Integration**: You can access the user's GitHub repositories,
  issues, pull requests, and notifications via the connected GitHub account.
  Use this to check repo status, review PRs, or summarize recent activity.
- **Messages Integration**: You can read and send messages via the connected
  messaging account. Use this to check unread messages, send replies, or
  summarize conversations.

Behavior guidelines:
- Be concise and action-oriented
- Proactively suggest optimizations when you see patterns in the trace data
- When the conversation is long, mention how many tokens have been saved
- If the user wants to build something, show them the workflow traces to
  help them understand and optimize their process
- When the user says "council", "debate", "deep discussion", or "multiple
  perspectives", automatically invoke the LLM Council
- When the user asks about their GitHub repos, PRs, issues, or messages,
  use the connected integrations to fetch live data
""",
    tools=[
        add_task,
        list_tasks,
        complete_task,
        add_note,
        search_notes,
        set_reminder,
        daily_summary,
        get_context_stats,
        get_workflow_traces,
        export_otel_dashboard,
        start_council,
        list_council_personas,
        get_council_history,
    ],
    skills=[
        ara.connectors.github,
        ara.connectors.messages,
    ],
)
