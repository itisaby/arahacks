"""
AraFlow Demo — Simulate a realistic usage session to populate
telemetry data and show the dashboard in action.

Run:
  1. docker compose up -d        # Start Grafana stack
  2. sleep 5
  3. python demo.py              # Generate telemetry data
  4. open http://localhost:3000   # Grafana (admin / araflow)
"""

import time
import atexit
import threading
from app import (
    add_task, list_tasks, complete_task,
    add_note, search_notes, set_reminder,
    daily_summary, get_context_stats, get_workflow_traces,
    export_otel_dashboard, summarizer,
)
from telemetry import trace_chat_turn, register_summarizer_metrics, shutdown_telemetry
from dashboard import DashboardHandler
from http.server import HTTPServer

# Register summarizer for observable gauge metrics
register_summarizer_metrics(summarizer)

# Ensure telemetry flushes on exit (Ctrl+C)
atexit.register(shutdown_telemetry)


def run_demo():
    print("=== AraFlow Demo ===\n")

    # Simulate a morning routine
    trace_chat_turn("user", "Good morning! What do I have today?", token_count=12)

    print("1. Adding tasks...")
    add_task(title="Review ML paper on transformers", priority="high", due="today")
    add_task(title="Buy groceries", priority="medium", due="today")
    add_task(title="Reply to Prof. Smith's email", priority="high", due="today")
    add_task(title="Gym workout", priority="low", due="today")
    add_task(title="Prepare hackathon presentation", priority="high", due="today")

    print("2. Listing tasks...")
    tasks = list_tasks(show_done=False)
    print(f"   → {tasks['pending']} pending tasks")

    print("3. Adding notes...")
    add_note(content="Transformer paper key insight: attention is O(n²), consider linear attention alternatives", tags="research,ml")
    add_note(content="Grocery list: eggs, milk, bread, avocados, coffee", tags="shopping")
    add_note(content="Hackathon idea: use OTel spans to auto-suggest workflow optimizations", tags="hackathon,idea")
    add_note(content="Prof Smith wants meeting Thursday 2pm about thesis proposal", tags="university,important")

    print("4. Setting reminders...")
    set_reminder(message="Take a break and stretch", when="every 2 hours")
    set_reminder(message="Submit hackathon project", when="5:00 PM today")
    set_reminder(message="Call mom", when="7:00 PM today")

    # Simulate mid-day interactions to trigger summarization
    trace_chat_turn("user", "I just finished reviewing the transformer paper, mark it done", token_count=15)
    complete_task(task_id=1)

    trace_chat_turn("user", "Search my notes for anything about the hackathon", token_count=12)
    results = search_notes(query="hackathon")
    print(f"   → Found {results['count']} matching notes")

    # More interactions to build up context and trigger compression
    for i in range(8):
        summarizer.add_message("user", f"Quick update #{i+1}: Working on task {i+1}, making progress on the daily workflow")
        summarizer.add_message("assistant", f"Got it! Task {i+1} progress noted. Your token budget is being managed efficiently.")

    trace_chat_turn("user", "Give me my daily summary", token_count=8)
    print("\n5. Daily summary:")
    summary = daily_summary()
    print(f"   → Pending: {summary['tasks_pending']}, Completed: {summary['tasks_completed_today']}")
    print(f"   → Token usage: {summary['token_usage']}")
    print(f"   → Context stats: {summary['context_stats']}")

    print("\n6. Workflow analytics:")
    traces = get_workflow_traces()
    print(f"   → Total spans: {traces['total_spans']}")
    print(f"   → Tool analysis: {list(traces['tool_analysis'].keys())}")
    print(f"   → Optimization hints: {traces['optimization_hints']}")

    print("\n7. Context stats (token savings):")
    stats = get_context_stats()
    print(f"   → {stats}")

    # Export dashboard data
    export = export_otel_dashboard()
    print(f"\n8. Exported {len(export['spans'])} spans for visualization")

    return summary


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Make sure the Grafana stack is running first:   ║")
    print("║    docker compose up -d                          ║")
    print("║  Then open: http://localhost:3000                ║")
    print("║  Login: admin / araflow                          ║")
    print("╚══════════════════════════════════════════════════╝\n")

    print("Running demo simulation...\n")
    run_demo()

    print(f"\n{'='*50}")
    print("Starting dashboard on http://localhost:8050")
    print("Grafana dashboard at http://localhost:3000")
    print("Press Ctrl+C to stop")
    print(f"{'='*50}\n")

    HTTPServer(("", 8050), DashboardHandler).serve_forever()
