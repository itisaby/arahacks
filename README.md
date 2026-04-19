# AraFlow — Self-Optimizing AI Personal Assistant

> Built at Ara x Johns Hopkins Hackathon 2026

## What it does

AraFlow is a daily-life AI assistant that **gets smarter about itself**. It manages tasks, notes, and reminders while:

1. **Recursive Summarization** — Compresses conversation history hierarchically to stay within token budgets. Older messages get summarized, and summaries themselves get re-summarized as they accumulate.

2. **OpenTelemetry Instrumentation** — Every tool call and chat turn is traced as an OTel span. The assistant can analyze its own performance data and suggest workflow optimizations.

3. **Live Dashboard** — A real-time web UI shows token savings, tool performance, span timelines, and auto-generated optimization hints.

## Architecture

```
User ↔ Ara SDK (cloud runtime)
         ↓
   ┌─────────────────┐
   │   app.py         │  ← Automation + 10 traced tools
   │   ├─ telemetry   │  ← OTel spans + token accounting
   │   ├─ summarizer  │  ← Recursive context compression
   │   └─ dashboard   │  ← Live visualization
   └─────────────────┘
```

## Quick Start

```bash
pip install ara-sdk opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
ara auth login           # use access code: ARAHOPKINS
ara run app.py           # single cloud run
ara deploy app.py --cron "0 8 * * *"  # daily 8 AM automation
```

## Demo

```bash
python demo.py           # simulates a day of usage + starts dashboard
# open http://localhost:8050
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Ara automation — 10 tools for tasks, notes, reminders, analytics |
| `telemetry.py` | OTel tracing, span logging, token accounting |
| `recursive_summarizer.py` | Hierarchical context compression engine |
| `dashboard.py` | Live web dashboard for workflow visualization |
| `demo.py` | Demo script that populates data and launches dashboard |

## Key Innovation

Most AI assistants burn through tokens replaying full conversation history. AraFlow uses a **recursive summarization tree** — like a B-tree for conversation context — that keeps the active token window small while preserving important information. The OTel layer then lets you **see and optimize** how the assistant actually works.
