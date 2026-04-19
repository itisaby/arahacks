# AraFlow — Self-Optimizing AI Personal Assistant

> Built at Ara x Johns Hopkins Hackathon 2026

## What it does

AraFlow is a daily-life AI assistant that **gets smarter about itself**. It manages tasks, notes, and reminders while:

1. **Recursive Summarization** — Compresses conversation history hierarchically to stay within token budgets. Older messages get summarized, and summaries themselves get re-summarized as they accumulate.

2. **OpenTelemetry Instrumentation** — Every tool call and chat turn is traced as an OTel span. The assistant can analyze its own performance data and suggest workflow optimizations.

3. **Live Dashboard** — A real-time web UI shows token savings, tool performance, span timelines, and auto-generated optimization hints.

## Architecture

```mermaid
graph TD
    User([fa:fa-user User])

    subgraph Browser["Browser — Comparison UI"]
        Optimized["Optimized Pane<br/><i>compressed context</i>"]
        Baseline["Baseline Pane<br/><i>raw context</i>"]
        Council["Council Mode<br/><i>3-persona pipeline</i>"]
    end

    subgraph Backend["Python Backend"]
        CompPy["comparison.py<br/><i>dual-pane server + parallel execution</i>"]
        CouncilPy["council.py<br/><i>Pragmatist → Critic → Synthesizer</i>"]
        Summarizer["recursive_summarizer.py<br/><i>L0 → L1 → L2 compression</i>"]
        Telemetry["telemetry.py<br/><i>OTel spans + metrics</i>"]
        ToolLoop["Multi-Turn Tool Use<br/><i>tool_use → tool_result loop</i>"]
        WebSearch["web_search tool<br/><i>DuckDuckGo API</i>"]
    end

    subgraph ClaudeFeatures["Claude API Features"]
        Claude["Claude API<br/><i>Anthropic</i>"]
        PromptCache["Prompt Caching<br/><i>cache_control: ephemeral</i>"]
        ModelRoute["Model Routing<br/><i>Haiku for speed · Sonnet for quality</i>"]
        CacheMetrics["Cache Metrics<br/><i>cache_read + cache_creation tokens</i>"]
    end

    subgraph External["External Services"]
        DDG["DuckDuckGo<br/><i>Instant Answer API</i>"]
    end

    subgraph Observability["Observability Stack"]
        CostTracker["Cost & Budget Tracker<br/><i>per-call CostRecord + budget limit</i>"]
        TokenAcct["Token Accounting<br/><i>prompt · completion · saved counters</i>"]
        OTEL["OTel Collector"]
        Prometheus["Prometheus"]
        Tempo["Tempo"]
        Grafana["Grafana Dashboards"]
    end

    User --> Browser
    Optimized --> CompPy
    Baseline --> CompPy
    Council --> CouncilPy

    CompPy --> Summarizer
    CompPy --> ToolLoop
    ToolLoop --> Claude
    ToolLoop --> WebSearch
    WebSearch --> DDG

    CompPy --> Claude
    CouncilPy --> Claude
    Claude --> PromptCache
    Claude --> CacheMetrics
    CouncilPy --> ModelRoute
    ModelRoute --> Claude

    CompPy --> Telemetry
    CouncilPy --> Telemetry
    Telemetry --> CostTracker
    Telemetry --> TokenAcct
    Telemetry --> OTEL
    OTEL --> Prometheus
    OTEL --> Tempo
    Prometheus --> Grafana
    Tempo --> Grafana

    style Browser fill:#e8f4fd,stroke:#2196F3
    style Backend fill:#e8f5e9,stroke:#4CAF50
    style ClaudeFeatures fill:#fff3e0,stroke:#FF9800
    style External fill:#f3e5f5,stroke:#9C27B0
    style Observability fill:#fce4ec,stroke:#E91E63
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
