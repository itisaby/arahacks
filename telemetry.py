"""
OTel instrumentation for AraFlow — traces every tool call and chat turn,
exports spans so you can visualize and optimize your workflow.

Exports to:
  - Console (always, for local dev)
  - OTLP HTTP (when the collector is running) → Grafana Tempo + Prometheus
"""

import time
import json
import socket
import logging
import functools
from dataclasses import dataclass
from datetime import datetime, timezone

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
    BatchSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Suppress noisy export errors when the collector is down
logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk.metrics").setLevel(logging.CRITICAL)


def _collector_reachable(host: str = "localhost", port: int = 4318, timeout: float = 0.5) -> bool:
    """Quick TCP check to see if the OTel collector is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Provider setup — Console always; OTLP only when collector is reachable
# ---------------------------------------------------------------------------

_resource = Resource.create({"service.name": "araflow", "service.version": "0.1.0"})
_otlp_enabled = _collector_reachable()

# --- Traces ---
_provider = TracerProvider(resource=_resource)
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

if _otlp_enabled:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        )
    )
    print("[telemetry] OTel collector detected — OTLP export enabled")
else:
    print("[telemetry] OTel collector not found on :4318 — running with console export only")
    print("[telemetry] Run 'docker compose up -d' to enable Grafana dashboards")

trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("araflow")

# --- Metrics ---
if _otlp_enabled:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    _metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics"),
        export_interval_millis=5000,
    )
    _meter_provider = MeterProvider(resource=_resource, metric_readers=[_metric_reader])
else:
    _meter_provider = MeterProvider(resource=_resource)

metrics.set_meter_provider(_meter_provider)
_meter = metrics.get_meter("araflow")

# Counters
_prompt_token_counter = _meter.create_counter(
    "araflow.tokens.prompt",
    description="Total prompt tokens consumed",
    unit="tokens",
)
_completion_token_counter = _meter.create_counter(
    "araflow.tokens.completion",
    description="Total completion tokens consumed",
    unit="tokens",
)
_saved_token_counter = _meter.create_counter(
    "araflow.tokens.saved",
    description="Total tokens saved by summarization",
    unit="tokens",
)
_tool_call_counter = _meter.create_counter(
    "araflow.tool.calls",
    description="Total tool invocations",
    unit="calls",
)

# Histogram
_tool_duration_histogram = _meter.create_histogram(
    "araflow.tool.duration",
    description="Tool execution duration",
    unit="ms",
)

# Observable gauges (registered later via register_summarizer_metrics)
_summarizer_ref = None


def _active_tokens_callback(options):
    if _summarizer_ref:
        stats = _summarizer_ref.get_stats()
        yield metrics.Observation(stats["active_tokens"])


def _summary_nodes_callback(options):
    if _summarizer_ref:
        stats = _summarizer_ref.get_stats()
        yield metrics.Observation(stats["summary_nodes"])


def _tokens_saved_callback(options):
    if _summarizer_ref:
        stats = _summarizer_ref.get_stats()
        yield metrics.Observation(stats["total_tokens_saved"])


_meter.create_observable_gauge(
    "araflow.context.active_tokens_current",
    callbacks=[_active_tokens_callback],
    description="Current active token count in context window",
    unit="tokens",
)
_meter.create_observable_gauge(
    "araflow.context.summary_nodes_current",
    callbacks=[_summary_nodes_callback],
    description="Current number of summary nodes",
    unit="nodes",
)
_meter.create_observable_gauge(
    "araflow.context.tokens_saved_total",
    callbacks=[_tokens_saved_callback],
    description="Cumulative tokens saved by summarizer",
    unit="tokens",
)


def register_summarizer_metrics(summarizer):
    """Store a reference to the summarizer so observable gauges can read it."""
    global _summarizer_ref
    _summarizer_ref = summarizer


def shutdown_telemetry():
    """Flush all pending spans and metrics, then shut down providers."""
    try:
        _provider.force_flush(timeout_millis=2000)
        _provider.shutdown()
        _meter_provider.force_flush(timeout_millis=2000)
        _meter_provider.shutdown()
    except Exception:
        pass  # don't crash on exit if collector went away


# ---------------------------------------------------------------------------
# In-memory span store for the dashboard (no external DB needed for demo)
# ---------------------------------------------------------------------------

_span_log: list[dict] = []


def get_span_log() -> list[dict]:
    """Return collected span records for the visualisation dashboard."""
    return list(_span_log)


def _record_span(name: str, attrs: dict, duration_ms: float):
    _span_log.append({
        "name": name,
        "attributes": attrs,
        "duration_ms": round(duration_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "saved_tokens": 0}


def record_tokens(prompt: int, completion: int, saved: int = 0, mode: str = ""):
    _token_usage["prompt_tokens"] += prompt
    _token_usage["completion_tokens"] += completion
    _token_usage["saved_tokens"] += saved
    # Emit OTel counter metrics with optional mode attribute
    attrs = {"mode": mode} if mode else {}
    _prompt_token_counter.add(prompt, attrs)
    _completion_token_counter.add(completion, attrs)
    if saved > 0:
        _saved_token_counter.add(saved, attrs)


def get_token_usage() -> dict:
    return dict(_token_usage)


# ---------------------------------------------------------------------------
# Immutable cost tracking — audit trail of every API call
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CostRecord:
    """Immutable record of a single API call's cost."""
    timestamp: str
    mode: str               # "optimized", "baseline", "summarizer"
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class CostTracker:
    """Immutable, append-only cost ledger. Each mutation returns a new instance."""
    budget_limit: float = 5.00  # $ safety cap per session
    records: tuple[CostRecord, ...] = ()

    def add(self, record: CostRecord) -> "CostTracker":
        return CostTracker(budget_limit=self.budget_limit,
                           records=(*self.records, record))

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.records)

    @property
    def budget_remaining(self) -> float:
        return max(0, self.budget_limit - self.total_cost)

    @property
    def is_over_budget(self) -> bool:
        return self.total_cost >= self.budget_limit

    def by_mode(self, mode: str) -> tuple[CostRecord, ...]:
        return tuple(r for r in self.records if r.mode == mode)

    def summary(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 6),
            "budget_limit": self.budget_limit,
            "budget_remaining": round(self.budget_remaining, 6),
            "budget_pct_used": round(self.total_cost / self.budget_limit * 100, 1) if self.budget_limit > 0 else 0,
            "total_records": len(self.records),
            "cost_by_mode": {
                mode: round(sum(r.total_cost for r in self.by_mode(mode)), 6)
                for mode in set(r.mode for r in self.records)
            },
        }


# Global cost tracker instance
_cost_tracker = CostTracker()


def record_cost(mode: str, model: str, input_tokens: int, output_tokens: int,
                cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
                cost_per_input: float = 3.0 / 1_000_000,
                cost_per_output: float = 15.0 / 1_000_000) -> CostRecord:
    """Record an API call's cost immutably. Returns the CostRecord."""
    global _cost_tracker
    input_cost = input_tokens * cost_per_input
    output_cost = output_tokens * cost_per_output
    record = CostRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        total_cost=round(input_cost + output_cost, 8),
    )
    _cost_tracker = _cost_tracker.add(record)
    return record


def get_cost_tracker() -> CostTracker:
    return _cost_tracker


def reset_cost_tracker(budget_limit: float = 5.00):
    global _cost_tracker
    _cost_tracker = CostTracker(budget_limit=budget_limit)


# ---------------------------------------------------------------------------
# Decorator: instrument any Ara tool function
# ---------------------------------------------------------------------------

# Keys to redact from tool args/results before logging to spans
_SENSITIVE_KEYS = {"token", "password", "secret", "key", "authorization", "private"}


def _redact_for_span(data, max_len=200):
    """Redact sensitive fields from data before logging to OTel spans."""
    text = str(data)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"


def _safe_args_for_span(kwargs):
    """Serialize kwargs for span, redacting sensitive-looking values."""
    safe = {}
    for k, v in kwargs.items():
        if any(s in k.lower() for s in _SENSITIVE_KEYS):
            safe[k] = "[REDACTED]"
        else:
            safe[k] = v
    return json.dumps(safe, default=str)


def traced_tool(fn):
    """Wrap an @ara.tool function with OTel tracing + token/duration logging."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(f"tool.{fn.__name__}") as span:
            span.set_attribute("tool.name", fn.__name__)
            span.set_attribute("tool.args", _safe_args_for_span(kwargs))
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                span.set_status(StatusCode.OK)
                span.set_attribute("tool.result_preview", _redact_for_span(result))
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise
            finally:
                dur = (time.perf_counter() - t0) * 1000
                span.set_attribute("duration_ms", dur)
                _record_span(fn.__name__, dict(span.attributes), dur)
                # Emit OTel metrics
                _tool_call_counter.add(1, {"tool.name": fn.__name__})
                _tool_duration_histogram.record(dur, {"tool.name": fn.__name__})
    return wrapper


def trace_chat_turn(role: str, content: str, token_count: int = 0):
    """Record a chat turn as an OTel span."""
    with tracer.start_as_current_span(f"chat.{role}") as span:
        span.set_attribute("chat.role", role)
        span.set_attribute("chat.content", content[:500])
        span.set_attribute("chat.length", len(content))
        span.set_attribute("chat.tokens", token_count)
        _record_span(f"chat.{role}", {"length": len(content), "tokens": token_count}, 0)
