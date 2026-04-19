"""
Dual-Pane Chat Comparison UI — shows AraFlow's token savings vs a naive baseline.

Sends each user message to Claude API twice in parallel:
  - Optimized: uses RecursiveSummarizer to compress context
  - Baseline: sends full uncompressed chat history

Serves a dark-themed comparison UI on :8060.
"""

import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

from recursive_summarizer import RecursiveSummarizer
from telemetry import record_tokens, record_cost, get_cost_tracker, reset_cost_tracker, tracer, register_summarizer_metrics
from council import run_live_council, _MODEL_PRICING

# ---------------------------------------------------------------------------
# Claude API pricing (Sonnet per-token costs as of 2025)
# ---------------------------------------------------------------------------
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000   # $3.00 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # $15.00 per 1M output tokens

MODEL = os.environ.get("ARAFLOW_MODEL", "claude-sonnet-4-20250514")
MODEL_CHEAP = os.environ.get("ARAFLOW_MODEL_CHEAP", "claude-haiku-4-5-20251001")
USE_LLM_SUMMARIZER = os.environ.get("ARAFLOW_LLM_SUMMARIZER", "1") == "1"

# ---------------------------------------------------------------------------
# Web search tool (DuckDuckGo instant answer API — no key needed)
# ---------------------------------------------------------------------------
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information. Use this when the user asks about recent events, news, current data, or anything that might need up-to-date information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            }
        },
        "required": ["query"],
    },
}


@traced_tool
def web_search(query: str) -> dict:
    """Search DuckDuckGo instant answer API and return results."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "AraFlow/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}

    results = []
    # Abstract (main answer)
    if data.get("Abstract"):
        results.append({
            "title": data.get("Heading", ""),
            "snippet": data["Abstract"],
            "url": data.get("AbstractURL", ""),
        })
    # Related topics
    for topic in (data.get("RelatedTopics") or [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })
    # If no results from instant answer, provide a fallback
    if not results:
        results.append({
            "title": f"Web search: {query}",
            "snippet": f"DuckDuckGo did not return an instant answer for '{query}'. The query was executed but no structured results were available. Try answering based on your knowledge.",
            "url": f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
        })

    return {"query": query, "results": results}


MAX_TOOL_ROUNDS = 3  # prevent infinite tool-use loops


def _run_tool_use_loop(client, model, system, messages, tools):
    """Run Claude with tools, handling tool_use blocks in a loop.

    Returns (reply_text, total_input_tokens, total_output_tokens, tool_calls_made).
    """
    total_input = 0
    total_output = 0
    tool_calls_made = 0
    current_messages = list(messages)

    for _round in range(MAX_TOOL_ROUNDS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=current_messages,
            tools=tools,
        )

        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        # Check if Claude wants to use a tool
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks or _round >= MAX_TOOL_ROUNDS:
            # No tool calls or max rounds reached — extract final text
            reply = text_blocks[0].text if text_blocks else ""
            # Extract cache metrics from last response
            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            return reply, total_input, total_output, tool_calls_made, cache_read, cache_creation

        # Execute tool calls
        # Build assistant message with all content blocks
        current_messages.append({"role": "assistant", "content": response.content})

        # Process each tool_use block
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_calls_made += 1
            if tool_block.name == "web_search":
                result = web_search(query=tool_block.input.get("query", ""))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result),
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps({"error": f"Unknown tool: {tool_block.name}"}),
                    "is_error": True,
                })

        current_messages.append({"role": "user", "content": tool_results})

    # Fallback (shouldn't reach here)
    return "", total_input, total_output, tool_calls_made, 0, 0


# ---------------------------------------------------------------------------
# LLM-powered summarizer function (uses cheap model)
# ---------------------------------------------------------------------------
def _llm_summarize(text: str) -> str:
    """Summarize text using the cheap model (Haiku). Used by RecursiveSummarizer."""
    client = _get_client()
    response = client.messages.create(
        model=MODEL_CHEAP,
        max_tokens=300,
        system=[{
            "type": "text",
            "text": "You are a context compressor. Summarize the conversation chunk into a concise paragraph preserving key facts, decisions, action items, and entities. Be extremely concise.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": text}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_summarizer = RecursiveSummarizer(
    chunk_size=2, max_summary_tokens=200,
    llm_summarize_fn=_llm_summarize if USE_LLM_SUMMARIZER else None,
)
register_summarizer_metrics(_summarizer)
_baseline_history: list[dict] = []
_cumulative = {
    "optimized_input": 0,
    "optimized_output": 0,
    "baseline_input": 0,
    "baseline_output": 0,
    "turns": 0,
}
# Per-turn token history for the divergence chart
_turn_history: list[dict] = []

# ---------------------------------------------------------------------------
# API client (lazy init so import doesn't crash without key)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# Chat engines
# ---------------------------------------------------------------------------

def _call_optimized(user_message: str) -> dict:
    """Send compressed context to Claude via RecursiveSummarizer."""
    _summarizer.add_message("user", user_message)
    context = _summarizer.get_context()

    # Count what AraFlow is actually sending
    summary_messages = [m for m in context if m["content"].startswith("[Summary L")]
    raw_messages = [m for m in context if not m["content"].startswith("[Summary L")]

    messages = [{"role": m["role"] if m["role"] != "system" else "user", "content": m["content"]} for m in context]
    # Merge consecutive same-role messages (API requirement)
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))

    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": "You are a helpful assistant. Be concise.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=merged,
    )

    reply = response.content[0].text
    _summarizer.add_message("assistant", reply)

    # Extract cache metrics if available
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

    stats = _summarizer.get_stats()

    return {
        "reply": reply,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "pipeline": {
            "summary_nodes": len(summary_messages),
            "raw_messages": len(raw_messages),
            "total_context_items": len(context),
            "summarizer_stats": stats,
            "summaries_preview": [m["content"][:120] for m in summary_messages],
        },
    }


def _call_baseline(user_message: str) -> dict:
    """Send full uncompressed history to Claude."""
    _baseline_history.append({"role": "user", "content": user_message})

    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": "You are a helpful assistant. Be concise.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=list(_baseline_history),
    )

    reply = response.content[0].text
    _baseline_history.append({"role": "assistant", "content": reply})

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

    return {
        "reply": reply,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "pipeline": {
            "total_messages": len(_baseline_history),
        },
    }


def _handle_chat(message: str) -> dict:
    """Run both engines in parallel and return comparison results."""
    with tracer.start_as_current_span("comparison.turn") as span:
        span.set_attribute("message", message[:200])

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_opt = pool.submit(_call_optimized, message)
            fut_base = pool.submit(_call_baseline, message)
            optimized = fut_opt.result()
            baseline = fut_base.result()

    # Record telemetry
    record_tokens(optimized["input_tokens"], optimized["output_tokens"], mode="optimized")
    record_tokens(baseline["input_tokens"], baseline["output_tokens"], mode="baseline")

    # Immutable cost records
    record_cost("optimized", MODEL, optimized["input_tokens"], optimized["output_tokens"],
                optimized.get("cache_read_tokens", 0), optimized.get("cache_creation_tokens", 0))
    record_cost("baseline", MODEL, baseline["input_tokens"], baseline["output_tokens"],
                baseline.get("cache_read_tokens", 0), baseline.get("cache_creation_tokens", 0))

    # Update cumulative stats
    with _lock:
        _cumulative["optimized_input"] += optimized["input_tokens"]
        _cumulative["optimized_output"] += optimized["output_tokens"]
        _cumulative["baseline_input"] += baseline["input_tokens"]
        _cumulative["baseline_output"] += baseline["output_tokens"]
        _cumulative["turns"] += 1

    # Record per-turn history for divergence chart
    with _lock:
        _turn_history.append({
            "turn": _cumulative["turns"],
            "optimized_input": optimized["input_tokens"],
            "baseline_input": baseline["input_tokens"],
            "cum_optimized": _cumulative["optimized_input"] + _cumulative["optimized_output"],
            "cum_baseline": _cumulative["baseline_input"] + _cumulative["baseline_output"],
        })

    tokens_saved = (baseline["input_tokens"] + baseline["output_tokens"]) - \
                   (optimized["input_tokens"] + optimized["output_tokens"])
    baseline_total = baseline["input_tokens"] + baseline["output_tokens"]
    savings_pct = (tokens_saved / baseline_total * 100) if baseline_total > 0 else 0

    cost_optimized = optimized["input_tokens"] * COST_PER_INPUT_TOKEN + \
                     optimized["output_tokens"] * COST_PER_OUTPUT_TOKEN
    cost_baseline = baseline["input_tokens"] * COST_PER_INPUT_TOKEN + \
                    baseline["output_tokens"] * COST_PER_OUTPUT_TOKEN

    return {
        "optimized": optimized,
        "baseline": baseline,
        "tokens_saved": tokens_saved,
        "savings_pct": round(savings_pct, 1),
        "cost_optimized": round(cost_optimized, 6),
        "cost_baseline": round(cost_baseline, 6),
        "cost_saved": round(cost_baseline - cost_optimized, 6),
    }


def _handle_solo_optimized(message: str) -> dict:
    """Run only the optimized engine and return results."""
    with tracer.start_as_current_span("comparison.solo_optimized") as span:
        span.set_attribute("message", message[:200])
        optimized = _call_optimized(message)

    record_tokens(optimized["input_tokens"], optimized["output_tokens"], mode="optimized")
    record_cost("optimized", MODEL, optimized["input_tokens"], optimized["output_tokens"],
                optimized.get("cache_read_tokens", 0), optimized.get("cache_creation_tokens", 0))

    with _lock:
        _cumulative["optimized_input"] += optimized["input_tokens"]
        _cumulative["optimized_output"] += optimized["output_tokens"]
        _cumulative["turns"] += 1

        _turn_history.append({
            "turn": _cumulative["turns"],
            "optimized_input": optimized["input_tokens"],
            "baseline_input": 0,
            "cum_optimized": _cumulative["optimized_input"] + _cumulative["optimized_output"],
            "cum_baseline": _cumulative["baseline_input"] + _cumulative["baseline_output"],
        })

    return {"optimized": optimized}


def _handle_solo_baseline(message: str) -> dict:
    """Run only the baseline engine and return results."""
    with tracer.start_as_current_span("comparison.solo_baseline") as span:
        span.set_attribute("message", message[:200])
        baseline = _call_baseline(message)

    record_tokens(baseline["input_tokens"], baseline["output_tokens"], mode="baseline")
    record_cost("baseline", MODEL, baseline["input_tokens"], baseline["output_tokens"],
                baseline.get("cache_read_tokens", 0), baseline.get("cache_creation_tokens", 0))

    with _lock:
        _cumulative["baseline_input"] += baseline["input_tokens"]
        _cumulative["baseline_output"] += baseline["output_tokens"]
        _cumulative["turns"] += 1

        _turn_history.append({
            "turn": _cumulative["turns"],
            "optimized_input": 0,
            "baseline_input": baseline["input_tokens"],
            "cum_optimized": _cumulative["optimized_input"] + _cumulative["optimized_output"],
            "cum_baseline": _cumulative["baseline_input"] + _cumulative["baseline_output"],
        })

    return {"baseline": baseline}


def _handle_council(message: str) -> dict:
    """Run the 3-persona council and return results."""
    with tracer.start_as_current_span("comparison.council") as span:
        span.set_attribute("message", message[:200])

        client = _get_client()

        # Feed through summarizer for context compression (same as optimized path)
        _summarizer.add_message("user", message)
        context = _summarizer.get_context()
        # Build history from compressed context (excluding latest user msg which council will add)
        history = []
        for m in context:
            role = m["role"] if m["role"] != "system" else "user"
            history.append({"role": role, "content": m["content"]})
        # Merge consecutive same-role messages
        merged_history = []
        for msg in history:
            if merged_history and merged_history[-1]["role"] == msg["role"]:
                merged_history[-1]["content"] += "\n" + msg["content"]
            else:
                merged_history.append(dict(msg))
        # Remove last user message (the current one) since council will add it
        if merged_history and merged_history[-1]["role"] == "user":
            last_content = merged_history[-1]["content"]
            if last_content.endswith(message):
                if len(last_content) == len(message):
                    merged_history.pop()
                else:
                    merged_history[-1]["content"] = last_content[:-(len(message))].rstrip("\n")

        result = run_live_council(client, message, merged_history)

    # Add synthesizer's reply to summarizer history
    _summarizer.add_message("assistant", result["final_reply"])

    # Record cost per persona call
    for rnd in result["council_rounds"]:
        pricing = _MODEL_PRICING.get(
            "claude-haiku-4-5-20251001" if rnd["model"] == "haiku" else "claude-sonnet-4-20250514",
            _MODEL_PRICING["claude-sonnet-4-20250514"]
        )
        record_cost(
            "council", rnd["model"], rnd["input_tokens"], rnd["output_tokens"],
            cost_per_input=pricing["cost_per_input"],
            cost_per_output=pricing["cost_per_output"],
        )

    record_tokens(result["total_input_tokens"], result["total_output_tokens"], mode="optimized")

    # Track in cumulative under optimized keys so savings bar still works
    with _lock:
        _cumulative["optimized_input"] += result["total_input_tokens"]
        _cumulative["optimized_output"] += result["total_output_tokens"]
        _cumulative["turns"] += 1

        _turn_history.append({
            "turn": _cumulative["turns"],
            "optimized_input": result["total_input_tokens"],
            "baseline_input": 0,
            "cum_optimized": _cumulative["optimized_input"] + _cumulative["optimized_output"],
            "cum_baseline": _cumulative["baseline_input"] + _cumulative["baseline_output"],
        })

    # Build pipeline info
    summary_messages = [m for m in context if m["content"].startswith("[Summary L")]
    raw_messages = [m for m in context if not m["content"].startswith("[Summary L")]
    stats = _summarizer.get_stats()

    return {
        "council": result,
        "optimized": {
            "reply": result["final_reply"],
            "input_tokens": result["total_input_tokens"],
            "output_tokens": result["total_output_tokens"],
            "pipeline": {
                "summary_nodes": len(summary_messages),
                "raw_messages": len(raw_messages),
                "total_context_items": len(context),
                "summarizer_stats": stats,
                "summaries_preview": [m["content"][:120] for m in summary_messages],
            },
        },
    }


def _get_stats() -> dict:
    with _lock:
        c = dict(_cumulative)
    total_opt = c["optimized_input"] + c["optimized_output"]
    total_base = c["baseline_input"] + c["baseline_output"]
    total_saved = total_base - total_opt
    pct = (total_saved / total_base * 100) if total_base > 0 else 0
    cost_saved = total_saved * COST_PER_INPUT_TOKEN  # approximate

    # Include immutable cost tracker summary
    tracker = get_cost_tracker()
    cost_summary = tracker.summary()

    # Context budget: what's the effective context utilization?
    summarizer_stats = _summarizer.get_stats()
    context_window = 200_000  # Claude Sonnet context window
    system_prompt_tokens = 12  # ~12 tokens for "You are a helpful assistant. Be concise."
    active_context = summarizer_stats["active_tokens"] + system_prompt_tokens
    context_pct = round(active_context / context_window * 100, 2)

    return {
        **c,
        "total_optimized": total_opt,
        "total_baseline": total_base,
        "total_saved": total_saved,
        "savings_pct": round(pct, 1),
        "cost_saved": round(cost_saved, 6),
        "cost_tracker": cost_summary,
        "model": MODEL,
        "model_cheap": MODEL_CHEAP,
        "llm_summarizer": USE_LLM_SUMMARIZER,
        "context_budget": {
            "window_size": context_window,
            "system_prompt_tokens": system_prompt_tokens,
            "summary_tokens": summarizer_stats["active_tokens"] - sum(m.token_estimate for m in _summarizer.messages),
            "raw_msg_tokens": sum(m.token_estimate for m in _summarizer.messages),
            "active_tokens": active_context,
            "utilization_pct": context_pct,
            "headroom": context_window - active_context,
            "summarizer_keeping_up": summarizer_stats["active_tokens"] < context_window * 0.5,
        },
    }


def _get_rlm_state() -> dict:
    """Return the full internal state of the RecursiveSummarizer for visualization."""
    stats = _summarizer.get_stats()

    # Summary tree nodes with full detail
    nodes = []
    for s in _summarizer.summaries:
        nodes.append({
            "level": s.level,
            "text": s.text,
            "source_tokens": s.source_tokens,
            "compressed_tokens": s.compressed_tokens,
            "compression_ratio": round(s.compression_ratio * 100, 1),
            "children_hashes": s.children_hashes,
        })

    # Message buffer (recent raw messages not yet compressed)
    buffer = []
    for m in _summarizer.messages:
        buffer.append({
            "role": m.role,
            "preview": m.content[:100],
            "token_estimate": m.token_estimate,
        })

    buffer_tokens = sum(m.token_estimate for m in _summarizer.messages)
    summary_tokens = sum(s.compressed_tokens for s in _summarizer.summaries)
    original_tokens_in_summaries = sum(s.source_tokens for s in _summarizer.summaries)

    return {
        "stats": stats,
        "summary_nodes": nodes,
        "message_buffer": buffer,
        "buffer_tokens": buffer_tokens,
        "summary_tokens": summary_tokens,
        "original_tokens_in_summaries": original_tokens_in_summaries,
        "context_tokens_sent": buffer_tokens + summary_tokens,
        "chunk_size": _summarizer.chunk_size,
        "compression_trigger": _summarizer.chunk_size * 2,
        "turn_history": list(_turn_history),
    }


def _reset():
    global _summarizer, _baseline_history
    with _lock:
        _summarizer = RecursiveSummarizer(
            chunk_size=2, max_summary_tokens=200,
            llm_summarize_fn=_llm_summarize if USE_LLM_SUMMARIZER else None,
        )
        register_summarizer_metrics(_summarizer)
        _baseline_history.clear()
        _turn_history.clear()
        for k in _cumulative:
            _cumulative[k] = 0
    reset_cost_tracker()


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

def _build_html() -> str:
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AraFlow — Comparison UI</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'SF Mono', 'Fira Code', monospace;
  background: #0a0a0f;
  color: #e0e0e0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}}
.header {{
  padding: 12px 20px;
  background: #111118;
  border-bottom: 1px solid #1e1e2e;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.header h1 {{
  font-size: 16px;
  color: #00d4ff;
}}
.header-actions {{ display: flex; gap: 10px; }}
.header-actions button, .header-actions a {{
  padding: 6px 14px;
  border-radius: 6px;
  border: 1px solid #1e1e2e;
  background: #16161e;
  color: #e0e0e0;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
}}
.header-actions button:hover, .header-actions a:hover {{
  border-color: #00d4ff;
  color: #00d4ff;
}}
#demo-btn {{
  background: #00d4ff;
  color: #0a0a0f;
  font-weight: bold;
  border-color: #00d4ff;
}}
#demo-btn:hover {{ background: #00b8e0; border-color: #00b8e0; color: #0a0a0f; }}
#demo-btn:disabled {{ background: #1a3a4a; color: #00d4ff88; border-color: #1a3a4a; cursor: wait; }}
.error-banner {{
  background: #2a1515;
  border: 1px solid #ff4444;
  color: #ff6666;
  padding: 12px 20px;
  font-size: 13px;
  display: {'block' if not api_key_set else 'none'};
}}
.metrics-bar {{
  padding: 10px 20px;
  background: #111118;
  border-bottom: 1px solid #1e1e2e;
  display: flex;
  gap: 24px;
  font-size: 12px;
  flex-wrap: wrap;
}}
.metric {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.metric .label {{ color: #888; }}
.metric .value {{ font-weight: bold; }}
.metric .value.green {{ color: #00ff88; }}
.metric .value.cyan {{ color: #00d4ff; }}
.metric .value.yellow {{ color: #ffd700; }}
.metric .value.purple {{ color: #b388ff; }}
.panes {{
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: #1e1e2e;
  overflow: hidden;
  min-height: 0;
}}
.pane {{
  background: #0a0a0f;
  display: flex;
  flex-direction: column;
  min-height: 0;
}}
.pane-header {{
  padding: 10px 16px;
  background: #111118;
  border-bottom: 1px solid #1e1e2e;
  font-size: 13px;
  font-weight: bold;
  display: flex;
  justify-content: space-between;
}}
.pane-header .tag {{
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: normal;
}}
.pane-header .tag.opt {{ background: #002a1a; color: #00ff88; }}
.pane-header .tag.base {{ background: #2a1500; color: #ffd700; }}
.messages {{
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.msg {{
  max-width: 85%;
  padding: 10px 14px;
  border-radius: 10px;
  font-size: 13px;
  line-height: 1.5;
  position: relative;
}}
.msg.user {{
  align-self: flex-end;
  background: #1a1a2e;
  border: 1px solid #2a2a3e;
}}
.msg.assistant {{
  align-self: flex-start;
  background: #111118;
  border: 1px solid #1e1e2e;
}}
.msg .token-badge {{
  position: absolute;
  top: -8px;
  right: 8px;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  background: #1e1e2e;
  color: #888;
}}
.pane-input {{
  padding: 10px 12px;
  background: #111118;
  border-top: 1px solid #1e1e2e;
  display: flex;
  gap: 8px;
  align-items: flex-end;
}}
.pane-input textarea {{
  flex: 1;
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2a2a3e;
  background: #0a0a0f;
  color: #e0e0e0;
  font-family: inherit;
  font-size: 13px;
  outline: none;
  resize: none;
  min-height: 38px;
  max-height: 120px;
  overflow-y: auto;
  line-height: 1.4;
}}
.pane-input textarea:focus {{ border-color: #00d4ff; }}
.pane-input button {{
  padding: 8px 14px;
  border-radius: 8px;
  border: none;
  font-family: inherit;
  font-size: 13px;
  font-weight: bold;
  cursor: pointer;
  flex-shrink: 0;
  height: 38px;
}}
.pane-input .send-btn {{
  background: #00d4ff;
  color: #0a0a0f;
}}
.pane-input .send-btn:hover {{ background: #00b8e0; }}
.pane-input .send-btn:disabled {{ background: #333; color: #666; cursor: not-allowed; }}
.pane-input .pdf-btn {{
  background: #1e1e2e;
  color: #e0e0e0;
  border: 1px solid #2a2a3e;
  font-size: 16px;
  padding: 8px 10px;
}}
.pane-input .pdf-btn:hover {{ border-color: #00d4ff; color: #00d4ff; }}
.spinner {{
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid #333;
  border-top-color: #00d4ff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.savings-bar {{
  padding: 8px 20px;
  background: #0d1a0d;
  border-top: 1px solid #1a3a1a;
  font-size: 12px;
  color: #00ff88;
  text-align: center;
}}
.pipeline {{
  padding: 8px 16px;
  background: #0d0d14;
  border-bottom: 1px solid #1e1e2e;
  font-size: 11px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.pipeline-row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.pipeline-label {{ color: #666; min-width: 80px; }}
.pipeline-value {{ color: #aaa; }}
.pipeline-value.highlight {{ color: #00ff88; font-weight: bold; }}
.pipeline-value.warn {{ color: #ffd700; }}
.context-bar {{
  height: 6px;
  border-radius: 3px;
  background: #1e1e2e;
  flex: 1;
  overflow: hidden;
  display: flex;
}}
.context-bar .seg-summary {{
  background: #00d4ff;
  transition: width 0.3s ease;
}}
.context-bar .seg-raw {{
  background: #00ff88;
  transition: width 0.3s ease;
}}
.context-bar .seg-full {{
  background: #ff6644;
  transition: width 0.3s ease;
}}
.pipeline-legend {{
  display: flex;
  gap: 12px;
  font-size: 10px;
  color: #666;
}}
.pipeline-legend span::before {{
  content: '';
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}}
.pipeline-legend .leg-summary::before {{ background: #00d4ff; }}
.pipeline-legend .leg-raw::before {{ background: #00ff88; }}
.pipeline-legend .leg-full::before {{ background: #ff6644; }}
.compression-event {{
  font-size: 10px;
  color: #00d4ff;
  padding: 4px 12px;
  margin: 4px 16px;
  background: #0a1520;
  border: 1px dashed #00d4ff44;
  border-radius: 4px;
  text-align: center;
}}

/* --- RLM Visualizer --- */
.rlm-toggle {{
  padding: 6px 20px;
  background: #0d0d14;
  border-bottom: 1px solid #1e1e2e;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
}}
.rlm-toggle:hover {{ background: #111118; }}
.rlm-toggle .rlm-title {{
  font-size: 12px;
  font-weight: bold;
  color: #00d4ff;
}}
.rlm-toggle .rlm-arrow {{
  color: #00d4ff;
  font-size: 10px;
  transition: transform 0.2s;
}}
.rlm-toggle .rlm-arrow.open {{ transform: rotate(180deg); }}
.rlm-panel {{
  display: none;
  background: #080810;
  border-bottom: 1px solid #1e1e2e;
  padding: 16px 20px;
  overflow-x: auto;
}}
.rlm-panel.open {{ display: block; }}

/* Flow diagram */
.rlm-flow {{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 16px;
}}
.rlm-flow-arrow {{
  color: #00d4ff44;
  font-size: 20px;
  margin-top: 20px;
  flex-shrink: 0;
}}
.rlm-box {{
  border: 1px solid #1e1e2e;
  border-radius: 8px;
  padding: 10px 14px;
  min-width: 160px;
  flex-shrink: 0;
}}
.rlm-box-title {{
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 6px;
}}
.rlm-box-value {{
  font-size: 18px;
  font-weight: bold;
  margin-bottom: 2px;
}}
.rlm-box-sub {{
  font-size: 10px;
  color: #888;
}}
.rlm-box.buffer {{ border-color: #ffd70044; }}
.rlm-box.buffer .rlm-box-value {{ color: #ffd700; }}
.rlm-box.chunk {{ border-color: #ff664444; }}
.rlm-box.chunk .rlm-box-value {{ color: #ff6644; }}
.rlm-box.tree {{ border-color: #00d4ff44; }}
.rlm-box.tree .rlm-box-value {{ color: #00d4ff; }}
.rlm-box.sent {{ border-color: #00ff8844; }}
.rlm-box.sent .rlm-box-value {{ color: #00ff88; }}

/* Token funnel */
.rlm-funnel {{
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  padding: 10px 14px;
  background: #0a0a14;
  border-radius: 8px;
  border: 1px solid #1e1e2e;
}}
.rlm-funnel-stage {{
  text-align: center;
  flex: 1;
}}
.rlm-funnel-num {{
  font-size: 22px;
  font-weight: bold;
}}
.rlm-funnel-label {{
  font-size: 10px;
  color: #666;
  margin-top: 2px;
}}
.rlm-funnel-arrow {{
  color: #00d4ff;
  font-size: 16px;
  flex-shrink: 0;
}}

/* Summary tree */
.rlm-tree {{
  margin-bottom: 16px;
}}
.rlm-tree-title {{
  font-size: 11px;
  font-weight: bold;
  color: #b388ff;
  margin-bottom: 8px;
}}
.rlm-node {{
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 10px;
  border-left: 2px solid #00d4ff44;
  margin-bottom: 4px;
  margin-left: 0;
  font-size: 11px;
}}
.rlm-node.l0 {{ border-left-color: #00d4ff; margin-left: 0; }}
.rlm-node.l1 {{ border-left-color: #b388ff; margin-left: 16px; }}
.rlm-node.l2 {{ border-left-color: #ff6644; margin-left: 32px; }}
.rlm-node-level {{
  font-weight: bold;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  flex-shrink: 0;
}}
.rlm-node.l0 .rlm-node-level {{ background: #00d4ff22; color: #00d4ff; }}
.rlm-node.l1 .rlm-node-level {{ background: #b388ff22; color: #b388ff; }}
.rlm-node.l2 .rlm-node-level {{ background: #ff664422; color: #ff6644; }}
.rlm-node-text {{
  color: #aaa;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.rlm-node-ratio {{
  color: #00ff88;
  font-weight: bold;
  flex-shrink: 0;
  font-size: 10px;
}}

/* Divergence chart */
.rlm-chart {{
  margin-bottom: 8px;
}}
.rlm-chart-title {{
  font-size: 11px;
  font-weight: bold;
  color: #b388ff;
  margin-bottom: 8px;
}}
.rlm-chart-canvas {{
  width: 100%;
  height: 120px;
  background: #0a0a14;
  border-radius: 8px;
  border: 1px solid #1e1e2e;
}}

/* Empty state */
.rlm-empty {{
  text-align: center;
  color: #444;
  padding: 24px;
  font-size: 12px;
}}

/* Council toggle */
.council-btn {{
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid #2a2a3e;
  background: #1e1e2e;
  color: #888;
  font-family: inherit;
  font-size: 11px;
  font-weight: bold;
  cursor: pointer;
  flex-shrink: 0;
  height: 38px;
  transition: all 0.2s;
}}
.council-btn:hover {{ border-color: #00d4ff; color: #00d4ff; }}
.council-btn.active {{
  background: #003344;
  color: #00d4ff;
  border-color: #00d4ff;
}}

/* Council debate collapsible */
.council-debate {{
  margin-top: 10px;
  border: 1px solid #1e1e2e;
  border-radius: 8px;
  overflow: hidden;
}}
.council-debate-toggle {{
  padding: 6px 12px;
  background: #0d0d14;
  cursor: pointer;
  font-size: 11px;
  color: #00d4ff;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.council-debate-toggle:hover {{ background: #111118; }}
.council-debate-body {{
  display: none;
  padding: 0;
}}
.council-debate-body.open {{ display: block; }}
.council-persona {{
  padding: 10px 12px;
  border-top: 1px solid #1e1e2e;
  font-size: 12px;
  line-height: 1.5;
}}
.council-persona-header {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 11px;
  font-weight: bold;
}}
.council-persona-header .model-badge {{
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: normal;
}}
.model-badge.haiku {{ background: #1a2a1a; color: #88cc88; }}
.model-badge.sonnet {{ background: #1a1a2a; color: #8888ff; }}
.council-persona-text {{
  color: #ccc;
  white-space: pre-wrap;
}}
</style>
</head>
<body>

<div class="header">
  <h1>AraFlow Comparison — Optimized vs Baseline</h1>
  <div class="header-actions">
    <button id="demo-btn" onclick="runDemoScenario()">Run Demo Scenario</button>
    <button onclick="resetChat()">Reset</button>
    <a href="http://localhost:3000" target="_blank">Open Grafana</a>
  </div>
</div>

<div class="error-banner" id="error-banner">
  ANTHROPIC_API_KEY is not set. Export it before running: <code>export ANTHROPIC_API_KEY="sk-ant-..."</code>
</div>

<div class="metrics-bar" id="metrics-bar">
  <div class="metric"><span class="label">Turn:</span> <span class="value cyan" id="m-turn">0</span></div>
  <div class="metric"><span class="label">Optimized:</span> <span class="value green" id="m-opt">0</span></div>
  <div class="metric"><span class="label">Baseline:</span> <span class="value yellow" id="m-base">0</span></div>
  <div class="metric"><span class="label">Tokens saved:</span> <span class="value green" id="m-saved">0</span></div>
  <div class="metric"><span class="label">Savings:</span> <span class="value green" id="m-cum-pct">0%</span></div>
  <div class="metric"><span class="label">Cost saved:</span> <span class="value purple" id="m-cost">$0.000000</span></div>
  <div class="metric"><span class="label">Session cost:</span> <span class="value purple" id="m-session-cost">$0.00</span></div>
  <div class="metric"><span class="label">Budget:</span> <span class="value cyan" id="m-budget">$5.00</span></div>
  <div class="metric"><span class="label">Context:</span> <span class="value cyan" id="m-ctx-pct">0%</span></div>
  <div class="metric"><span class="label">Summarizer:</span> <span class="value green" id="m-summarizer">{'LLM (Haiku)' if USE_LLM_SUMMARIZER else 'Heuristic'}</span></div>
</div>

<div class="rlm-toggle" onclick="toggleRlm()">
  <span class="rlm-title">RLM Pipeline Visualizer — See how Recursive Summarization compresses your conversation</span>
  <span class="rlm-arrow" id="rlm-arrow">&#9660;</span>
</div>
<div class="rlm-panel" id="rlm-panel">
  <div id="rlm-content">
    <div class="rlm-empty">Send a message to see the RLM pipeline in action</div>
  </div>
</div>

<div class="panes">
  <div class="pane">
    <div class="pane-header">
      <span>AraFlow Optimized</span>
      <span class="tag opt">RLM Compression</span>
    </div>
    <div class="pipeline" id="opt-pipeline">
      <div class="pipeline-row">
        <span class="pipeline-label">Context:</span>
        <span class="pipeline-value" id="opt-ctx">Waiting for first message...</span>
      </div>
      <div class="pipeline-row">
        <span class="pipeline-label">Sent to API:</span>
        <div class="context-bar" id="opt-bar"></div>
      </div>
      <div class="pipeline-legend">
        <span class="leg-summary">Summaries</span>
        <span class="leg-raw">Recent msgs</span>
      </div>
    </div>
    <div class="messages" id="opt-messages"></div>
    <div class="pane-input">
      <button class="pdf-btn" onclick="triggerPdf('opt')" title="Upload PDF">&#128206;</button>
      <textarea id="opt-input" rows="1" placeholder="Type a message or upload PDF..."
                onkeydown="handlePaneKey(event, 'opt')"
                oninput="autoGrow(this)"></textarea>
      <button class="council-btn" id="council-toggle" onclick="toggleCouncil()" title="Toggle Council mode (3-model debate)">Council</button>
      <button class="send-btn" id="opt-send" onclick="sendPaneMessage('opt')">Send</button>
      <input type="file" id="opt-file" accept=".pdf" style="display:none" onchange="handlePdf(event, 'opt')">
    </div>
  </div>
  <div class="pane">
    <div class="pane-header">
      <span>Baseline (Full History)</span>
      <span class="tag base">No Compression</span>
    </div>
    <div class="pipeline" id="base-pipeline">
      <div class="pipeline-row">
        <span class="pipeline-label">Context:</span>
        <span class="pipeline-value" id="base-ctx">Waiting for first message...</span>
      </div>
      <div class="pipeline-row">
        <span class="pipeline-label">Sent to API:</span>
        <div class="context-bar" id="base-bar"></div>
      </div>
      <div class="pipeline-legend">
        <span class="leg-full">All messages (uncompressed)</span>
      </div>
    </div>
    <div class="messages" id="base-messages"></div>
    <div class="pane-input">
      <button class="pdf-btn" onclick="triggerPdf('base')" title="Upload PDF">&#128206;</button>
      <textarea id="base-input" rows="1" placeholder="Type a message or upload PDF..."
                onkeydown="handlePaneKey(event, 'base')"
                oninput="autoGrow(this)"></textarea>
      <button class="send-btn" id="base-send" onclick="sendPaneMessage('base')">Send</button>
      <input type="file" id="base-file" accept=".pdf" style="display:none" onchange="handlePdf(event, 'base')">
    </div>
  </div>
</div>

<div class="savings-bar" id="savings-bar">Send a message to compare token usage</div>

<script>
const optMessages = document.getElementById('opt-messages');
const baseMessages = document.getElementById('base-messages');

// --- Textarea auto-grow ---
function autoGrow(el) {{
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}}

function handlePaneKey(e, pane) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendPaneMessage(pane);
  }}
}}

// Sonnet pricing for cost display
const SONNET_INPUT = 3.0 / 1_000_000;
const SONNET_OUTPUT = 15.0 / 1_000_000;

function calcCost(inputTok, outputTok) {{
  return (inputTok * SONNET_INPUT + outputTok * SONNET_OUTPUT);
}}

function addMsg(container, role, text, tokens, cost) {{
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (tokens !== undefined) {{
    const badge = document.createElement('span');
    badge.className = 'token-badge';
    let label = tokens + ' tok';
    if (cost !== undefined) {{
      label += ' · $' + cost.toFixed(4);
    }}
    badge.textContent = label;
    div.appendChild(badge);
  }}
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}}

function addSpinner(container) {{
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.id = container.id + '-spinner';
  div.innerHTML = '<span class="spinner"></span> Thinking...';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}}

function removeSpinner(container) {{
  const el = document.getElementById(container.id + '-spinner');
  if (el) el.remove();
}}

// --- Council toggle ---
let councilMode = false;

function toggleCouncil() {{
  councilMode = !councilMode;
  const btn = document.getElementById('council-toggle');
  const textarea = document.getElementById('opt-input');
  if (councilMode) {{
    btn.classList.add('active');
    btn.textContent = 'Council ON';
    textarea.placeholder = 'Council mode — 3 models will debate your prompt...';
  }} else {{
    btn.classList.remove('active');
    btn.textContent = 'Council';
    textarea.placeholder = 'Type a message or upload PDF...';
  }}
}}

function addCouncilMsg(container, data) {{
  const council = data.council;
  const tok = council.total_input_tokens + council.total_output_tokens;

  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.style.maxWidth = '95%';

  // Main reply (synthesizer output)
  const replyText = document.createElement('div');
  replyText.textContent = council.final_reply;
  div.appendChild(replyText);

  // Token badge
  const badge = document.createElement('span');
  badge.className = 'token-badge';
  badge.textContent = tok + ' tok';
  div.appendChild(badge);

  // Collapsible council debate
  const debate = document.createElement('div');
  debate.className = 'council-debate';

  const roundSummary = council.council_rounds.map(r => r.model === 'haiku' ? 'Haiku' : 'Sonnet');
  const toggleBar = document.createElement('div');
  toggleBar.className = 'council-debate-toggle';
  toggleBar.innerHTML = `<span>Council Debate (` + roundSummary.join(' + ') + `) — $` + council.total_cost.toFixed(6) + `</span><span id="council-arrow">&#9660;</span>`;

  const body = document.createElement('div');
  body.className = 'council-debate-body';

  const emojiMap = {{ pragmatist: '🔧', critic: '🔍', synthesizer: '🧬' }};

  for (const rnd of council.council_rounds) {{
    const persona = document.createElement('div');
    persona.className = 'council-persona';

    const header = document.createElement('div');
    header.className = 'council-persona-header';
    const emoji = emojiMap[rnd.persona] || '';
    const totalTok = rnd.input_tokens + rnd.output_tokens;
    header.innerHTML = `${{emoji}} ${{rnd.persona.charAt(0).toUpperCase() + rnd.persona.slice(1)}} <span class="model-badge ${{rnd.model}}">${{rnd.model.charAt(0).toUpperCase() + rnd.model.slice(1)}}</span> <span style="color:#666;font-weight:normal">${{totalTok}} tok, $${{rnd.cost.toFixed(4)}}</span>`;

    const text = document.createElement('div');
    text.className = 'council-persona-text';
    text.textContent = rnd.reply;

    persona.appendChild(header);
    persona.appendChild(text);
    body.appendChild(persona);
  }}

  toggleBar.onclick = () => {{
    body.classList.toggle('open');
  }};

  debate.appendChild(toggleBar);
  debate.appendChild(body);
  div.appendChild(debate);

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}}

// --- PDF upload via pdf.js CDN ---
function triggerPdf(pane) {{
  document.getElementById(pane + '-file').click();
}}

async function handlePdf(event, pane) {{
  const file = event.target.files[0];
  if (!file) return;
  const textarea = document.getElementById(pane + '-input');
  textarea.value = 'Extracting PDF text...';
  textarea.disabled = true;
  try {{
    const pdfjsLib = await import('https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs');
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs';
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({{ data: arrayBuffer }}).promise;
    let text = '';
    for (let i = 1; i <= pdf.numPages; i++) {{
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(item => item.str).join(' ') + '\\n';
    }}
    textarea.value = text.trim();
    textarea.disabled = false;
    autoGrow(textarea);
    textarea.focus();
  }} catch (err) {{
    textarea.value = 'PDF extraction failed: ' + err.message;
    textarea.disabled = false;
  }}
  event.target.value = '';
}}

// --- Per-pane independent send ---
async function sendPaneMessage(pane) {{
  const textarea = document.getElementById(pane + '-input');
  const sendBtn = document.getElementById(pane + '-send');
  const msgContainer = pane === 'opt' ? optMessages : baseMessages;
  const useCouncil = pane === 'opt' && councilMode;
  const endpoint = useCouncil ? '/api/chat/council' : (pane === 'opt' ? '/api/chat/optimized' : '/api/chat/baseline');
  const dataKey = pane === 'opt' ? 'optimized' : 'baseline';

  const msg = textarea.value.trim();
  if (!msg) return;

  textarea.value = '';
  textarea.style.height = 'auto';
  sendBtn.disabled = true;

  addMsg(msgContainer, 'user', msg);
  addSpinner(msgContainer);

  try {{
    const resp = await fetch(endpoint, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{message: msg}})
    }});
    const data = await resp.json();

    removeSpinner(msgContainer);

    if (data.error) {{
      addMsg(msgContainer, 'assistant', 'Error: ' + data.error);
      return;
    }}

    if (useCouncil && data.council) {{
      // Show council debate UI
      addCouncilMsg(msgContainer, data);
    }} else {{
      const result = data[dataKey];
      const tok = result.input_tokens + result.output_tokens;
      const cost = calcCost(result.input_tokens, result.output_tokens);
      addMsg(msgContainer, 'assistant', result.reply, tok, cost);
    }}

    // Update pipeline viz for this pane
    const optResult = data[dataKey] || data.optimized;
    if (pane === 'opt' && optResult && optResult.pipeline) {{
      updateOptPipeline(optResult.pipeline);
    }} else if (pane === 'base') {{
      updateBasePipeline(data.baseline.pipeline);
    }}

    // Update metrics from cumulative stats
    await refreshMetrics();

    // Refresh RLM visualizer if open and this was the optimized pane
    if (pane === 'opt' && document.getElementById('rlm-panel').classList.contains('open')) {{
      refreshRlm();
    }}

  }} catch (err) {{
    removeSpinner(msgContainer);
    addMsg(msgContainer, 'assistant', 'Request failed: ' + err.message);
  }} finally {{
    sendBtn.disabled = false;
    textarea.focus();
  }}
}}

function updateOptPipeline(optP) {{
  const ss = optP.summarizer_stats;
  let optDesc = '';
  if (optP.summary_nodes > 0) {{
    optDesc = `${{optP.summary_nodes}} summary node${{optP.summary_nodes > 1 ? 's' : ''}} + ${{optP.raw_messages}} recent msgs`;
    optDesc += ` | ${{ss.total_tokens_saved}} tokens compressed away`;
  }} else {{
    optDesc = `${{optP.raw_messages}} messages (no compression yet — triggers at ${{optP.raw_messages}}/12)`;
  }}
  document.getElementById('opt-ctx').textContent = optDesc;
  document.getElementById('opt-ctx').className = 'pipeline-value' + (optP.summary_nodes > 0 ? ' highlight' : '');

  // Show compression event
  if (optP.summary_nodes > 0 && optP.summaries_preview.length > 0) {{
    const existingEvents = optMessages.querySelectorAll('.compression-event');
    const lastCount = existingEvents.length > 0 ? parseInt(existingEvents[existingEvents.length - 1].dataset.nodes || '0') : 0;
    if (optP.summary_nodes > lastCount) {{
      const evt = document.createElement('div');
      evt.className = 'compression-event';
      evt.dataset.nodes = optP.summary_nodes;
      evt.innerHTML = `AraFlow compressed ${{ss.total_tokens_saved}} tokens into ${{optP.summary_nodes}} summary node${{optP.summary_nodes > 1 ? 's' : ''}}`;
      optMessages.appendChild(evt);
      optMessages.scrollTop = optMessages.scrollHeight;
    }}
  }}
}}

function updateBasePipeline(baseP) {{
  document.getElementById('base-ctx').textContent = `${{baseP.total_messages}} messages — ALL sent to Claude every turn`;
  document.getElementById('base-ctx').className = 'pipeline-value warn';
}}

async function refreshMetrics() {{
  const stats = await fetch('/api/stats').then(r => r.json());
  document.getElementById('m-turn').textContent = stats.turns;
  document.getElementById('m-opt').textContent = stats.total_optimized;
  document.getElementById('m-base').textContent = stats.total_baseline;
  document.getElementById('m-saved').textContent = stats.total_saved;
  document.getElementById('m-cum-pct').textContent = stats.savings_pct + '%';
  document.getElementById('m-cost').textContent = '$' + stats.cost_saved.toFixed(6);

  // Cost tracker
  if (stats.cost_tracker) {{
    document.getElementById('m-session-cost').textContent = '$' + stats.cost_tracker.total_cost.toFixed(4);
    const budgetEl = document.getElementById('m-budget');
    budgetEl.textContent = '$' + stats.cost_tracker.budget_remaining.toFixed(2);
    if (stats.cost_tracker.budget_pct_used > 80) {{
      budgetEl.className = 'value yellow';
    }} else if (stats.cost_tracker.budget_pct_used > 95) {{
      budgetEl.className = 'value';
      budgetEl.style.color = '#ff4444';
    }} else {{
      budgetEl.className = 'value cyan';
    }}
  }}

  // Context budget
  if (stats.context_budget) {{
    const ctxEl = document.getElementById('m-ctx-pct');
    ctxEl.textContent = stats.context_budget.utilization_pct + '%';
    if (stats.context_budget.utilization_pct > 50) {{
      ctxEl.className = 'value yellow';
    }} else if (stats.context_budget.utilization_pct > 80) {{
      ctxEl.className = 'value';
      ctxEl.style.color = '#ff4444';
    }} else {{
      ctxEl.className = 'value cyan';
    }}
  }}

  const bar = document.getElementById('savings-bar');
  if (stats.total_saved > 0) {{
    bar.textContent = `Cumulative: ${{stats.total_saved}} tokens saved (${{stats.savings_pct}}%) | Context: ${{stats.context_budget ? stats.context_budget.active_tokens : '?'}}/${{stats.context_budget ? stats.context_budget.window_size.toLocaleString() : '?'}} tokens used`;
    bar.style.background = '#0d1a0d';
    bar.style.color = '#00ff88';
  }} else if (stats.turns > 0) {{
    bar.textContent = `Savings will appear as conversations grow.`;
    bar.style.background = '#1a1a0d';
    bar.style.color = '#ffd700';
  }}
}}

// --- Shared send (used by demo scenario) ---
async function sendSharedMessage(msg) {{
  addMsg(optMessages, 'user', msg);
  addMsg(baseMessages, 'user', msg);
  addSpinner(optMessages);
  addSpinner(baseMessages);

  const resp = await fetch('/api/chat', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{message: msg}})
  }});
  const data = await resp.json();

  removeSpinner(optMessages);
  removeSpinner(baseMessages);

  if (data.error) {{
    addMsg(optMessages, 'assistant', 'Error: ' + data.error);
    addMsg(baseMessages, 'assistant', 'Error: ' + data.error);
    return data;
  }}

  const optTok = data.optimized.input_tokens + data.optimized.output_tokens;
  const baseTok = data.baseline.input_tokens + data.baseline.output_tokens;
  const optCost = calcCost(data.optimized.input_tokens, data.optimized.output_tokens);
  const baseCost = calcCost(data.baseline.input_tokens, data.baseline.output_tokens);

  addMsg(optMessages, 'assistant', data.optimized.reply, optTok, optCost);
  addMsg(baseMessages, 'assistant', data.baseline.reply, baseTok, baseCost);

  // Update pipelines
  updateOptPipeline(data.optimized.pipeline);
  updateBasePipeline(data.baseline.pipeline);

  // Context bars
  const optP = data.optimized.pipeline;
  const baseP = data.baseline.pipeline;
  const totalCtx = Math.max(optP.total_context_items, baseP.total_messages, 1);
  document.getElementById('opt-bar').innerHTML = `<div class="seg-summary" style="width:${{optP.summary_nodes / totalCtx * 100}}%"></div><div class="seg-raw" style="width:${{optP.raw_messages / totalCtx * 100}}%"></div>`;
  document.getElementById('base-bar').innerHTML = `<div class="seg-full" style="width:${{baseP.total_messages / totalCtx * 100}}%"></div>`;

  return data;
}}

async function resetChat() {{
  await fetch('/api/reset', {{method: 'POST'}});
  optMessages.innerHTML = '';
  baseMessages.innerHTML = '';
  document.getElementById('m-turn').textContent = '0';
  document.getElementById('m-opt').textContent = '0';
  document.getElementById('m-base').textContent = '0';
  document.getElementById('m-saved').textContent = '0';
  document.getElementById('m-cum-pct').textContent = '0%';
  document.getElementById('m-cost').textContent = '$0.000000';
  document.getElementById('m-session-cost').textContent = '$0.00';
  document.getElementById('m-budget').textContent = '$5.00';
  document.getElementById('m-budget').className = 'value cyan';
  document.getElementById('m-ctx-pct').textContent = '0%';
  document.getElementById('m-ctx-pct').className = 'value cyan';
  document.getElementById('savings-bar').textContent = 'Send a message to compare token usage';
  document.getElementById('opt-ctx').textContent = 'Waiting for first message...';
  document.getElementById('opt-ctx').className = 'pipeline-value';
  document.getElementById('base-ctx').textContent = 'Waiting for first message...';
  document.getElementById('base-ctx').className = 'pipeline-value';
  document.getElementById('opt-bar').innerHTML = '';
  document.getElementById('base-bar').innerHTML = '';
  document.getElementById('opt-input').value = '';
  document.getElementById('base-input').value = '';
  document.getElementById('demo-btn').disabled = false;
  document.getElementById('demo-btn').textContent = 'Run Demo Scenario';
  document.getElementById('rlm-content').innerHTML = '<div class="rlm-empty">Send a message to see the RLM pipeline in action</div>';
}}

// ---------------------------------------------------------------------------
// RLM Pipeline Visualizer
// ---------------------------------------------------------------------------
function toggleRlm() {{
  const panel = document.getElementById('rlm-panel');
  const arrow = document.getElementById('rlm-arrow');
  panel.classList.toggle('open');
  arrow.classList.toggle('open');
  if (panel.classList.contains('open')) {{ refreshRlm(); }}
}}

async function refreshRlm() {{
  try {{
    const state = await fetch('/api/rlm-state').then(r => r.json());
    renderRlm(state);
  }} catch(e) {{}}
}}

function renderRlm(state) {{
  const el = document.getElementById('rlm-content');
  const s = state.stats;

  if (s.raw_messages === 0 && s.summary_nodes === 0) {{
    el.innerHTML = '<div class="rlm-empty">Send a message to see the RLM pipeline in action</div>';
    return;
  }}

  let html = '';

  // --- 1. Flow diagram ---
  html += '<div class="rlm-flow">';

  // Message buffer box
  html += `<div class="rlm-box buffer">
    <div class="rlm-box-title">Message Buffer</div>
    <div class="rlm-box-value">${{state.message_buffer.length}} msgs</div>
    <div class="rlm-box-sub">~${{state.buffer_tokens}} tokens</div>
    <div class="rlm-box-sub">Trigger at ${{state.compression_trigger}} msgs</div>
  </div>`;

  html += '<div class="rlm-flow-arrow">&#10132;</div>';

  // Chunking stage
  html += `<div class="rlm-box chunk">
    <div class="rlm-box-title">Chunking</div>
    <div class="rlm-box-value">${{state.chunk_size}} per chunk</div>
    <div class="rlm-box-sub">${{state.summary_nodes.length > 0 ? state.summary_nodes.length + ' chunks compressed' : 'No chunks yet'}}</div>
    <div class="rlm-box-sub">${{state.original_tokens_in_summaries}} tokens ingested</div>
  </div>`;

  html += '<div class="rlm-flow-arrow">&#10132;</div>';

  // Summary tree
  html += `<div class="rlm-box tree">
    <div class="rlm-box-title">Summary Tree</div>
    <div class="rlm-box-value">${{state.summary_nodes.length}} nodes</div>
    <div class="rlm-box-sub">${{state.summary_tokens}} tokens total</div>
    <div class="rlm-box-sub">Levels: ${{[...new Set(state.summary_nodes.map(n => n.level))].sort().map(l => 'L' + l).join(', ') || 'none'}}</div>
  </div>`;

  html += '<div class="rlm-flow-arrow">&#10132;</div>';

  // What gets sent
  html += `<div class="rlm-box sent">
    <div class="rlm-box-title">Sent to Claude</div>
    <div class="rlm-box-value">${{state.context_tokens_sent}} tok</div>
    <div class="rlm-box-sub">${{state.summary_nodes.length}} summaries + ${{state.message_buffer.length}} raw</div>
    <div class="rlm-box-sub">vs ${{state.turn_history.length > 0 ? state.turn_history[state.turn_history.length-1].cum_baseline : '?'}} baseline</div>
  </div>`;

  html += '</div>';

  // --- 2. Token funnel ---
  if (state.original_tokens_in_summaries > 0) {{
    const saved = s.total_tokens_saved;
    const savePct = state.original_tokens_in_summaries > 0
      ? Math.round(saved / state.original_tokens_in_summaries * 100) : 0;
    html += `<div class="rlm-funnel">
      <div class="rlm-funnel-stage">
        <div class="rlm-funnel-num" style="color:#ffd700">${{state.original_tokens_in_summaries}}</div>
        <div class="rlm-funnel-label">Original tokens<br>(in compressed chunks)</div>
      </div>
      <div class="rlm-funnel-arrow">&#10132; RLM &#10132;</div>
      <div class="rlm-funnel-stage">
        <div class="rlm-funnel-num" style="color:#00d4ff">${{state.summary_tokens}}</div>
        <div class="rlm-funnel-label">Summary tokens<br>(after compression)</div>
      </div>
      <div class="rlm-funnel-arrow">= </div>
      <div class="rlm-funnel-stage">
        <div class="rlm-funnel-num" style="color:#00ff88">${{saved}} saved</div>
        <div class="rlm-funnel-label">${{savePct}}% reduction<br>by recursive summarization</div>
      </div>
    </div>`;
  }}

  // --- 3. Summary node tree ---
  if (state.summary_nodes.length > 0) {{
    html += '<div class="rlm-tree">';
    html += '<div class="rlm-tree-title">Summary Node Tree (what replaces old messages)</div>';

    // Sort by level descending so higher-level summaries appear first
    const sorted = [...state.summary_nodes].sort((a, b) => b.level - a.level);
    for (const node of sorted) {{
      const lClass = node.level <= 2 ? 'l' + node.level : 'l2';
      html += `<div class="rlm-node ${{lClass}}">
        <span class="rlm-node-level">L${{node.level}}</span>
        <span class="rlm-node-text" title="${{node.text.replace(/"/g, '&quot;')}}">${{node.text.substring(0, 150)}}</span>
        <span class="rlm-node-ratio">${{node.source_tokens}} &#10132; ${{node.compressed_tokens}} tok (${{node.compression_ratio}}% saved)</span>
      </div>`;
    }}
    html += '</div>';
  }}

  // --- 4. Per-turn divergence chart (canvas) ---
  if (state.turn_history.length >= 2) {{
    html += '<div class="rlm-chart">';
    html += '<div class="rlm-chart-title">Per-Turn Input Tokens: Optimized (green) vs Baseline (red)</div>';
    html += '<canvas class="rlm-chart-canvas" id="rlm-canvas"></canvas>';
    html += '</div>';
  }}

  // --- 5. Message buffer preview ---
  if (state.message_buffer.length > 0) {{
    html += '<div class="rlm-tree">';
    html += `<div class="rlm-tree-title">Message Buffer (${{state.message_buffer.length}} messages, ~${{state.buffer_tokens}} tokens — kept as raw)</div>`;
    for (const m of state.message_buffer.slice(-6)) {{
      html += `<div class="rlm-node l0">
        <span class="rlm-node-level">${{m.role}}</span>
        <span class="rlm-node-text">${{m.preview}}</span>
        <span class="rlm-node-ratio">~${{m.token_estimate}} tok</span>
      </div>`;
    }}
    html += '</div>';
  }}

  el.innerHTML = html;

  // Draw the chart if we have data
  if (state.turn_history.length >= 2) {{
    drawDivergenceChart(state.turn_history);
  }}
}}

function drawDivergenceChart(history) {{
  const canvas = document.getElementById('rlm-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Set actual pixel dimensions
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  ctx.scale(2, 2);  // retina
  const W = rect.width;
  const H = rect.height;

  // Data
  const optVals = history.map(h => h.optimized_input);
  const baseVals = history.map(h => h.baseline_input);
  const maxVal = Math.max(...baseVals, ...optVals, 1);
  const n = history.length;

  // Padding
  const padL = 50, padR = 16, padT = 12, padB = 24;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  // Background
  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1e1e2e';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = padT + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    // Label
    ctx.fillStyle = '#444';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(maxVal * (1 - i / 4)), padL - 4, y + 3);
  }}

  // Turn labels
  ctx.fillStyle = '#444';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {{
    const x = padL + (chartW / Math.max(n - 1, 1)) * i;
    ctx.fillText('T' + (i + 1), x, H - 4);
  }}

  function drawLine(vals, color) {{
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {{
      const x = padL + (chartW / Math.max(n - 1, 1)) * i;
      const y = padT + chartH - (vals[i] / maxVal) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Dots
    for (let i = 0; i < n; i++) {{
      const x = padL + (chartW / Math.max(n - 1, 1)) * i;
      const y = padT + chartH - (vals[i] / maxVal) * chartH;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}
  }}

  // Fill the area between the two lines (savings area)
  if (n >= 2) {{
    ctx.beginPath();
    ctx.fillStyle = '#00ff8815';
    for (let i = 0; i < n; i++) {{
      const x = padL + (chartW / Math.max(n - 1, 1)) * i;
      const y = padT + chartH - (baseVals[i] / maxVal) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    for (let i = n - 1; i >= 0; i--) {{
      const x = padL + (chartW / Math.max(n - 1, 1)) * i;
      const y = padT + chartH - (optVals[i] / maxVal) * chartH;
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.fill();
  }}

  drawLine(baseVals, '#ff6644');
  drawLine(optVals, '#00ff88');

  // Legend
  ctx.font = '10px monospace';
  ctx.fillStyle = '#ff6644';
  ctx.textAlign = 'left';
  ctx.fillText('Baseline', padL + 4, padT + 12);
  ctx.fillStyle = '#00ff88';
  ctx.fillText('Optimized', padL + 74, padT + 12);
  ctx.fillStyle = '#00ff8866';
  ctx.fillText('= Savings', padL + 154, padT + 12);
}}

// ---------------------------------------------------------------------------
// Demo scenario: realistic enterprise support conversation
// Simulates a SaaS customer support agent handling an escalation.
// These are long, context-heavy messages — exactly the type that
// costs companies thousands of dollars/month at scale.
// ---------------------------------------------------------------------------
const DEMO_PROMPTS = [
  // Turn 1 — customer opens a detailed ticket
  `I'm the lead DevOps engineer at Meridian Financial (enterprise plan, account #MF-20948). We're experiencing a critical production incident. Our Kubernetes cluster running 47 microservices started throwing cascading 503 errors at 2:47 AM UTC. The error rate spiked from 0.1% to 34% in under 90 seconds. Our payment processing pipeline (services: payment-gateway, fraud-detector, ledger-sync, and settlement-engine) is completely down. We've already tried rolling back the last deployment (commit abc123f from yesterday), restarting the ingress controller, and scaling the payment-gateway from 3 to 12 replicas. None of these helped. Our P0 SLA requires resolution within 4 hours and we're at hour 2. Can you help us diagnose this?`,

  // Turn 2 — provide more diagnostic context
  `Here's what our monitoring shows. Datadog APM traces reveal that payment-gateway is waiting on fraud-detector, which in turn is blocked on a gRPC call to our ML scoring service. The ML scoring service logs show: "RESOURCE_EXHAUSTED: memory limit exceeded, current usage 14.2GB / 16GB limit". Prometheus shows the ML model was updated via our CI/CD pipeline at 2:31 AM UTC — 16 minutes before the incident. The new model file is 4.7GB (previous was 1.2GB). GPU utilization went from 45% to 99% immediately after deployment. The health check endpoint returns 200 but the actual inference latency jumped from 23ms to 8400ms p99.`,

  // Turn 3 — ask about architecture impact
  `Our architecture is: API Gateway (Kong) -> payment-gateway (Go, 12 pods) -> fraud-detector (Python, 8 pods) -> ML scoring (Python/PyTorch, 4 pods on GPU nodes, g4dn.xlarge). The circuit breaker between fraud-detector and ML scoring is configured with a 2-second timeout, 5 retries, and 60% failure threshold. What concerns me is that even after the circuit breaker should have tripped, we're still seeing requests queue up. Could the combination of the long timeout (2s x 5 retries = 10s total) and the 60% threshold be masking the actual failure? Also, what's the recommended approach for ML model rollback in a live payment system?`,

  // Turn 4 — discuss the business impact
  `Let me add the business context. We process roughly $2.3M in transactions per hour during peak. We're currently losing approximately $38,000 per minute in failed transactions. Our compliance team is asking whether we need to file an incident report with our banking partners under PCI DSS requirement 12.10. We also have 3 enterprise clients (each >$500K ARR) whose webhook integrations are failing, and their SRE teams are paging us. Can you help me draft the internal incident communication and suggest what we should tell these clients? I need something that's honest but doesn't create legal liability.`,

  // Turn 5 — get into remediation specifics
  `Okay, we've implemented your suggestion and rolled back the ML model. Inference latency is back to 28ms p99. But now we have a new problem: during the 2-hour outage, approximately 12,400 payment transactions were stuck in a "pending" state in our PostgreSQL database. Our settlement-engine has a reconciliation job that runs every 15 minutes, but it's designed for maybe 50-100 pending transactions, not 12,400. If we let it run normally it'll take roughly 31 hours to clear the backlog. Can you help me write a one-time batch reconciliation script that can safely process these in parallel while respecting our rate limits with the banking API (200 req/s)?`,

  // Turn 6 — pivot to post-mortem planning
  `Good, the batch reconciliation is running. Let me shift to the post-mortem. Our CTO wants a full root cause analysis by Friday. I need to cover: 1) Why the ML model deployment wasn't caught by our canary process, 2) Why the circuit breaker didn't protect downstream services, 3) Why our alerting had a 14-minute gap before PagerDuty fired, 4) The total financial impact including lost transactions and engineering time. Can you help me structure this post-mortem document? We use the Google SRE format. Also suggest 3-5 concrete action items that will prevent this class of failure.`,

  // Turn 7 — discuss infrastructure changes
  `For action item #2 (the circuit breaker tuning), can you explain the tradeoffs between different circuit breaker patterns for our specific case? We're considering: (a) switching from Hystrix-style to a token bucket pattern, (b) implementing a separate fast-fail path for payment-critical services, (c) adding a model-size gate in our CI/CD pipeline that blocks deployments if the artifact is >2x the current production size. Also, our VP of Engineering asked about implementing a "shadow traffic" system where we can test new ML models against production traffic without affecting real transactions. What would that architecture look like?`,

  // Turn 8 — compliance and regulatory
  `Our compliance officer just confirmed we DO need to file with our banking partners. Under PCI DSS 12.10.5 we have 24 hours. Can you help me draft the incident notification? It needs to include: the timeline of events, services affected, data exposure assessment (we believe no cardholder data was exposed, only transaction processing was disrupted), remediation steps taken, and preventive measures. It also needs to reference our last penetration test (completed March 15) and our SOC 2 Type II audit (completed January 8). The notification goes to Chase Paymentech, Stripe Connect, and our acquiring bank Wells Fargo Merchant Services.`,

  // Turn 9 — capacity planning
  `Based on this incident, we're re-evaluating our capacity planning. Currently we're spending $47,000/month on our GKE cluster (32 nodes: 20x n2-standard-8, 8x n2-highmem-16, 4x g4dn.xlarge GPU). Our traffic has been growing 15% month-over-month. The ML team wants to deploy 3 more models in Q3 (each 2-3GB). Finance is pushing back on infra costs. Can you help me model out: what our projected costs look like for Q3-Q4, where we can optimize (spot instances, autoscaling policies, model quantization), and what headroom we need to maintain to prevent another incident like today's?`,

  // Turn 10 — wrap up and summarize
  `This has been incredibly helpful. Can you give me a complete summary of everything we've discussed and decided? I need to present this to the leadership team in 30 minutes. Include: the root cause, timeline, business impact ($), immediate remediation steps we took, the 5 action items for the post-mortem, the compliance filing status, and the capacity planning recommendations. Format it as an executive briefing — bullet points, no jargon, focus on business impact and risk mitigation.`,

  // Turn 11-15: follow-up conversation that builds more context
  `One more thing — the leadership team wants to know how this compares to industry benchmarks. What's the typical MTTR for a P0 incident at fintech companies our size (200-500 engineers, $50M+ ARR)? And what's considered an acceptable error budget for payment processing systems? Our current SLO is 99.95% but the CFO thinks we should target 99.99%.`,

  `The CTO also asked about chaos engineering. We've never done it. Given what just happened, should we implement something like Chaos Monkey or Litmus? What's the minimum viable chaos engineering practice we could start with that would have caught today's failure mode? We need something that won't terrify our compliance team.`,

  `Our ML team lead just told me the new model was actually supposed to go through a staged rollout (10% canary -> 50% -> 100% over 48 hours) but someone manually overrode the pipeline and pushed it to 100% immediately. We need to lock down our deployment pipeline. What GitOps practices would you recommend to prevent manual overrides of safety gates? We use ArgoCD and GitHub Actions.`,

  `Final question — we're evaluating whether to build an internal incident management platform or buy one. We're looking at PagerDuty (already have it for alerting), FireHydrant, incident.io, and Rootly. Our requirements: automatic Slack channel creation, stakeholder notifications, timeline tracking, post-mortem templates, and integration with Jira and Datadog. We have budget for about $3,000/month. What would you recommend and why?`,

  `Perfect, thank you for all of this. Can you now compile everything from our entire conversation into a structured incident report document? Include all sections: executive summary, detailed timeline, root cause analysis, impact assessment, remediation actions, action items with owners and deadlines, compliance status, capacity planning recommendations, tooling recommendations, and lessons learned. This will be our official incident record.`
];

let demoRunning = false;

async function runDemoScenario() {{
  if (demoRunning) return;
  demoRunning = true;

  const btn = document.getElementById('demo-btn');
  btn.disabled = true;
  document.getElementById('opt-send').disabled = true;
  document.getElementById('base-send').disabled = true;
  document.getElementById('opt-input').disabled = true;
  document.getElementById('base-input').disabled = true;

  // Reset first
  await fetch('/api/reset', {{method: 'POST'}});
  optMessages.innerHTML = '';
  baseMessages.innerHTML = '';

  for (let i = 0; i < DEMO_PROMPTS.length; i++) {{
    btn.textContent = `Running ${{i + 1}}/${{DEMO_PROMPTS.length}}...`;

    try {{
      const data = await sendSharedMessage(DEMO_PROMPTS[i]);

      if (data.error) break;

      await refreshMetrics();

      const stats = await fetch('/api/stats').then(r => r.json());
      const savBar = document.getElementById('savings-bar');
      if (data.tokens_saved > 0) {{
        savBar.textContent = `Turn ${{i+1}}: ${{data.tokens_saved}} tokens saved (${{data.savings_pct}}%) — Cumulative: ${{stats.total_saved}} tokens (${{stats.savings_pct}}%)`;
        savBar.style.background = '#0d1a0d';
        savBar.style.color = '#00ff88';
      }} else {{
        savBar.textContent = `Turn ${{i+1}}: Building context... savings kick in after compression triggers.`;
        savBar.style.background = '#1a1a0d';
        savBar.style.color = '#ffd700';
      }}

      if (document.getElementById('rlm-panel').classList.contains('open')) {{
        await refreshRlm();
      }}

    }} catch (err) {{
      removeSpinner(optMessages);
      removeSpinner(baseMessages);
      addMsg(optMessages, 'assistant', 'Request failed: ' + err.message);
      addMsg(baseMessages, 'assistant', 'Request failed: ' + err.message);
      break;
    }}

    if (i < DEMO_PROMPTS.length - 1) {{
      await new Promise(r => setTimeout(r, 500));
    }}
  }}

  btn.textContent = 'Demo Complete — Reset to Rerun';
  btn.disabled = false;
  btn.onclick = async () => {{
    await resetChat();
    btn.textContent = 'Run Demo Scenario';
    btn.onclick = runDemoScenario;
  }};
  document.getElementById('opt-send').disabled = false;
  document.getElementById('base-send').disabled = false;
  document.getElementById('opt-input').disabled = false;
  document.getElementById('base-input').disabled = false;
  demoRunning = false;
}}
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ComparisonHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send_html(_build_html())
        elif self.path == "/api/stats":
            self._send_json(_get_stats())
        elif self.path == "/api/rlm-state":
            self._send_json(_get_rlm_state())
        elif self.path == "/api/costs":
            tracker = get_cost_tracker()
            self._send_json({
                "summary": tracker.summary(),
                "records": [
                    {
                        "timestamp": r.timestamp,
                        "mode": r.mode,
                        "model": r.model,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cache_read_tokens": r.cache_read_tokens,
                        "total_cost": r.total_cost,
                    }
                    for r in tracker.records
                ],
            })
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            message = body.get("message", "").strip()
            if not message:
                self._send_json({"error": "Empty message"}, 400)
                return
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._send_json({"error": "ANTHROPIC_API_KEY not set"}, 500)
                return
            try:
                result = _handle_chat(message)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path in ("/api/chat/optimized", "/api/chat/baseline", "/api/chat/council"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            message = body.get("message", "").strip()
            if not message:
                self._send_json({"error": "Empty message"}, 400)
                return
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._send_json({"error": "ANTHROPIC_API_KEY not set"}, 500)
                return
            try:
                if self.path == "/api/chat/optimized":
                    result = _handle_solo_optimized(message)
                elif self.path == "/api/chat/council":
                    result = _handle_council(message)
                else:
                    result = _handle_solo_baseline(message)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/api/reset":
            _reset()
            self._send_json({"status": "ok"})
        else:
            self.send_error(404)


def start_comparison_server(port=8060):
    """Start the comparison UI server (blocking)."""
    server = HTTPServer(("", port), ComparisonHandler)
    print(f"Comparison UI running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  AraFlow Comparison UI                           ║")
    print("║  Make sure ANTHROPIC_API_KEY is set               ║")
    print("╚══════════════════════════════════════════════════╝\n")
    start_comparison_server()
