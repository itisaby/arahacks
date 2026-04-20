# AraFlow Agent Log — GitHub Tool Integration

**Date:** 2026-04-19
**Task:** Add GitHub tool calls to Comparison UI via Ara Connectors / Direct API

---

## Session Summary

### Initial Plan
The original plan was to add a `web_search` tool (DuckDuckGo API) to `comparison.py` to light up Grafana's Tool Performance panels and make the demo more impressive.

### Pivot: User Requested Ara Connectors
User asked to use **Ara Connectors** (Messages + GitHub) instead of web search, since they had already connected both on the Ara platform.

### Investigation: Ara Connectors Architecture
Explored the `ara_sdk` package to understand how connectors work:

```python
import ara_sdk
print(dir(ara_sdk))
# ['AraClient', 'Automation', '__all__', ..., 'connectors', 'core', 'env', 'secret', 'tool']

gh = ara_sdk.connectors.github   # → <ara.connectors.github>
msg = ara_sdk.connectors.messages # → <ara.connectors.messages>
```

**Key finding:** Ara connectors are **platform-level integrations**. They are specified via `skills=[ara.connectors.github]` in `ara.Automation()` and the Ara platform injects the actual tool implementations at runtime. They do **not** work when calling the Claude API directly (as `comparison.py` does).

**Source:** `/opt/anaconda3/lib/python3.13/site-packages/ara_sdk/core.py`
- `_ConnectorSkillRef` class (line 960) — generates string tokens like `connector:github`
- `_ConnectorsNamespace` class (line 988) — lazy attribute access creates refs
- `Automation.__init__` (line 1999) — merges connector refs into `tool_privileges`

### Decision: User Chose Comparison UI Only
User confirmed they want GitHub integration in the **Comparison UI** (`comparison.py` on `:8060`), not through `ara run app.py`. This required direct GitHub REST API implementation.

---

## Changes Made

### 1. `app.py` — Added Ara Connectors (for platform deployment)

**What changed:**
- Added `skills=[ara.connectors.github, ara.connectors.messages]` to `ara.Automation()`
- Updated `system_instructions` to describe GitHub and Messages capabilities

**Lines affected:** Automation definition block (~line 261-316)

### 2. `comparison.py` — Added Direct GitHub API Integration

#### New imports (line 11-16)
```python
import urllib.request
import urllib.parse
from telemetry import ..., traced_tool
```

#### New: `GH_TOKEN` env var (line 31)
```python
GH_TOKEN = os.environ.get("GH_TOKEN", "")
```

#### New: GitHub API helper (line 36)
```python
def _gh_api(endpoint: str) -> dict | list:
    """Call GitHub REST API with Bearer auth."""
```

#### New: 5 GitHub tool functions (all decorated with `@traced_tool`)
| Function | Description |
|----------|-------------|
| `github_list_repos()` | List repos for authenticated user or a username |
| `github_get_repo()` | Detailed repo info (stars, forks, issues, topics) |
| `github_list_issues()` | List issues for a repo (filterable by state) |
| `github_list_pull_requests()` | List PRs for a repo (filterable by state) |
| `github_get_notifications()` | Get unread notifications |

#### New: Claude tool schemas (`GITHUB_TOOLS` list)
5 tool definitions with `input_schema` for Claude's tool_use protocol.

#### New: Tool handler dispatch map (`_TOOL_HANDLERS`)
Maps tool names to lambda wrappers that call the `@traced_tool` functions.

#### New: `_run_tool_use_loop()` function
- Runs Claude with `tools=GITHUB_TOOLS` (only if `GH_TOKEN` is set)
- Handles `tool_use` content blocks in response
- Executes tool calls, appends `tool_result` messages, calls Claude again
- Max 3 iterations to prevent infinite loops
- Accumulates tokens across all rounds
- Returns `(reply, input_tokens, output_tokens, tool_calls_made, cache_read, cache_creation)`

#### New: Dynamic system prompt (`_get_system_prompt()`)
- With `GH_TOKEN`: instructs Claude about GitHub tool availability
- Without `GH_TOKEN`: original simple prompt

#### Modified: `_call_optimized()`
- Replaced direct `client.messages.create()` with `_run_tool_use_loop()`
- Added `tool_calls_made` to return dict

#### Modified: `_call_baseline()`
- Same changes as optimized

#### Modified: Frontend JavaScript
- `addMsg()` function now accepts `toolCalls` parameter
- Badge shows `GH:N` when tools are used (e.g., `362 tok · $0.0059 · GH:2`)
- Both per-pane send and shared send (demo scenario) pass tool call counts

---

## Bug Fix

### UnicodeEncodeError with emoji in f-string template
**Error:** `UnicodeEncodeError: 'utf-8' codec can't encode characters in position 17267-17268: surrogates not allowed`

**Cause:** The magnifying glass emoji (U+1F50D) was written as a Python surrogate pair `\ud83d\udd0d` inside the f-string HTML template. Python's `str.encode('utf-8')` rejects surrogate characters.

**Fix:** Replaced emoji with plain text `GH:N` in the badge label.

---

## How to Use

```bash
export GH_TOKEN="ghp_your_token_here"
export ANTHROPIC_API_KEY="sk-ant-..."
python comparison.py
```

Then in the chat UI at `http://localhost:8060`:
- "What repos do I have?"
- "Show me open issues on owner/repo"
- "Any open PRs on my-org/my-repo?"
- "What are my GitHub notifications?"

When `GH_TOKEN` is not set, the UI falls back to normal chat (no tools).

---

## Grafana Impact

All 5 GitHub tools use `@traced_tool` from `telemetry.py`, which automatically records:
- `araflow.tool.calls` counter (by tool name)
- `araflow.tool.duration` histogram (by tool name)
- OTel spans with `tool.name`, `tool.args`, `duration_ms` attributes

This populates the Grafana Tool Performance panels (Duration Heatmap, Call Frequency, P95 Latency).

---

## Files Modified
| File | Status |
|------|--------|
| `comparison.py` | Modified — GitHub tools + tool-use loop + frontend badge |
| `app.py` | Modified — Added `skills=[ara.connectors.github, ara.connectors.messages]` |
