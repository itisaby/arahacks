"""
AraFlow Dashboard — Local web UI for visualizing OTel traces, token savings,
and workflow optimization hints.

Run: python dashboard.py
Open: http://localhost:8050
"""

import json
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

from telemetry import get_span_log, get_token_usage
from recursive_summarizer import RecursiveSummarizer

# Import the shared summarizer from app
try:
    from app import summarizer
except ImportError:
    summarizer = RecursiveSummarizer()


def _build_html() -> str:
    spans = get_span_log()
    tokens = get_token_usage()
    ctx_stats = summarizer.get_stats()

    # Per-tool breakdown
    tool_data: dict[str, list[float]] = {}
    for s in spans:
        tool_data.setdefault(s["name"], []).append(s["duration_ms"])

    tool_rows = ""
    for name, durations in sorted(tool_data.items()):
        avg = sum(durations) / len(durations) if durations else 0
        tool_rows += f"""
        <tr>
          <td>{name}</td>
          <td>{len(durations)}</td>
          <td>{avg:.1f}ms</td>
          <td>{max(durations):.1f}ms</td>
          <td>{sum(durations):.1f}ms</td>
        </tr>"""

    # Timeline data for chart
    timeline = json.dumps([
        {"name": s["name"], "time": s["timestamp"], "duration": s["duration_ms"]}
        for s in spans[-50:]
    ])

    total_tokens = tokens["prompt_tokens"] + tokens["completion_tokens"]
    saved = tokens["saved_tokens"]
    efficiency = (saved / (total_tokens + saved) * 100) if (total_tokens + saved) > 0 else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AraFlow Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0f; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.8rem; color: #00d4ff; margin-bottom: 8px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: #12121a; border: 1px solid #1e1e2e; border-radius: 12px;
    padding: 20px; transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #00d4ff33; }}
  .card h3 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #666; margin-bottom: 8px; }}
  .card .value {{ font-size: 2rem; font-weight: bold; }}
  .card .value.green {{ color: #00ff88; }}
  .card .value.blue {{ color: #00d4ff; }}
  .card .value.yellow {{ color: #ffd700; }}
  .card .value.purple {{ color: #b388ff; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #1e1e2e; }}
  th {{ color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .section {{ background: #12121a; border: 1px solid #1e1e2e; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  .section h2 {{ font-size: 1rem; color: #00d4ff; margin-bottom: 16px; }}
  #chart {{ width: 100%; height: 200px; position: relative; }}
  .bar {{ position: absolute; background: #00d4ff44; border-left: 2px solid #00d4ff; border-radius: 0 4px 4px 0; height: 20px; min-width: 4px; }}
  .bar:hover {{ background: #00d4ff88; }}
  .bar-label {{ font-size: 0.65rem; color: #888; white-space: nowrap; overflow: hidden; padding-left: 4px; line-height: 20px; }}
  .refresh-btn {{
    position: fixed; bottom: 24px; right: 24px; background: #00d4ff; color: #0a0a0f;
    border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer;
    font-weight: bold; font-family: inherit;
  }}
  .refresh-btn:hover {{ background: #00b8e0; }}
  .grafana-btn {{
    position: fixed; bottom: 24px; right: 160px; background: #ff6600; color: #fff;
    border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer;
    font-weight: bold; font-family: inherit; text-decoration: none;
  }}
  .grafana-btn:hover {{ background: #e55b00; }}
</style>
</head>
<body>
  <h1>AraFlow Dashboard</h1>
  <p class="subtitle">Self-Optimizing AI Assistant — Workflow Telemetry</p>

  <div class="grid">
    <div class="card">
      <h3>Total Spans</h3>
      <div class="value blue">{len(spans)}</div>
    </div>
    <div class="card">
      <h3>Tokens Used</h3>
      <div class="value yellow">{total_tokens:,}</div>
    </div>
    <div class="card">
      <h3>Tokens Saved</h3>
      <div class="value green">{saved:,}</div>
    </div>
    <div class="card">
      <h3>Efficiency</h3>
      <div class="value purple">{efficiency:.1f}%</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Active Messages</h3>
      <div class="value blue">{ctx_stats.get('raw_messages', 0)}</div>
    </div>
    <div class="card">
      <h3>Summary Nodes</h3>
      <div class="value green">{ctx_stats.get('summary_nodes', 0)}</div>
    </div>
    <div class="card">
      <h3>Active Tokens</h3>
      <div class="value yellow">{ctx_stats.get('active_tokens', 0):,}</div>
    </div>
    <div class="card">
      <h3>Unique Tools</h3>
      <div class="value purple">{len(tool_data)}</div>
    </div>
  </div>

  <div class="section">
    <h2>Tool Performance</h2>
    <table>
      <tr><th>Tool</th><th>Calls</th><th>Avg</th><th>Max</th><th>Total</th></tr>
      {tool_rows if tool_rows else '<tr><td colspan="5" style="color:#666">No spans recorded yet — run the assistant first</td></tr>'}
    </table>
  </div>

  <div class="section">
    <h2>Span Timeline</h2>
    <div id="chart"></div>
  </div>

  <a href="http://localhost:3000/d/araflow-overview" target="_blank" class="grafana-btn">Open Grafana Dashboard</a>
  <button class="refresh-btn" onclick="location.reload()">Refresh</button>

  <script>
    const data = {timeline};
    const chart = document.getElementById('chart');
    if (data.length === 0) {{
      chart.innerHTML = '<p style="color:#666;padding:20px">No spans yet — use the assistant to generate telemetry data</p>';
    }} else {{
      const maxDur = Math.max(...data.map(d => d.duration), 1);
      data.forEach((d, i) => {{
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.top = (i * 24) + 'px';
        bar.style.width = Math.max(4, (d.duration / maxDur) * 80) + '%';
        bar.title = d.name + ' — ' + d.duration.toFixed(1) + 'ms';
        bar.innerHTML = '<span class="bar-label">' + d.name + ' (' + d.duration.toFixed(1) + 'ms)</span>';
        chart.appendChild(bar);
      }});
      chart.style.height = (data.length * 24 + 20) + 'px';
    }}
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "spans": get_span_log(),
                "tokens": get_token_usage(),
                "context": summarizer.get_stats(),
            }).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_build_html().encode())

    def log_message(self, format, *args):
        pass  # suppress noisy logs


if __name__ == "__main__":
    port = 8050
    print(f"AraFlow Dashboard → http://localhost:{port}")
    HTTPServer(("", port), DashboardHandler).serve_forever()
