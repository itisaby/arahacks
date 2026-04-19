## What is AraFlow?

AraFlow is a **cost optimization layer for LLM APIs** that proves you can get better answers while spending less. It tackles the two biggest problems with production LLM usage: **exploding context windows** and **overreliance on expensive models**.

## How It Works

### Recursive Context Compression (RLM)
- Automatically summarizes older conversation history into compact summary nodes
- Only sends **compressed context + recent messages** to the API — not the entire history
- Achieves **40-60% token reduction** as conversations grow longer
- Uses a hierarchical summary tree with multiple compression levels (L0, L1, L2)

### LLM Council
- Routes each prompt through **3 personas sequentially**: Pragmatist (Haiku) → Critic (Haiku) → Synthesizer (Sonnet)
- Each persona builds on the previous — catching flaws, adding depth, then unifying the answer
- **2x Haiku + 1x Sonnet costs less** than a single Sonnet call, with richer output

## See It Live
- **Dual-pane comparison UI** — optimized vs baseline, side by side
- **Real-time Grafana dashboards** tracking tokens, cost, compression efficiency, and trace visualization via OpenTelemetry
- **One-click demo scenario** with 15-turn enterprise support conversation
