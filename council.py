"""
LLM Council — Multi-persona debate engine for deep discussions.

Multiple AI personas with distinct thinking styles debate a topic
in rounds, challenge each other, and converge on a best solution.
All rounds are OTel-traced for workflow visualization.
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telemetry import tracer, traced_tool, record_tokens, _record_span

# ---------------------------------------------------------------------------
# Council Member Personas
# ---------------------------------------------------------------------------

PERSONAS = {
    "pragmatist": {
        "name": "The Pragmatist",
        "emoji": "🔧",
        "style": (
            "You are The Pragmatist. You focus on practical, actionable solutions. "
            "You prioritize feasibility, cost, and time-to-deliver. You push back on "
            "over-engineered ideas and ask 'but will it actually work in production?'"
        ),
    },
    "critic": {
        "name": "The Critic",
        "emoji": "🔍",
        "style": (
            "You are The Critic. You find flaws, edge cases, and risks in every proposal. "
            "You play devil's advocate and stress-test ideas. You ask 'what could go wrong?' "
            "and 'what are we missing?' You are not negative — you make ideas stronger."
        ),
    },
    "visionary": {
        "name": "The Visionary",
        "emoji": "🚀",
        "style": (
            "You are The Visionary. You think big and propose creative, ambitious solutions. "
            "You connect ideas across domains and see possibilities others miss. You ask "
            "'what if we could...' and 'imagine a world where...'"
        ),
    },
    "synthesizer": {
        "name": "The Synthesizer",
        "emoji": "🧬",
        "style": (
            "You are The Synthesizer. You find common ground between opposing views. "
            "You combine the best parts of each argument into a unified solution. "
            "You summarize debates fairly and propose compromises that satisfy all sides."
        ),
    },
}


@dataclass
class CouncilRound:
    """One round of debate."""
    round_num: int
    responses: list[dict] = field(default_factory=list)

    def add_response(self, persona_id: str, content: str):
        self.responses.append({
            "persona": persona_id,
            "name": PERSONAS[persona_id]["name"],
            "content": content,
        })


@dataclass
class CouncilSession:
    """A full council debate session."""
    topic: str
    persona_ids: list[str]
    rounds: list[CouncilRound] = field(default_factory=list)
    verdict: str = ""
    total_tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "personas": [PERSONAS[p]["name"] for p in self.persona_ids],
            "num_rounds": len(self.rounds),
            "rounds": [
                {
                    "round": r.round_num,
                    "responses": r.responses,
                }
                for r in self.rounds
            ],
            "verdict": self.verdict,
            "total_tokens_used": self.total_tokens_used,
        }


# ---------------------------------------------------------------------------
# Debate Engine (local heuristic — no extra API calls needed)
# ---------------------------------------------------------------------------

class CouncilEngine:
    """
    Runs a structured multi-round debate between personas.

    This uses prompt-construction so the Ara agent itself role-plays
    each persona in sequence. The tool returns the full structured
    debate for the agent to present to the user.
    """

    def __init__(self):
        self.sessions: list[CouncilSession] = []

    def create_debate_prompt(
        self,
        topic: str,
        persona_ids: list[str] | None = None,
        num_rounds: int = 3,
        user_context: str = "",
    ) -> dict:
        """
        Build a structured council debate prompt that the Ara agent
        can execute turn-by-turn.
        """
        if persona_ids is None:
            persona_ids = ["pragmatist", "visionary", "critic", "synthesizer"]

        # Validate personas
        persona_ids = [p for p in persona_ids if p in PERSONAS]
        if len(persona_ids) < 2:
            persona_ids = ["pragmatist", "critic"]

        session = CouncilSession(topic=topic, persona_ids=persona_ids)
        self.sessions.append(session)

        # Build the structured debate instructions
        debate_script = self._build_debate_script(topic, persona_ids, num_rounds, user_context)

        return {
            "session_index": len(self.sessions) - 1,
            "debate_prompt": debate_script,
            "personas": {pid: PERSONAS[pid] for pid in persona_ids},
            "num_rounds": num_rounds,
            "instructions": (
                "Execute this debate by role-playing each persona in order. "
                "For each round, adopt the persona's thinking style and respond "
                "to previous arguments. After all rounds, The Synthesizer delivers "
                "the final verdict combining the best ideas."
            ),
        }

    def _build_debate_script(
        self, topic: str, persona_ids: list[str], num_rounds: int, user_context: str
    ) -> str:
        lines = []
        lines.append(f"# LLM Council Debate: {topic}\n")

        if user_context:
            lines.append(f"**User context:** {user_context}\n")

        lines.append("## Council Members\n")
        for pid in persona_ids:
            p = PERSONAS[pid]
            lines.append(f"- **{p['emoji']} {p['name']}**: {p['style'][:80]}...")
        lines.append("")

        for r in range(1, num_rounds + 1):
            lines.append(f"## Round {r}" + (" — Opening Statements" if r == 1 else f" — Rebuttals & Refinement"))
            lines.append("")
            for pid in persona_ids:
                p = PERSONAS[pid]
                if r == 1:
                    lines.append(
                        f"### {p['emoji']} {p['name']}\n"
                        f"*Present your initial position on: {topic}*\n"
                        f"*Stay in character: {p['style'][:60]}...*\n"
                    )
                else:
                    lines.append(
                        f"### {p['emoji']} {p['name']}\n"
                        f"*Respond to the other council members' arguments. "
                        f"Challenge weak points. Build on strong ideas.*\n"
                    )

        # Final synthesis
        if "synthesizer" in persona_ids:
            lines.append("## Final Verdict — The Synthesizer\n")
            lines.append(
                "*Combine the strongest arguments from all rounds into a clear, "
                "actionable recommendation. List concrete next steps.*\n"
            )
        else:
            lines.append("## Final Verdict\n")
            lines.append(
                "*Summarize the key points of agreement and disagreement. "
                "Provide a balanced recommendation with concrete next steps.*\n"
            )

        return "\n".join(lines)

    def get_session(self, index: int) -> dict | None:
        if 0 <= index < len(self.sessions):
            return self.sessions[index].to_dict()
        return None

    def list_sessions(self) -> list[dict]:
        return [
            {"index": i, "topic": s.topic, "rounds": len(s.rounds)}
            for i, s in enumerate(self.sessions)
        ]


# Shared engine instance
council_engine = CouncilEngine()


# ---------------------------------------------------------------------------
# Live API Council — 3 sequential persona calls for comparison UI
# ---------------------------------------------------------------------------

# Per-model pricing
_MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {
        "cost_per_input": 1.0 / 1_000_000,   # $1 per 1M input
        "cost_per_output": 5.0 / 1_000_000,   # $5 per 1M output
    },
    "claude-sonnet-4-20250514": {
        "cost_per_input": 3.0 / 1_000_000,    # $3 per 1M input
        "cost_per_output": 15.0 / 1_000_000,   # $15 per 1M output
    },
}

_COUNCIL_PERSONAS = [
    {
        "id": "pragmatist",
        "model": "claude-haiku-4-5-20251001",
        "model_label": "haiku",
        "system": (
            "You are The Pragmatist. Focus on practical, actionable solutions. "
            "Prioritize feasibility, cost, and time-to-deliver. Be concise and direct."
        ),
    },
    {
        "id": "critic",
        "model": "claude-haiku-4-5-20251001",
        "model_label": "haiku",
        "system": (
            "You are The Critic. Find flaws, edge cases, and risks in the previous analysis. "
            "Play devil's advocate. Stress-test ideas. Point out what could go wrong. Be concise."
        ),
    },
    {
        "id": "synthesizer",
        "model": "claude-sonnet-4-20250514",
        "model_label": "sonnet",
        "system": (
            "You are The Synthesizer. You have read the Pragmatist's practical analysis "
            "and the Critic's objections. Combine the strongest points into a single, "
            "unified, actionable answer. Resolve disagreements. Be thorough but concise."
        ),
    },
]


def run_live_council(client, message: str, history: list[dict] | None = None) -> dict:
    """
    Run a live 3-persona council via the Claude API.

    Each persona is called sequentially — later personas see earlier responses.
    Returns structured result with per-persona breakdowns and cost.
    """
    if history is None:
        history = []

    council_rounds = []
    accumulated_context = ""
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for persona in _COUNCIL_PERSONAS:
        # Build messages for this persona
        messages = list(history)  # copy conversation history

        # Compose user content: original prompt + prior persona outputs
        user_content = message
        if accumulated_context:
            user_content = (
                f"{message}\n\n"
                f"--- Prior council analysis ---\n{accumulated_context}"
            )

        messages.append({"role": "user", "content": user_content})

        response = client.messages.create(
            model=persona["model"],
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": persona["system"],
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )

        reply = response.content[0].text
        usage = response.usage
        inp_tok = usage.input_tokens
        out_tok = usage.output_tokens

        # Cost for this persona's model
        pricing = _MODEL_PRICING.get(persona["model"], _MODEL_PRICING["claude-sonnet-4-20250514"])
        call_cost = inp_tok * pricing["cost_per_input"] + out_tok * pricing["cost_per_output"]

        council_rounds.append({
            "persona": persona["id"],
            "model": persona["model_label"],
            "reply": reply,
            "input_tokens": inp_tok,
            "output_tokens": out_tok,
            "cost": round(call_cost, 8),
        })

        total_input += inp_tok
        total_output += out_tok
        total_cost += call_cost

        # Accumulate context for next persona
        name = PERSONAS[persona["id"]]["name"]
        emoji = PERSONAS[persona["id"]]["emoji"]
        accumulated_context += f"\n{emoji} {name}:\n{reply}\n"

    return {
        "final_reply": council_rounds[-1]["reply"],  # Synthesizer's output
        "council_rounds": council_rounds,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(total_cost, 8),
    }
