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
