"""
Recursive Language Model summarization — compress conversation history
hierarchically to keep context within a token budget while preserving
important information.

Strategy:
  1. Chunk the conversation into windows of N messages
  2. Summarize each chunk into a compact representation
  3. When summaries themselves grow large, recursively summarize them
  4. Track tokens saved at each compression level
"""

import hashlib
from dataclasses import dataclass, field
from telemetry import record_tokens, tracer
from opentelemetry.trace import StatusCode


@dataclass
class Message:
    role: str
    content: str
    token_estimate: int = 0

    def __post_init__(self):
        if not self.token_estimate:
            # rough 4-char-per-token heuristic
            self.token_estimate = max(1, len(self.content) // 4)


@dataclass
class SummaryNode:
    """One node in the recursive summary tree."""
    level: int
    text: str
    source_tokens: int
    compressed_tokens: int
    children_hashes: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.source_tokens == 0:
            return 0.0
        return 1 - (self.compressed_tokens / self.source_tokens)


class RecursiveSummarizer:
    """
    Manages a hierarchical summary tree for conversation context.

    Instead of sending the full chat history every turn (expensive),
    older messages are progressively summarized, and the summaries
    themselves are re-summarized when they exceed the budget.
    """

    def __init__(self, chunk_size: int = 6, max_summary_tokens: int = 200):
        self.chunk_size = chunk_size
        self.max_summary_tokens = max_summary_tokens
        self.messages: list[Message] = []
        self.summaries: list[SummaryNode] = []
        self._total_saved: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # trigger compression when buffer exceeds 2x chunk size
        if len(self.messages) > self.chunk_size * 2:
            self._compress_oldest_chunk()

    def get_context(self) -> list[dict]:
        """
        Return a token-efficient context window:
        [recursive summaries] + [recent raw messages]
        """
        ctx = []
        for s in self.summaries:
            ctx.append({"role": "system", "content": f"[Summary L{s.level}] {s.text}"})
        for m in self.messages:
            ctx.append({"role": m.role, "content": m.content})
        return ctx

    def get_stats(self) -> dict:
        raw_tokens = sum(m.token_estimate for m in self.messages)
        summary_tokens = sum(s.compressed_tokens for s in self.summaries)
        return {
            "raw_messages": len(self.messages),
            "summary_nodes": len(self.summaries),
            "active_tokens": raw_tokens + summary_tokens,
            "total_tokens_saved": self._total_saved,
        }

    # ------------------------------------------------------------------
    # Internal compression
    # ------------------------------------------------------------------

    def _compress_oldest_chunk(self):
        with tracer.start_as_current_span("summarizer.compress") as span:
            chunk = self.messages[:self.chunk_size]
            self.messages = self.messages[self.chunk_size:]

            source_tokens = sum(m.token_estimate for m in chunk)
            summary_text = self._summarize_chunk(chunk)
            compressed_tokens = max(1, len(summary_text) // 4)
            saved = source_tokens - compressed_tokens

            node = SummaryNode(
                level=0,
                text=summary_text,
                source_tokens=source_tokens,
                compressed_tokens=compressed_tokens,
                children_hashes=[self._hash(m.content) for m in chunk],
            )
            self.summaries.append(node)
            self._total_saved += saved

            record_tokens(prompt=source_tokens, completion=compressed_tokens, saved=saved)
            span.set_attribute("source_tokens", source_tokens)
            span.set_attribute("compressed_tokens", compressed_tokens)
            span.set_attribute("tokens_saved", saved)

            # recursively compress summaries if too many accumulate
            if len(self.summaries) > self.chunk_size:
                self._compress_summaries()

    def _compress_summaries(self):
        """Level-up: summarize the summaries themselves."""
        with tracer.start_as_current_span("summarizer.meta_compress") as span:
            to_compress = self.summaries[:self.chunk_size]
            self.summaries = self.summaries[self.chunk_size:]

            combined = " | ".join(s.text for s in to_compress)
            source_tokens = sum(s.compressed_tokens for s in to_compress)
            meta_summary = self._extract_key_points(combined)
            compressed_tokens = max(1, len(meta_summary) // 4)
            saved = source_tokens - compressed_tokens

            new_level = max(s.level for s in to_compress) + 1
            node = SummaryNode(
                level=new_level,
                text=meta_summary,
                source_tokens=source_tokens,
                compressed_tokens=compressed_tokens,
            )
            self.summaries.insert(0, node)
            self._total_saved += saved

            record_tokens(prompt=source_tokens, completion=compressed_tokens, saved=saved)
            span.set_attribute("meta_level", new_level)
            span.set_attribute("tokens_saved", saved)

    # ------------------------------------------------------------------
    # Summarization heuristics (local, no extra LLM call needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_chunk(chunk: list[Message]) -> str:
        """
        Extractive summary: keep the first sentence of each message
        and any line that looks like a key fact or action item.
        """
        parts = []
        for m in chunk:
            lines = m.content.strip().split("\n")
            # first line is usually the most important
            if lines:
                parts.append(f"[{m.role}] {lines[0][:120]}")
            # keep action items / key facts
            for line in lines[1:]:
                lower = line.lower().strip()
                if any(kw in lower for kw in ["todo", "remind", "important", "deadline", "action", "schedule", "budget"]):
                    parts.append(f"  -> {line.strip()[:100]}")
        return " ".join(parts)[:800]  # hard cap

    @staticmethod
    def _extract_key_points(text: str) -> str:
        """Further compress by extracting only bracketed role tags and action items."""
        important = []
        for segment in text.split("|"):
            seg = segment.strip()
            if any(kw in seg.lower() for kw in ["todo", "remind", "important", "deadline", "action", "schedule", "budget"]):
                important.append(seg[:100])
            elif seg.startswith("["):
                important.append(seg[:80])
        return " | ".join(important)[:400] if important else text[:400]

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]
