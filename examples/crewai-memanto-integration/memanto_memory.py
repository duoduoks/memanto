"""
MemantoAgentMemory — drop-in memory provider for CrewAI

Uses the official Memanto Python SDK (pip install memanto) as an active
memory layer. Provides typed semantic memory (fact, preference, goal, …)
with provenance, confidence scoring, and cross-session persistence.

Implements the memory interface expected by `crewai.Agent(memory=True)`
so CrewAI agents can remember across sessions without any CLI hackery.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from memanto import MemantoClient

logger = logging.getLogger(__name__)

# ---- constants matching memanto.app.constants ----

SUPPORTED_TYPES = frozenset({
    "fact", "preference", "goal", "decision", "artifact",
    "learning", "event", "instruction", "relationship",
    "context", "observation", "commitment", "error",
})

DEFAULT_TYPE = "fact"
MEMANTO_TYPE_KEY = "_memanto_type"  # stored in crewai memory metadata

# ---- helper: ensure a type string is valid ----

def _coerce_type(raw: str) -> str:
    raw = raw.strip().lower()
    return raw if raw in SUPPORTED_TYPES else DEFAULT_TYPE


class MemantoAgentMemory:
    """
    A CrewAI-compatible memory backend backed by Memanto.

    Typical usage inside a CrewAI agent:

        from memanto_memory import MemantoAgentMemory
        agent = Agent(
            role="Researcher",
            goal="…",
            memory=True,              # crewai will look for agent.memory
            memory_provider=MemantoAgentMemory(
                agent_id="researcher",
                api_key=os.getenv("MOORCHEH_API_KEY"),
            ),
        )

    Design notes
    -------------
    • Every agent gets its own Memanto *namespace* derived from agent_id.
    • Metadata is stored as flat document fields (moorcheh native).
    • Contradiction detection: when store() receives content whose hash
      differs from an existing memory with a similar title, the old
      memory is marked 'superseded' and the new one carries a reference.
    • Confidence defaults to 0.8 unless overridden by the caller.
    """

    def __init__(
        self,
        agent_id: str = "default_agent",
        api_key: Optional[str] = None,
        confidence: float = 0.8,
    ):
        self.agent_id = agent_id
        self.confidence = confidence
        self._api_key = api_key or os.environ.get("MOORCHEH_API_KEY", "")

        if not self._api_key:
            raise RuntimeError(
                "MemantoAgentMemory requires a Moorcheh API key. "
                "Set MOORCHEH_API_KEY env var or pass api_key= explicitly."
            )

        # MemantoClient wraps moorcheh_sdk under the hood
        self._client = MemantoClient(api_key=self._api_key)
        self._namespace = f"crewai_{agent_id}"
        self._ensure_namespace()

    # ── lifecycle ──────────────────────────────────────────────────

    def _ensure_namespace(self) -> None:
        """Create the agent's namespace if it doesn't exist."""
        try:
            namespaces = self._client.namespaces.list()
            existing = {ns.get("name", ns.get("namespace_name", ""))
                        for ns in (namespaces or [])}
        except Exception:
            existing = set()

        if self._namespace not in existing:
            logger.info("Creating namespace %s", self._namespace)
            self._client.namespaces.create(
                namespace_name=self._namespace,
                type="text",
            )

    # ── storage ────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        memory_type: str = DEFAULT_TYPE,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **metadata,
    ) -> dict[str, Any]:
        """
        Store a single memory.

        Parameters
        ----------
        content : str        — the memory text (max ~10 000 chars).
        memory_type : str    — one of the 13 MEMANTO types (auto-coerced).
        title : str | None   — short label; auto-truncated from content.
        tags : list[str]     — optional tags.
        **metadata           — additional flat fields (confidence, source,
                                provenance, …).

        Returns
        -------
        dict with keys: id, title, content, type, confidence, status,
                        created_at, namespace.
        """
        mem_type = _coerce_type(memory_type)
        resolved_title = (
            title or (content[:60] + "…" if len(content) > 60 else content)
        )
        doc_id = f"mem_{int(time.time() * 1000)}_{hash(content) % 10**10}"
        tags_list = tags or []
        confidence = metadata.get("confidence", self.confidence)
        source = metadata.get("source", "agent")
        provenance = metadata.get("provenance", "explicit_statement")

        # ── contradiction check ────────────────────────────────────────
        self._detect_and_mark_contradictions(doc_id, content, resolved_title)

        # ── upload ─────────────────────────────────────────────────────
        doc = {
            "id": doc_id,
            "text": content,
            "metadata": {
                "title": resolved_title,
                MEMANTO_TYPE_KEY: mem_type,
                "confidence": float(confidence),
                "source": source,
                "provenance": provenance,
                "status": "active",
                "agent_id": self.agent_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tags": json.dumps(tags_list),
            },
        }
        # Merge extra metadata
        for k, v in metadata.items():
            if k not in doc["metadata"]:
                doc["metadata"][k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

        result = self._client.documents.upload(
            namespace_name=self._namespace,
            documents=[doc],
        )
        logger.info(
            "Stored memory [%s] | title=%s | id=%s",
            mem_type, resolved_title, doc_id,
        )
        return {
            "id": doc_id,
            "title": resolved_title,
            "content": content,
            "type": mem_type,
            "confidence": confidence,
            "status": "active",
            "created_at": doc["metadata"]["created_at"],
            "namespace": self._namespace,
        }

    def store_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Store multiple memories in one API call.

        Each item should be a dict with at least ``content``.
        """
        results = []
        for item in items:
            results.append(self.store(**item))
        return results

    # ── retrieval ──────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over stored memories.

        Parameters
        ----------
        query : str                    — natural-language search.
        top_k : int                    — how many results (default 10).
        memory_type : str | None       — filter by memory type.
        min_confidence : float         — minimum confidence threshold.

        Returns
        -------
        List of dicts ordered by relevance.
        """
        try:
            raw = self._client.similarity_search.query(
                namespaces=[self._namespace],
                query=query,
                top_k=top_k * 2,  # fetch more for post-filter
            )
        except Exception as exc:
            logger.warning("recall failed: %s", exc)
            return []

        hits = self._parse_search_results(raw)
        hits = self._filter_results(hits, memory_type, min_confidence)
        return hits[:top_k]

    def recall_by_type(
        self,
        memory_type: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Convenience: recall only memories of a given type."""
        return self.recall("", top_k=top_k, memory_type=_coerce_type(memory_type))

    def get_all_active(self) -> list[dict[str, Any]]:
        """Retrieve every active memory for this agent."""
        return self.recall("", top_k=1000)

    # ── answer (grounded QA) ───────────────────────────────────────

    def answer(self, query: str) -> str:
        """
        Generate a grounded answer from the agent's memory.

        Uses Memanto's built-in LLM RAG — no extra API key needed.
        """
        try:
            resp = self._client.answer.generate(
                namespace=self._namespace,
                query=query,
            )
            return resp.get("answer", "") or resp.get("result", "")
        except Exception as exc:
            logger.error("answer failed: %s", exc)
            return ""

    # ── deletion ──────────────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory by its document id."""
        try:
            self._client.documents.delete(
                namespace_name=self._namespace,
                ids=[memory_id],
            )
            return True
        except Exception as exc:
            logger.warning("forget failed for %s: %s", memory_id, exc)
            return False

    def clear(self) -> int:
        """Delete all memories for this agent. Returns count removed."""
        all_memories = self.get_all_active()
        ids = [m["id"] for m in all_memories if m.get("id")]
        if not ids:
            return 0
        try:
            self._client.documents.delete(
                namespace_name=self._namespace,
                ids=ids,
            )
            return len(ids)
        except Exception as exc:
            logger.warning("clear failed: %s", exc)
            return 0

    # ── contradiction detection ────────────────────────────────────

    def _detect_and_mark_contradictions(
        self, new_id: str, new_content: str, new_title: str
    ) -> None:
        """
        Search for existing memories with similar titles.
        If we find one whose content *differs*, mark it as superseded.
        """
        similar = self.recall(new_title, top_k=3)
        new_hash = hash(new_content)
        for mem in similar:
            old_hash = hash(mem.get("content", ""))
            if old_hash != new_hash and mem.get("status") == "active":
                old_id = mem.get("id")
                if old_id and old_id != new_id:
                    logger.info(
                        "Contradiction detected: superseding %s with %s",
                        old_id, new_id,
                    )
                    # Mark old as superseded by updating metadata
                    try:
                        self._client.documents.upload(
                            namespace_name=self._namespace,
                            documents=[{
                                "id": old_id,
                                "text": mem.get("content", ""),
                                "metadata": {
                                    **mem.get("metadata", {}),
                                    "status": "superseded",
                                    "superseded_by": new_id,
                                },
                            }],
                        )
                    except Exception:
                        logger.exception("Failed to mark superseded")

    # ── internal helpers ───────────────────────────────────────────

    @staticmethod
    def _parse_search_results(raw: Any) -> list[dict[str, Any]]:
        """Normalise the moorcheh search response into a uniform list."""
        if isinstance(raw, dict):
            hits = raw.get("results", raw.get("data", raw.get("hits", [])))
        elif isinstance(raw, list):
            hits = raw
        else:
            hits = getattr(raw, "results", []) or getattr(raw, "data", [])

        parsed = []
        for h in (hits or []):
            if isinstance(h, dict):
                meta = h.get("metadata", {})
                text_field = (
                    h.get("text")
                    or h.get("content")
                    or meta.get("text")
                    or meta.get("content", "")
                )
                parsed.append({
                    "id": h.get("id", meta.get("id", "")),
                    "title": meta.get("title", text_field[:50]),
                    "content": text_field,
                    "type": meta.get(MEMANTO_TYPE_KEY, DEFAULT_TYPE),
                    "confidence": float(meta.get("confidence", 0.8)),
                    "status": meta.get("status", "active"),
                    "score": h.get("score", h.get("similarity", 0.0)),
                    "created_at": meta.get("created_at", ""),
                    "tags": json.loads(meta.get("tags", "[]"))
                             if isinstance(meta.get("tags"), str)
                             else meta.get("tags", []),
                    "source": meta.get("source", "agent"),
                    "metadata": meta,
                })
        return parsed

    @staticmethod
    def _filter_results(
        hits: list[dict[str, Any]],
        memory_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Post-filter by type and confidence."""
        filtered = [
            h for h in hits
            if h.get("status") in ("active", None)
            and h.get("confidence", 0.0) >= min_confidence
        ]
        if memory_type:
            filtered = [h for h in filtered if h.get("type") == memory_type]
        return filtered
