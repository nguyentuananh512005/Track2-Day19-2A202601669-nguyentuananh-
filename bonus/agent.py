"""Hybrid Memory Agent for AI Assistant (Bonus Challenge).

Combines:
  1. Episodic Memory (Vector Store via Qdrant in-memory + FastEmbed)
     Stores user-specific conversation turns, reading notes, and past interactions.
  2. Stable User Profile & Real-time Velocity (Feature Store via Feast)
     Stores slowly-moving user profile (topic affinity, reading speed, preferred language)
     and fast-moving query velocity (queries in last hour).

Assembles grounding context for LLM prompt generation with zero hallucinations and low latency.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# Optional import of Embedder from app.embeddings, fallback to direct FastEmbed
try:
    from app.embeddings import Embedder
except ImportError:
    Embedder = None


class HybridMemoryAgent:
    """Agent combining Episodic Memory (Qdrant) and User Profile (Feast Feature Store)."""

    def __init__(
        self,
        collection_name: str = "agent_episodic_memory",
        feast_repo_path: str = "app/feast_repo",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.collection_name = collection_name
        self.feast_repo_path = Path(feast_repo_path).resolve()
        self.mem_counter = 0

        # 1. Initialize FastEmbed Embedding Model
        if Embedder is not None:
            self.embedder = Embedder()
            self.dim = self.embedder.dim
        else:
            from fastembed import TextEmbedding

            self.text_embedding = TextEmbedding(model_name=embedding_model)
            self.dim = 384
            self.embedder = None

        # 2. Initialize In-Memory Qdrant Client for Episodic Memory
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

        # 3. Initialize Feast Feature Store
        self.feature_store = None
        self._init_feature_store()

    def _init_feature_store(self) -> None:
        """Connect to Feast Feature Store if initialized, else maintain graceful fallback."""
        try:
            from feast import FeatureStore

            if (self.feast_repo_path / "feature_store.yaml").exists():
                self.feature_store = FeatureStore(repo_path=str(self.feast_repo_path))
        except Exception:
            self.feature_store = None

    def _embed(self, text: str) -> list[float]:
        """Generate embedding vector for input text."""
        if self.embedder is not None:
            vec = next(self.embedder.embed([text]))
            return vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
        else:
            vec = next(self.text_embedding.embed([text]))
            return vec.tolist() if isinstance(vec, np.ndarray) else list(vec)

    def remember(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        user_id: str = "u_001",
    ) -> str:
        """Add a new piece of episodic memory for a specific user into Qdrant Vector Store.

        Args:
            text: Content of the interaction, note, or document chunk.
            metadata: Additional attributes (e.g. topic, source, tag, timestamp).
            user_id: User identifier for multi-tenant isolation.

        Returns:
            memory_id: Unique identifier for the recorded memory.
        """
        self.mem_counter += 1
        memory_id = f"mem_{self.mem_counter:04d}_{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        payload = {
            "memory_id": memory_id,
            "user_id": user_id,
            "text": text,
            "metadata": metadata or {},
            "created_at": now_iso,
        }

        vector = self._embed(text)

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self.mem_counter,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return memory_id

    def get_user_features(self, user_id: str = "u_001") -> dict[str, Any]:
        """Fetch stable profile and recent velocity features from Feast Online Store."""
        default_features = {
            "user_id": user_id,
            "reading_speed_wpm": 200,
            "preferred_language": "vi",
            "topic_affinity": "cloud",
            "queries_last_hour": 5,
            "distinct_topics_24h": 2,
        }

        if self.feature_store is not None:
            try:
                features = self.feature_store.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()

                def _get_val(key: str, default: Any) -> Any:
                    vals = features.get(key)
                    if vals and len(vals) > 0 and vals[0] is not None:
                        return vals[0]
                    return default

                return {
                    "user_id": user_id,
                    "reading_speed_wpm": int(_get_val("reading_speed_wpm", default_features["reading_speed_wpm"])),
                    "preferred_language": str(_get_val("preferred_language", default_features["preferred_language"])),
                    "topic_affinity": str(_get_val("topic_affinity", default_features["topic_affinity"])),
                    "queries_last_hour": int(_get_val("queries_last_hour", default_features["queries_last_hour"])),
                    "distinct_topics_24h": int(_get_val("distinct_topics_24h", default_features["distinct_topics_24h"])),
                }
            except Exception:
                pass

        return default_features

    def recall(
        self,
        query: str,
        user_id: str = "u_001",
        top_k: int = 3,
    ) -> str:
        """Retrieve relevant episodic memories from Qdrant + stable profile from Feast.

        Assembles an enriched prompt context string for the AI assistant.

        Args:
            query: User's prompt/question.
            user_id: Target user ID.
            top_k: Maximum number of episodic memories to retrieve.

        Returns:
            assembled_context: Formatted context string containing profile & relevant memories.
        """
        # 1. Retrieve stable user profile + velocity from Feast
        profile = self.get_user_features(user_id=user_id)

        # 2. Semantic vector search on Qdrant, strictly filtered by user_id
        q_vector = self._embed(query)
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vector,
            query_filter=user_filter,
            limit=top_k,
        )

        retrieved_memories = []
        for point in search_result.points:
            score = point.score
            p_load = point.payload or {}
            retrieved_memories.append(
                {
                    "memory_id": p_load.get("memory_id"),
                    "text": p_load.get("text"),
                    "metadata": p_load.get("metadata", {}),
                    "score": round(float(score), 4),
                }
            )

        # 3. Assemble synthesized context string
        context_lines = [
            "=== ASSEMBLED HYBRID CONTEXT ===",
            f"[User Profile] User ID: {profile['user_id']}",
            f"  - Topic Affinity     : {profile['topic_affinity']}",
            f"  - Preferred Language : {profile['preferred_language']}",
            f"  - Reading Speed      : {profile['reading_speed_wpm']} WPM",
            f"  - Recent Activity    : {profile['queries_last_hour']} queries in last hour | {profile['distinct_topics_24h']} distinct topics in 24h",
            f"[Episodic Memories for Query: '{query}'] (Top {len(retrieved_memories)} retrieved)",
        ]

        if retrieved_memories:
            for idx, mem in enumerate(retrieved_memories, 1):
                tags = mem["metadata"].get("topic", "general")
                context_lines.append(
                    f"  {idx}. [Score: {mem['score']:.4f} | Tag: {tags}] {mem['text']}"
                )
        else:
            context_lines.append("  (No matching episodic memories found for this user)")

        context_lines.append("================================")
        return "\n".join(context_lines)
