"""Experience bank: episodic stores of past retrieval decisions (Eq. 10).

Two separate banks (Planning and Reflection) hold past experiences indexed by
condition and situation, used to guide traversal via few-shot prompting.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Literal, TypedDict

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Module = Literal["Planning", "Reflection"]


class EmbeddingBackend(ABC):
    """Abstract interface for embedding strategies.

    Pluggable backend for condition/situation encoding and semantic similarity.
    Two implementations: SentenceTransformerEmbeddingBackend (neural) and
    KeywordOverlapEmbeddingBackend (dependency-free fallback).
    """

    @abstractmethod
    def encode(self, text: str) -> object:
        """Encode text into an embedding representation.

        Args:
            text: The input text.

        Returns:
            Embedding object (type depends on backend).
        """
        ...

    @abstractmethod
    def similarity(self, a: object, b: object) -> float:
        """Compute similarity between two embeddings (0 to 1).

        Args:
            a: First embedding.
            b: Second embedding.

        Returns:
            Similarity score (higher is more similar).
        """
        ...


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using sentence-transformers neural models.

    Lazily loads a small sentence-transformers model on first encode() call.
    Never raises on construction; only encode() can raise (allowing graceful
    fallback to KeywordOverlapEmbeddingBackend if the model unavailable).

    Attributes:
        model_name: Name/path of the sentence-transformers model (e.g., "all-MiniLM-L6-v2").
        _model: Lazily-loaded SentenceTransformer instance (None if unavailable).

    Example:
        >>> backend = SentenceTransformerEmbeddingBackend()
        >>> embedding = backend.encode("This is a query")
        >>> similarity = backend.similarity(embedding, another_embedding)
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError(
                f"SentenceTransformerEmbeddingBackend could not load '{self.model_name}' "
                "(package missing or model not downloaded for offline use)."
            )
        return self._model.encode(text).tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:  # type: ignore[override]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class KeywordOverlapEmbeddingBackend(EmbeddingBackend):
    """Dependency-free embedding backend using token counts (sparse vectors).

    Encodes text as a dict of token → count and computes cosine similarity
    over sparse vectors. Requires no external models or packages.

    This backend is used as a fallback when sentence-transformers is unavailable,
    and in offline trajectory collection where neural models aren't needed.

    Example:
        >>> backend = KeywordOverlapEmbeddingBackend()
        >>> emb1 = backend.encode("shopping trip")  # {"shopping": 1, "trip": 1}
        >>> emb2 = backend.encode("shopping is fun")  # {"shopping": 1, "is": 1, "fun": 1}
        >>> sim = backend.similarity(emb1, emb2)  # Cosine similarity
    """

    def encode(self, text: str) -> dict[str, int]:  # type: ignore[override]
        counts: dict[str, int] = {}
        for token in text.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if token:
                counts[token] = counts.get(token, 0) + 1
        return counts

    def similarity(self, a: dict[str, int], b: dict[str, int]) -> float:  # type: ignore[override]
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
        return dot / (norm_a * norm_b)


def build_default_embedding_backend() -> EmbeddingBackend:
    """Build the best available embedding backend, with fallback.

    Tries to load a SentenceTransformer model; falls back to keyword-overlap
    if the model is unavailable or the package is not installed.

    Returns:
        A usable EmbeddingBackend instance (never raises).
    """
    backend = SentenceTransformerEmbeddingBackend()
    try:
        backend.encode("warmup")
        return backend
    except Exception:
        return KeywordOverlapEmbeddingBackend()


class ExperienceEntry(TypedDict):
    condition: str
    situation: str
    experience: str
    module: Module
    embedding: object  # List[float] or Dict[str, int], depending on backend


class ExperienceBank:
    """Episodic store of past retrieval decisions for Planning and Reflection.

    Maintains two separate experience banks (D^P and D^R from Eq. 10):
      - Planning bank: Experiences keyed by refined queries (q_i) for planning steps
      - Reflection bank: Experiences keyed by (original query + evidence) for reflection

    Retrieval (Eq. 12) uses embedding-based similarity over condition+situation pairs.
    Embeddings are re-derived on load, allowing flexible backend selection.

    Attributes:
        backend: The EmbeddingBackend used for encoding and similarity.
        planning_bank: List of Planning experiences.
        reflection_bank: List of Reflection experiences.

    Example:
        >>> backend = build_default_embedding_backend()
        >>> bank = ExperienceBank(backend)
        >>> entry = bank.add_experience(
        ...     condition="find related documents",
        ...     situation="retrieval query",
        ...     experience="IF retrieval query THEN prefer cue expansion",
        ...     module="Planning"
        ... )
        >>> retrieved = bank.retrieve("find docs", "query", module="Planning", k=3)
    """

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.planning_bank: list[ExperienceEntry] = []
        self.reflection_bank: list[ExperienceEntry] = []

    def _bank_for(self, module: Module) -> list[ExperienceEntry]:
        return self.planning_bank if module == "Planning" else self.reflection_bank

    def add_experience(
        self, condition: str, situation: str, experience: str, module: Module
    ) -> ExperienceEntry:
        """Add a past experience to the appropriate bank (Planning or Reflection).

        Encodes the condition+situation pair and stores the full entry.

        Args:
            condition: The retrieval condition (e.g., current query for Planning).
            situation: Abstracted situation matching past experience (e.g., "find_information").
            experience: The past advice/decision (e.g., "retrieved X, which was helpful").
            module: Bank to add to ("Planning" or "Reflection").

        Returns:
            The created ExperienceEntry with embedding attached.
        """
        embedding = self.backend.encode(condition + " " + situation)
        entry: ExperienceEntry = {
            "condition": condition,
            "situation": situation,
            "experience": experience,
            "module": module,
            "embedding": embedding,
        }
        self._bank_for(module).append(entry)
        return entry

    def retrieve(
        self, condition: str, situation: str, module: Module, k: int = 3
    ) -> list[ExperienceEntry]:
        """Retrieve top-k relevant past experiences (Eq. 12).

        Encodes the condition+situation pair and returns the k most similar
        entries from the specified bank, ranked by embedding similarity.

        Args:
            condition: The current retrieval condition.
            situation: Abstracted current situation.
            module: Bank to retrieve from ("Planning" or "Reflection").
            k: Number of experiences to return (default 3).

        Returns:
            List of up to k ExperienceEntry objects, sorted by descending similarity.
        """
        bank = self._bank_for(module)
        if not bank:
            return []
        query_embedding = self.backend.encode(condition + " " + situation)
        scored = [
            (self.backend.similarity(query_embedding, entry["embedding"]), entry) for entry in bank
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    # ------------------------------------------------------------------
    # Serialization (embeddings stripped, re-derived on load)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Export bank as a dict, stripping embeddings (re-derived on load).

        Returns:
            Dict with keys "planning_bank" and "reflection_bank", each a list
            of entries (condition, situation, experience, module).
        """

        def strip_embedding(entries: list[ExperienceEntry]) -> list[dict]:
            return [{k: v for k, v in entry.items() if k != "embedding"} for entry in entries]

        return {
            "planning_bank": strip_embedding(self.planning_bank),
            "reflection_bank": strip_embedding(self.reflection_bank),
        }

    def save(self, path: str) -> None:
        """Serialize the bank to JSON (embeddings omitted).

        Args:
            path: File path to write to.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str, backend: EmbeddingBackend) -> ExperienceBank:
        """Load a bank from JSON and re-derive embeddings.

        Reads the serialized dict and re-encodes the condition+situation pairs
        using the provided backend. This allows embeddings to be regenerated
        on a different environment if the backend (neural model, etc.) differs.

        Args:
            path: Path to the JSON checkpoint.
            backend: EmbeddingBackend to use for re-encoding.

        Returns:
            A new ExperienceBank with both banks populated and embeddings re-derived.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        bank = cls(backend)
        for entry in data.get("planning_bank", []):
            bank.add_experience(
                entry["condition"], entry["situation"], entry["experience"], "Planning"
            )
        for entry in data.get("reflection_bank", []):
            bank.add_experience(
                entry["condition"], entry["situation"], entry["experience"], "Reflection"
            )
        return bank


def empty_experience_bank() -> ExperienceBank:
    """Create an empty bank for offline trajectory bootstrapping.

    Used when no past experiences are available (e.g., the first run or
    offline collection phase). Always returns a bank with the keyword-overlap
    backend, making it independent of external model availability.

    Returns:
        An ExperienceBank with both Planning and Reflection banks empty.
    """
    return ExperienceBank(KeywordOverlapEmbeddingBackend())


# ----------------------------------------------------------------------
# Condition construction (R2-Mem Eq. 11)
# ----------------------------------------------------------------------
def planning_condition(current_refined_query: str) -> str:
    """Construct a Planning bank lookup condition (Eq. 11).

    For Planning, the condition is simply the current refined query (c_i = q_i).
    This is used for experience retrieval: past experiences for similar queries
    guide the current planning step.

    Args:
        current_refined_query: The query as refined by previous reflection steps
            (or the original query on the first round).

    Returns:
        The condition string (identical to input).

    Example:
        >>> cond = planning_condition("find information about Alice")
        >>> cond  # "find information about Alice"
    """
    return current_refined_query


def reflection_condition(original_query: str, accumulated_evidence: list[str]) -> str:
    """Construct a Reflection bank lookup condition (Eq. 11).

    For Reflection, the condition combines the original query and evidence
    gathered so far: [Q + m_i]. This context guides the reflection step's
    judgment on whether to continue or finalize the answer.

    Args:
        original_query: The original (unchanged) user query.
        accumulated_evidence: List of evidence gathered in previous rounds.

    Returns:
        A combined condition string: "{original_query} | evidence so far: {joined}".

    Example:
        >>> cond = reflection_condition(
        ...     "find Alice's job",
        ...     ["Alice works at Company X", "She started in 2020"]
        ... )
        >>> "evidence so far:" in cond  # True
    """
    evidence_block = "; ".join(accumulated_evidence)
    return f"{original_query} | evidence so far: {evidence_block}"
