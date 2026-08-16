"""Cue-Tag-Content (CTC) graph structure and traversal operators.

Implements the memory graph used during closed-loop retrieval. Cues (entities,
predicates) link to tags (relational patterns), which link to content (episodic,
semantic, topic layers).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

ContentLayer = Literal["episodic", "semantic", "topic"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class CueNode:
    """A named cue in the graph (entity, predicate, or other retrieval term).

    Attributes:
        cue_id: Unique identifier for the cue.
        tag_set: Set of tags (relational patterns) reachable from this cue.
    """

    cue_id: str
    tag_set: set[str] = field(default_factory=set)


@dataclass
class ContentNode:
    """A content node in the graph (episode, fact, or thematic grouping).

    Attributes:
        content_id: Unique identifier for the content.
        text: The text content (episode description, semantic fact, or topic summary).
        layer: Layer type ("episodic", "semantic", or "topic").
        time: Optional timestamp (for episodic layer).
        tags: List of tags (relational patterns) this content is associated with.
        topic_links: For "topic" layer: list of episode ids grouped under this topic.
    """

    content_id: str
    text: str
    layer: ContentLayer = "episodic"
    time: str | None = None
    tags: list[str] = field(default_factory=list)
    topic_links: list[str] = field(default_factory=list)  # for layer == "topic": episode ids

    def display_text(self) -> str:
        """Text as it should be shown to the LLM: episodic memories carry a
        timestamp (MRAgent Sec. 3.2: "episodic memories are further
        organized along a unified timeline, allowing temporal constraints
        to be imposed during reconstruction") -- without surfacing it here,
        that timeline is stored but never actually reaches the model, so
        temporal questions become unanswerable even when the right episode
        was retrieved.
        """
        return f"[{self.time}] {self.text}" if self.time else self.text


class CueTagContentGraph:
    """The CTC memory graph plus its traversal operators.

    Implements the Cue-Tag-Content (CTC) graph structure used by COLM for
    closed-loop memory-augmented retrieval. The graph stores three types of nodes
    and their relationships:
      - Cues: Retrieval terms (entities, predicates, attributes)
      - Tags: Relational patterns connecting cues to content
      - Content: Episodic episodes, semantic facts, and topic abstractions

    The graph maintains dual indices for efficient bidirectional traversal:
      - cue_tag_to_content: (cue, tag) → content ids
      - content_to_cue_tag: content id → {(cue, tag)} pairs

    Attributes:
        cues: Dict mapping cue_id → CueNode (with associated tag_set).
        contents: Dict mapping content_id → ContentNode (episode, fact, or topic).
        cue_tag_to_content: Forward index from (cue, tag) pairs to content ids.
        content_to_cue_tag: Reverse index from content ids to (cue, tag) pairs.

    Example:
        >>> graph = CueTagContentGraph()
        >>> cue = graph.add_cue("Alice")
        >>> content = graph.add_content("e1", "Alice went to the store", layer="episodic")
        >>> graph.link("Alice", "Shopping", "e1")
        >>> graph.cue_to_tags("Alice")  # {'Shopping'}
        >>> graph.match_query_to_cues("Alice went shopping")  # {'Alice'}
    """

    def __init__(self) -> None:
        self.cues: dict[str, CueNode] = {}
        self.contents: dict[str, ContentNode] = {}
        # phi_(c,g)->v : (cue, tag) -> content ids
        self.cue_tag_to_content: dict[tuple[str, str], set[str]] = {}
        # phi_v->(c,g) : content id -> {(cue, tag)}
        self.content_to_cue_tag: dict[str, set[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def add_cue(self, cue_id: str) -> CueNode:
        """Add or retrieve a cue node in the graph.

        Args:
            cue_id: The cue identifier (e.g., entity name, predicate).

        Returns:
            The CueNode, created if not already present.

        Example:
            >>> graph = CueTagContentGraph()
            >>> cue = graph.add_cue("Alice")
            >>> cue.cue_id
            'Alice'
        """
        return self.cues.setdefault(cue_id, CueNode(cue_id=cue_id))

    def add_content(
        self,
        content_id: str,
        text: str,
        *,
        layer: ContentLayer = "episodic",
        time: str | None = None,
        topic_links: list[str] | None = None,
    ) -> ContentNode:
        """Add a content node (episode, fact, or topic) to the graph.

        Args:
            content_id: Unique identifier for the content (e.g., 'e1', 's2', 't3').
            text: The content text (episode description, semantic fact, or topic summary).
            layer: Content layer type ("episodic", "semantic", or "topic"). Default "episodic".
            time: Optional timestamp for episodic layer items (e.g., "2025-08-17").
            topic_links: For "topic" layer, list of episode ids grouped under this topic.

        Returns:
            The created ContentNode.

        Example:
            >>> graph = CueTagContentGraph()
            >>> episode = graph.add_content("e1", "Alice went shopping", time="2025-08-17")
            >>> fact = graph.add_content("s1", "Alice likes gardening", layer="semantic")
        """
        node = ContentNode(
            content_id=content_id,
            text=text,
            layer=layer,
            time=time,
            topic_links=list(topic_links or []),
        )
        self.contents[content_id] = node
        return node

    def link(self, cue_id: str, tag: str, content_id: str) -> None:
        """Register a Cue-Tag-Content triple (c, g, v) in the graph.

        Creates bidirectional links: the cue node gains the tag, the content
        node tracks the (cue, tag) pair, and both forward and reverse indices
        are updated for efficient traversal.

        Args:
            cue_id: The retrieval cue (entity, predicate, etc.).
            tag: The relational pattern tag (e.g., "Shopping", "Job Change").
            content_id: The content node to link to (must already exist).

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_content("e1", "Alice went shopping")
            >>> graph.link("Alice", "Shopping", "e1")
            >>> ("Alice", "Shopping") in graph.content_to_cue_tag.get("e1", set())
            True
        """
        cue = self.add_cue(cue_id)
        cue.tag_set.add(tag)
        content = self.contents[content_id]
        if tag not in content.tags:
            content.tags.append(tag)
        self.cue_tag_to_content.setdefault((cue_id, tag), set()).add(content_id)
        self.content_to_cue_tag.setdefault(content_id, set()).add((cue_id, tag))

    # ------------------------------------------------------------------
    # Traversal operators (Eq. 5, 8-9)
    # ------------------------------------------------------------------
    def cue_to_tags(self, cue_id: str) -> set[str]:
        """phi_c->g(c): tags activated by a single cue.

        Args:
            cue_id: The cue to query.

        Returns:
            Set of all tags associated with this cue (empty if cue not found).

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_cue("Alice")
            >>> graph.link("Alice", "Shopping", "e1")
            >>> graph.link("Alice", "Work", "e2")
            >>> graph.cue_to_tags("Alice")  # {'Shopping', 'Work'}
        """
        cue = self.cues.get(cue_id)
        return set(cue.tag_set) if cue else set()

    def forward_cue_to_tag(self, active_cues: Iterable[str]) -> set[str]:
        """Pi_c->g(C^(t)) = union_{c in C^(t)} phi_c->g(c)  (Eq. 8).

        Forward traversal: given a set of active cues, return all tags reachable
        from any of them. This is the first step in tag-guided expansion.

        Args:
            active_cues: Iterable of cue ids to expand.

        Returns:
            Set of all tags reachable from the active cues.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.link("Alice", "Shopping", "e1")
            >>> graph.link("Bob", "Work", "e2")
            >>> graph.forward_cue_to_tag(["Alice", "Bob"])  # {'Shopping', 'Work'}
        """
        tags: set[str] = set()
        for cue_id in active_cues:
            tags |= self.cue_to_tags(cue_id)
        return tags

    def forward_tag_to_content(
        self,
        active_cues: Iterable[str],
        active_tags: Iterable[str],
        *,
        exclude_tags: Iterable[str] = (),
        exclude_content_ids: Iterable[str] = (),
    ) -> set[str]:
        """Pi_(c,g)->v(C^(t), G^(t))  (Eq. 8).

        Forward traversal: given active cues and tags, return all content ids
        reachable via (cue, tag) pairs. Can exclude specific tags and content ids,
        providing the mechanism for gap-aware pruning and avoiding re-visited content.

        Args:
            active_cues: Iterable of active cue ids.
            active_tags: Iterable of tags to traverse through.
            exclude_tags: Tags to skip during traversal (gap-aware pruning).
            exclude_content_ids: Content ids already visited, to skip.

        Returns:
            Set of new content ids reachable from the active (cue, tag) pairs,
            excluding any ids in exclude_content_ids.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.link("Alice", "Shopping", "e1")
            >>> graph.link("Alice", "Work", "e2")
            >>> graph.forward_tag_to_content(["Alice"], ["Shopping"])  # {'e1'}
            >>> graph.forward_tag_to_content(
            ...     ["Alice"], ["Shopping", "Work"],
            ...     exclude_tags=["Work"]
            ... )  # {'e1'} (Work is excluded)
        """
        exclude_tags = set(exclude_tags)
        exclude_content_ids = set(exclude_content_ids)
        contents: set[str] = set()
        for cue_id in active_cues:
            for tag in active_tags:
                if tag in exclude_tags:
                    continue
                contents |= self.cue_tag_to_content.get((cue_id, tag), set())
        return contents - exclude_content_ids

    def reverse_content_to_cue_tag(self, active_contents: Iterable[str]) -> set[tuple[str, str]]:
        """Pi_v->(c,g)(V^(t))  (Eq. 9): contents activate new (cue, tag) pairs.

        Reverse traversal: given a set of content ids, return all (cue, tag) pairs
        they are associated with. This enables content-driven cue/tag discovery.

        Args:
            active_contents: Iterable of content ids to query.

        Returns:
            Set of (cue, tag) tuples associated with the active content ids.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.link("Alice", "Shopping", "e1")
            >>> graph.link("Bob", "Shopping", "e1")
            >>> graph.reverse_content_to_cue_tag(["e1"])  # {('Alice', 'Shopping'), ('Bob', 'Shopping')}
        """
        pairs: set[tuple[str, str]] = set()
        for content_id in active_contents:
            pairs |= self.content_to_cue_tag.get(content_id, set())
        return pairs

    def topic_to_episodes(self, topic_content_id: str) -> list[str]:
        """phi_tau->e: a topic node's constituent episodes.

        Args:
            topic_content_id: The id of a topic-layer content node.

        Returns:
            List of episode ids grouped under this topic (empty if not found or not a topic).

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_content("t1", "Topic: Shopping", layer="topic", topic_links=["e1", "e2"])
            >>> graph.topic_to_episodes("t1")  # ['e1', 'e2']
        """
        node = self.contents.get(topic_content_id)
        return list(node.topic_links) if node else []

    # ------------------------------------------------------------------
    # Relevance ranking, for truncating an over-budget candidate set
    # (nodes.py's MAX_ACTIVE_TAGS / MAX_NEW_CONTENT_PER_ROUND) without
    # just keeping an arbitrary (e.g. alphabetical-by-id) slice of it.
    # Same dependency-free token-overlap heuristic used throughout this
    # package wherever a real embedding model isn't assumed to be present.
    # ------------------------------------------------------------------
    def rank_tags_by_relevance(self, tags: Iterable[str], query: str) -> list[str]:
        """Rank tags by token overlap with query.

        Uses a simple dependency-free heuristic: tags are scored by the number
        of tokens they share with the query, then sorted descending by score.
        Ties are broken alphabetically.

        Args:
            tags: Iterable of tag strings to rank.
            query: The reference query string.

        Returns:
            List of tags sorted by relevance to query (most relevant first).

        Example:
            >>> graph = CueTagContentGraph()
            >>> ranked = graph.rank_tags_by_relevance(["Shopping", "Work", "Sleep"], "shopping trip")
            >>> ranked[0]  # 'Shopping' (shares 'shopping' with query)
        """
        query_tokens = _tokenize(query)
        return sorted(tags, key=lambda tag: (-len(_tokenize(tag) & query_tokens), tag))

    def rank_contents_by_relevance(self, content_ids: Iterable[str], query: str) -> list[str]:
        """Rank content ids by token overlap of their text with the query.

        Args:
            content_ids: Iterable of content ids to rank.
            query: The reference query string.

        Returns:
            List of content ids sorted by relevance (most relevant first).

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_content("e1", "Alice went shopping at the store")
            >>> graph.add_content("e2", "Bob went to work")
            >>> ranked = graph.rank_contents_by_relevance(["e1", "e2"], "shopping")
            >>> ranked[0]  # 'e1' (text contains 'shopping')
        """
        query_tokens = _tokenize(query)

        def _score(content_id: str) -> int:
            node = self.contents.get(content_id)
            return -len(_tokenize(node.text) & query_tokens) if node else 0

        return sorted(content_ids, key=lambda cid: (_score(cid), cid))

    # ------------------------------------------------------------------
    # Initial cue matching (MRAgent Sec. 4.2: "extracts a set of
    # fine-grained cues and matches them against the stored cue set")
    # ------------------------------------------------------------------
    def match_query_to_cues(self, query: str, *, max_matches: int = 25) -> set[str]:
        """Match cues to a query using token subset matching.

        A cue matches only if every one of its tokens appears in the query
        (strict subset match). This avoids over-matching on shared filler words
        (e.g., "group of bowls" would not match "group discussion"). Results are
        ranked by specificity (cues with more tokens first), then capped at
        max_matches to prevent context window overflow on large graphs.

        Args:
            query: The query string to match against.
            max_matches: Maximum number of cues to return (default 25).

        Returns:
            Set of matched cue ids (up to max_matches), preferring the most specific.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_cue("Alice Smith")
            >>> graph.add_cue("Alice")
            >>> matches = graph.match_query_to_cues("Alice Smith went shopping")
            >>> "Alice Smith" in matches  # True (2-token cue)
            >>> "Alice" in matches  # True (1-token cue, if within max_matches)
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return set()
        query_lower = query.lower()
        scored: list[tuple[int, str]] = []
        for cue_id in self.cues:
            cue_tokens = _tokenize(cue_id)
            if not cue_tokens:
                continue
            if cue_tokens.issubset(query_tokens) or cue_id.lower() in query_lower:
                scored.append((len(cue_tokens), cue_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {cue_id for _, cue_id in scored[:max_matches]}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a JSON-compatible dict.

        Returns:
            Dict with keys "cues", "contents", and "links". The "links" key
            contains triples (cue_id, tag, content_id) for all graph edges.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_content("e1", "text")
            >>> graph.link("Alice", "Shopping", "e1")
            >>> d = graph.to_dict()
            >>> ("Alice", "Shopping", "e1") in d["links"]
            True
        """
        return {
            "cues": {cid: {"tag_set": sorted(c.tag_set)} for cid, c in self.cues.items()},
            "contents": {
                cid: {
                    "text": c.text,
                    "layer": c.layer,
                    "time": c.time,
                    "tags": c.tags,
                    "topic_links": c.topic_links,
                }
                for cid, c in self.contents.items()
            },
            "links": sorted(
                {
                    (cue_id, tag, content_id)
                    for (cue_id, tag), content_ids in self.cue_tag_to_content.items()
                    for content_id in content_ids
                }
            ),
        }

    def save(self, path: str) -> None:
        """Write the graph to a JSON file.

        Args:
            path: File path to write to.

        Raises:
            IOError: If the file cannot be written.

        Example:
            >>> graph = CueTagContentGraph()
            >>> graph.add_content("e1", "text")
            >>> graph.link("Alice", "Shopping", "e1")
            >>> graph.save("graph.json")
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> CueTagContentGraph:
        """Load a graph from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            A new CueTagContentGraph with all cues, contents, and links restored.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.

        Example:
            >>> graph = CueTagContentGraph.load("graph.json")
            >>> len(graph.contents)  # Check content count
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        graph = cls()
        for cid, content in data.get("contents", {}).items():
            graph.add_content(
                cid,
                content["text"],
                layer=content.get("layer", "episodic"),
                time=content.get("time"),
                topic_links=content.get("topic_links", []),
            )
        for cue_id, tag, content_id in data.get("links", []):
            graph.link(cue_id, tag, content_id)
        return graph
