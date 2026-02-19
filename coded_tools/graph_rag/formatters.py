# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

"""
Formatting and citation mixin for GraphSearchTool.

Provides methods for formatting search results (facts, entities, relationships,
episodes), building citations, validating claims against source text, and
fetching supplementary data (node details, episode metadata) from the graph
database.
"""
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class FormattersMixin:
    """
    Mixin providing result formatting and citation methods for GraphSearchTool.

    Contains formatters for each result type (facts, entities, relationships,
    episodes), citation builders, claim validation, and database lookup helpers
    for node/episode metadata. All formatting constants are defined here.
    """

    # --- Content truncation ---
    DEFAULT_TRUNCATE_CHARS = 8000

    # --- Display limits ---
    MAX_KEY_CONCEPTS_DISPLAY = 5
    MAX_CITATION_KEY_TERMS = 5
    MAX_CITATION_EXCERPTS = 3
    MAX_CITATION_EPISODES = 3

    # --- Citation validation thresholds ---
    HIGH_CONFIDENCE_THRESHOLD = 0.6
    SUPPORT_CONFIDENCE_THRESHOLD = 0.3
    MIN_CLAIM_WORD_LENGTH = 3
    MIN_CLAIM_LENGTH = 20

    COMMON_STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "must", "shall", "can",
    }

    @staticmethod
    def _truncate_content(content: str, max_chars: int = DEFAULT_TRUNCATE_CHARS) -> str:
        """Truncate content to fit within context limits."""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n\n[... content truncated. Use a more specific query for full text ...]"

    def _format_decision_result(self, decision_id: str, episode) -> str:
        """Format a found decision episode into a readable result."""
        episode_content = episode.content if hasattr(episode, 'content') else str(episode)
        metadata = episode.metadata if hasattr(episode, 'metadata') and episode.metadata else {}

        conference = metadata.get("conference_name", "Unknown")
        year = metadata.get("year", "Unknown")
        location = metadata.get("location", "Unknown")

        action_type = metadata.get("decision_action_type", "")
        action_label = f" [{action_type.upper()}]" if action_type else ""

        parts = [
            f">>> DECISION {decision_id} ({year}){action_label} <<<",
            f"Conference: {conference}",
            f"Year: {year}, Location: {location}",
            "",
            "EXACT TEXT FROM SOURCE:",
            "-" * 80,
            episode_content,
        ]

        return "\n".join(parts)

    def _validate_citation(
        self, claim: str, source_text: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that a claim can be supported by the source text.

        Checks if key terms from the claim appear in the source text and
        returns confidence score and matching excerpts.

        :param claim: The statement/claim to validate
        :param source_text: The source document text
        :param metadata: Document metadata (decision_id, annex, etc.)
        :return: Validation result with confidence score and evidence
        """
        claim_lower = claim.lower()
        source_lower = source_text.lower()

        claim_words = [
            word for word in re.findall(r'\b\w+\b', claim_lower)
            if word not in self.COMMON_STOP_WORDS and len(word) > 3
        ]

        matches = sum(1 for word in claim_words if word in source_lower)
        match_ratio = matches / len(claim_words) if claim_words else 0

        excerpts = []
        for word in claim_words[:self.MAX_CITATION_KEY_TERMS]:
            pattern = re.compile(r'.{0,100}\b' + re.escape(word) + r'\b.{0,100}', re.IGNORECASE | re.DOTALL)
            found = pattern.search(source_text)
            if found:
                excerpt = found.group(0).strip()
                excerpt = ' '.join(excerpt.split())
                if excerpt not in excerpts:
                    excerpts.append(excerpt)

        confidence = (
            "high"
            if match_ratio >= self.HIGH_CONFIDENCE_THRESHOLD
            else "medium"
            if match_ratio >= self.SUPPORT_CONFIDENCE_THRESHOLD
            else "low"
        )

        validation_result = {
            "is_supported": match_ratio >= self.SUPPORT_CONFIDENCE_THRESHOLD,
            "confidence": confidence,
            "match_ratio": round(match_ratio, 2),
            "matched_terms": matches,
            "total_terms": len(claim_words),
            "excerpts": excerpts[:self.MAX_CITATION_EXCERPTS],
            "metadata": metadata,
        }

        return validation_result

    def _extract_claims_from_fact(self, fact: str) -> List[str]:
        """
        Extract individual verifiable claims from a compound fact statement.

        Splits facts on conjunctions and punctuation to isolate individual claims.

        :param fact: The fact statement to break down
        :return: List of individual claims
        """
        claims = re.split(r'[;]|\band\b|\bor\b|\balso\b|\bfurthermore\b|\badditionally\b', fact, flags=re.IGNORECASE)

        claims = [claim.strip() for claim in claims if claim.strip() and len(claim.strip()) > 20]

        if not claims:
            claims = [fact]

        return claims

    async def _format_facts(self, query: str, results: List[Any]) -> str:
        """
        Format general graph facts with SOURCE CITATIONS and connected node details.

        **ENHANCED**: Includes episode sources, metadata, and citation guidance.

        :param query: Original search query
        :param results: List of graph fact results
        :return: Formatted string representation
        """
        if not results:
            return f"No graph facts found matching query: {query}"

        output_parts: List[str] = [
            f"SEARCH RESULTS FOR: '{query}'",
            f"Found {len(results)} knowledge graph fact(s)\n",
            "=" * 80,
            "",
            "NOTE: These facts are extracted from UNFCCC documents.",
            "For exact text and formal citations, search type='episode' is recommended.",
            "",
            "=" * 80,
            "",
        ]

        for i, result in enumerate(results, 1):
            fact: str = result.fact
            name: str = getattr(result, "name", "")
            source_uuid: Optional[str] = getattr(result, "source_node_uuid", None)
            target_uuid: Optional[str] = getattr(result, "target_node_uuid", None)
            episode_uuids: List[str] = getattr(result, "episode_uuids", [])
            created_at = getattr(result, "created_at", None)
            valid_at = getattr(result, "valid_at", None)

            output_parts.append(f"[FACT {i}]")
            if name:
                output_parts.append(f"Relationship Type: {name}")
            output_parts.append(f"Statement: {fact}")

            if episode_uuids:
                output_parts.append(f"\nSource Documents: {len(episode_uuids)} episode(s)")
                for ep_uuid in episode_uuids[:self.MAX_CITATION_EPISODES]:
                    ep_citation_data = await self._get_episode_citation_data(ep_uuid)
                    if ep_citation_data:
                        citation = self._build_citation(
                            ep_citation_data.get("name", ""),
                            ep_citation_data.get("metadata", {}),
                            ep_citation_data.get("source_description", "")
                        )
                        if citation:
                            output_parts.append(f"  • {citation}")

            if valid_at:
                output_parts.append(f"Valid as of: {valid_at}")
            elif created_at:
                output_parts.append(f"Recorded: {created_at}")

            if source_uuid and target_uuid:
                source_node: Dict[str, Any] = await self._get_node_by_uuid(source_uuid)
                target_node: Dict[str, Any] = await self._get_node_by_uuid(target_uuid)

                if source_node or target_node:
                    output_parts.append("\nConnected Entities:")
                    if source_node:
                        output_parts.append(
                            f"  From: {source_node.get('name', 'Unknown')}"
                        )
                        if source_node.get("summary"):
                            output_parts.append(f"       {source_node['summary']}")
                    if target_node:
                        output_parts.append(
                            f"  To: {target_node.get('name', 'Unknown')}"
                        )
                        if target_node.get("summary"):
                            output_parts.append(f"      {target_node['summary']}")

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        output_parts.append("")
        output_parts.append("HOW TO USE THESE RESULTS:")
        output_parts.append("- These facts summarize relationships extracted from documents")
        output_parts.append("- For exact wording and formal citations, use search_type='episode'")
        output_parts.append("- Verify critical claims by checking the source documents listed above")
        output_parts.append("- Temporal context (dates) indicates when the information was valid")

        return "\n".join(output_parts)

    async def _get_episode_citation_data(self, episode_uuid: str) -> Dict[str, Any]:
        """
        Fetch episode data for building citations.

        Queries Episodic nodes for name, source_description, content, and source
        fields used by _build_citation.

        :param episode_uuid: Episode UUID to look up
        :return: Dictionary with episode citation data or empty dict if not found
        """
        if not self._driver_instance:
            return {}

        try:
            query: str = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.name as name, e.source_description as source_description,
                   e.content as content, e.source as source
            LIMIT 1
            """
            records, _, _ = await self._driver_instance.execute_query(
                query, uuid=episode_uuid
            )
            if records:
                record_dict = dict(records[0])
                metadata = {}
                source = record_dict.get("source", "")
                if source:
                    decision_match = re.search(r'decision[s]?\s*(\d+/[A-Z]+\.\d+)', source, re.IGNORECASE)
                    if decision_match:
                        metadata["decision_id"] = decision_match.group(1)

                record_dict["metadata"] = metadata
                return record_dict
            return {}
        except Exception as e:
            print(f"Warning: Failed to fetch episode {episode_uuid}: {e}")
            return {}

    async def _get_node_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """
        Fetch node details by UUID.

        :param uuid: Node UUID to look up
        :return: Dictionary with node details or empty dict if not found
        """
        if not self._driver_instance:
            return {}

        try:
            query: str = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN n.name as name, n.summary as summary, n.entity_type as entity_type, labels(n) as labels
            LIMIT 1
            """
            records, _, _ = await self._driver_instance.execute_query(query, uuid=uuid)
            if records:
                return dict(records[0])
            return {}
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch node {uuid}: {e}")
            return {}

    async def _format_entities(self, query: str, results: List[Any]) -> str:
        """
        Format entity nodes with connected relationships.

        :param query: Original search query
        :param results: List of entity node results
        :return: Formatted string representation
        """
        if not results:
            return f"No entities found matching query: {query}"

        output_parts: List[str] = [
            f"Found {len(results)} relevant entities for query: '{query}'\n",
            "=" * 80,
            "",
        ]

        for i, node in enumerate(results, 1):
            name: str = node.name
            summary: str = getattr(node, "summary", "")
            labels: List[str] = getattr(node, "labels", [])
            entity_type: str = getattr(node, "entity_type", "")
            uuid: str = getattr(node, "uuid", "")

            output_parts.append(f"ENTITY {i}: {name}")

            if entity_type:
                output_parts.append(f"Type: {entity_type}")
            elif labels:
                output_parts.append(f"Type: {', '.join(labels)}")

            if summary:
                output_parts.append(f"\nDescription: {summary}")

            if uuid:
                connections: List[Dict[str, Any]] = await self._get_entity_connections(
                    uuid
                )
                if connections:
                    output_parts.append(
                        f"\nConnected to ({len(connections)} relationships):"
                    )
                    for (
                        conn
                    ) in connections:
                        output_parts.append(
                            f"  - {conn['relationship']}: {conn['target_name']}"
                        )

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    async def _get_entity_connections(self, uuid: str) -> List[Dict[str, Any]]:
        """
        Get relationships connected to an entity.

        :param uuid: Entity UUID to find connections for
        :return: List of connection dictionaries
        """
        if not self._driver_instance:
            return []

        try:
            query: str = """
            MATCH (source:Entity {uuid: $uuid})-[r:RELATES_TO]->(target:Entity)
            RETURN r.name as relationship, target.name as target_name, r.fact as fact
            LIMIT 5
            """
            records, _, _ = await self._driver_instance.execute_query(query, uuid=uuid)
            return [dict(record) for record in records]
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch connections for entity {uuid}: {e}")
            return []

    async def _format_relationships(self, query: str, results: List[Any]) -> str:
        """
        Format relationship edges with connected entity details.

        :param query: Original search query
        :param results: List of relationship edge results
        :return: Formatted string representation
        """
        if not results:
            return f"No relationships found matching query: {query}"

        output_parts: List[str] = [
            f"Found {len(results)} relevant relationships for query: '{query}'\n",
            "=" * 80,
            "",
        ]

        for i, edge in enumerate(results, 1):
            fact: str = edge.fact
            name: str = getattr(edge, "name", "")
            source_uuid: Optional[str] = getattr(edge, "source_node_uuid", None)
            target_uuid: Optional[str] = getattr(edge, "target_node_uuid", None)

            output_parts.append(f"RELATIONSHIP {i}")
            if name:
                output_parts.append(f"Type: {name}")

            output_parts.append(f"Description: {fact}")

            if source_uuid and target_uuid:
                source_node: Dict[str, Any] = await self._get_node_by_uuid(source_uuid)
                target_node: Dict[str, Any] = await self._get_node_by_uuid(target_uuid)

                if source_node or target_node:
                    output_parts.append("\nConnected Entities:")
                    if source_node:
                        output_parts.append(
                            f"  From: {source_node.get('name', 'Unknown')}"
                        )
                        if source_node.get("summary"):
                            output_parts.append(f"        {source_node['summary']}")
                    if target_node:
                        output_parts.append(
                            f"  To: {target_node.get('name', 'Unknown')}"
                        )
                        if target_node.get("summary"):
                            output_parts.append(f"      {target_node['summary']}")

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    async def _format_episodes(self, query: str, results: List[Any]) -> str:
        """
        Format episode nodes with primary source content and metadata.

        If total content exceeds MAX_OUTPUT_CHARS, each episode's content
        is truncated to fit within context limits.

        :param query: Original search query
        :param results: List of episode node results
        :return: Formatted string representation with STRICT CITATIONS
        """
        if not results:
            return f"No primary source documents found matching query: {query}"

        total_content_chars = sum(len(getattr(ep, "content", "") or "") for ep in results)
        needs_truncation = total_content_chars > self.MAX_OUTPUT_CHARS

        if needs_truncation and len(results) > 0:
            per_episode_limit = self.MAX_OUTPUT_CHARS // len(results)
            print(
                f"Total content: {total_content_chars} chars > "
                f"{self.MAX_OUTPUT_CHARS} limit. Truncating each episode "
                f"to ~{per_episode_limit} chars."
            )
        else:
            per_episode_limit = 0

        output_parts: List[str] = [
            f"SEARCH RESULTS FOR: '{query}'",
            f"Found {len(results)} source document(s)\n",
            "=" * 80,
            "",
        ]

        for i, episode in enumerate(results, 1):
            name = getattr(episode, "name", "Unnamed Episode")
            content = getattr(episode, "content", "") or ""
            source_description = getattr(episode, "source_description", "")

            ep_metadata = getattr(episode, 'metadata', None) or {}
            db_metadata: Dict[str, Any] = await self._get_episode_metadata(
                getattr(episode, "uuid", "")
            )
            metadata = {**db_metadata, **{k: v for k, v in ep_metadata.items() if v}}

            citation = self._build_citation(name, metadata, source_description)

            decision_id = metadata.get("decision_id", "")
            annex_id = metadata.get("annex_id", "")
            year = metadata.get("year", "")
            conf_type = metadata.get("conference_type", "")
            action_type = metadata.get("decision_action_type", "")

            output_parts.append(f"[RESULT {i}]")
            if decision_id:
                header = f">>> DECISION {decision_id}"
                if year:
                    header += f" ({year})"
                if action_type:
                    header += f" [{action_type.upper()}]"
                header += " <<<"
                output_parts.append(header)
            if annex_id:
                output_parts.append(f">>> ANNEX {annex_id} to Decision {decision_id} <<<")
            output_parts.append(f"CITATION: {citation}")
            if conf_type:
                output_parts.append(f"Conference: {conf_type} {year}")
            output_parts.append("")
            output_parts.append("EXACT TEXT FROM SOURCE:")
            output_parts.append("-" * 80)

            if content:
                if needs_truncation:
                    output_parts.append(self._truncate_content(content.strip(), per_episode_limit))
                else:
                    output_parts.append(content.strip())
            else:
                output_parts.append("[No content available in database]")

            output_parts.append("")
            output_parts.append("=" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    def _build_citation(
        self, episode_name: str, metadata: Dict[str, Any], source_description: str
    ) -> str:
        """
        Build a formal citation string for an episode.

        :param episode_name: The episode name from the database
        :param metadata: Episode metadata dictionary
        :param source_description: Human-readable source description
        :return: Formatted citation string
        """
        citation_parts = []

        decision_id = metadata.get("decision_id")
        if not decision_id and "::" in episode_name:
            title_part = episode_name.split("::", 1)[1] if "::" in episode_name else ""
            if "Decision" in title_part or "Resolution" in title_part:
                match = re.search(
                    r"(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:/[A-Za-z]+\.\d+)?)",
                    title_part
                )
                if match:
                    decision_id = match.group(1)

        if decision_id:
            citation_parts.append(f"Decision {decision_id}")

        annex_id = metadata.get("annex_id")
        if annex_id:
            citation_parts.append(f"Annex {annex_id}")

        conf_type = metadata.get("conference_type")
        session = metadata.get("session_number")
        year = metadata.get("year")

        if conf_type and session and year:
            citation_parts.append(f"{conf_type} Session {session} ({year})")
        elif source_description:
            citation_parts.append(source_description)

        if citation_parts:
            return ", ".join(citation_parts)
        else:
            return f"Source: {episode_name}"

    async def _get_episode_metadata(self, uuid: str) -> Dict[str, Any]:
        """
        Fetch episode metadata by UUID.

        :param uuid: Episode UUID to look up
        :return: Dictionary with episode metadata or empty dict if not found
        """
        if not self._driver_instance or not uuid:
            return {}

        try:
            query: str = """
            MATCH (e:Episode {uuid: $uuid})
            RETURN e.decision_id as decision_id,
                   e.conference_type as conference_type,
                   e.year as year,
                   e.session_number as session_number,
                   e.annex_id as annex_id
            LIMIT 1
            """
            records, _, _ = await self._driver_instance.execute_query(query, uuid=uuid)
            if records:
                return dict(records[0])
            return {}
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch episode metadata {uuid}: {e}")
            return {}
