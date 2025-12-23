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
Graph search tool for UNFCCC climate documents using FalkorDB knowledge graph.

This module implements a Graph-based Retrieval-Augmented Generation (RAG) tool that provides
semantic search capabilities over a knowledge graph of climate conference documents stored in
FalkorDB. It leverages the Graphiti library to perform hybrid search (combining vector similarity
and keyword matching) over entities and relationships extracted from UNFCCC documents including
COP, CMA, CMP, SBI, and SBSTA proceedings.

Architecture:
    - Uses Graphiti Core for graph operations and hybrid search
    - Connects to FalkorDB (Redis-based graph database) for knowledge storage
    - Implements singleton pattern for graph connections (shared across invocations)
    - Supports three search modes: general facts, entities, and relationships
    - Automatically enriches results with connected nodes and relationships

Search Types:
    1. "general" (default): Searches for general facts and relationships in the graph.
       Returns facts with connected entity details.

    2. "entity": Searches for entity nodes (e.g., countries, organizations, concepts).
       Returns entities with their summaries and connected relationships.

    3. "relationship": Searches for relationship edges between entities.
       Returns relationships with source and target entity details.
"""
import os
import traceback
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from neuro_san.interfaces.coded_tool import CodedTool

# Load environment variables BEFORE importing graphiti
_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver


class GraphSearchTool(CodedTool):
    """
    Graph-based RAG tool for semantic search over UNFCCC climate documents.

    Performs hybrid search (vector + keyword) over knowledge graph stored in FalkorDB.
    Supports three search modes: general facts, entities, and relationships.
    """

    # Class-level instances shared across all invocations
    _graphiti_instance: Optional[Graphiti] = None
    _driver_instance: Optional[FalkorDriver] = None

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """
        Execute graph relationship search and return relevant facts.

        :param args: Dictionary with query and optional parameters
        :param sly_data: Additional context data (not used)
        :return: Formatted string with retrieved graph facts
        """
        try:
            # Initialize graph if needed
            if GraphSearchTool._graphiti_instance is None:
                await self._initialize_graph()

            # Extract and validate arguments
            query: str = args.get("query")
            if not query:
                return "Error: query parameter is required"

            # Convert string inputs to appropriate types
            limit: int = int(args.get("limit", 10))
            search_type: str = args.get("search_type", "general")

            # Perform search based on type
            if search_type == "entity":
                results: List[Any] = await self._search_entities(query, limit)
                formatted_output: str = await self._format_entities(query, results)
            elif search_type == "relationship":
                results = await self._search_relationships(query, limit)
                formatted_output = await self._format_relationships(query, results)
            else:
                results = await self._search_graph(query, limit)
                formatted_output = await self._format_facts(query, results)

            return formatted_output

        # pylint: disable=broad-exception-caught
        except Exception as exception:
            error_msg: str = f"Error executing graph search: {str(exception)}"
            traceback.print_exc()
            return error_msg

    async def _initialize_graph(self) -> None:
        """
        Initialize FalkorDB connection and Graphiti instance.

        :return: None
        """
        GraphSearchTool._driver_instance = FalkorDriver(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            username=os.getenv("FALKORDB_USERNAME"),
            password=os.getenv("FALKORDB_PASSWORD"),
            database=os.getenv("GRAPH_NAME", "unfccc_knowledge_graph"),
        )

        GraphSearchTool._graphiti_instance = Graphiti(
            graph_driver=GraphSearchTool._driver_instance
        )

    async def _search_graph(self, query: str, limit: int) -> List[Any]:
        """
        Search graph for general facts and relationships.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of search results
        """
        results = await GraphSearchTool._graphiti_instance.search(
            query=query, num_results=limit
        )
        return results or []

    async def _search_entities(self, query: str, limit: int) -> List[Any]:
        """
        Search for entity nodes in the graph.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of entity nodes
        """
        from graphiti_core.search.search_config_recipes import \
            NODE_HYBRID_SEARCH_RRF

        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        results = await GraphSearchTool._graphiti_instance._search(
            query=query, config=config
        )

        return results.nodes if results and results.nodes else []

    async def _search_relationships(self, query: str, limit: int) -> List[Any]:
        """
        Search for relationship edges in the graph.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of relationship edges
        """
        from graphiti_core.search.search_config_recipes import \
            EDGE_HYBRID_SEARCH_RRF

        config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        results = await GraphSearchTool._graphiti_instance._search(
            query=query, config=config
        )

        return results.edges if results and results.edges else []

    async def _format_facts(self, query: str, results: List[Any]) -> str:
        """
        Format general graph facts with connected node details.

        :param query: Original search query
        :param results: List of graph fact results
        :return: Formatted string representation
        """
        if not results:
            return f"No graph facts found matching query: {query}"

        output_parts: List[str] = [
            f"Found {len(results)} relevant facts for query: '{query}'\n",
            "=" * 80,
            "",
        ]

        for i, result in enumerate(results, 1):
            fact: str = result.fact
            name: str = getattr(result, "name", "")
            source_uuid: Optional[str] = getattr(result, "source_node_uuid", None)
            target_uuid: Optional[str] = getattr(result, "target_node_uuid", None)

            output_parts.append(f"FACT {i}")
            if name:
                output_parts.append(f"Relationship Type: {name}")
            output_parts.append(f"Statement: {fact}")

            # Get connected node details
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

        return "\n".join(output_parts)

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
        except Exception:
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

            # Get connected relationships
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
                    ) in connections:  # Show all connections (limited to 5 in query)
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
        except Exception:
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

            # Get connected entity details
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
