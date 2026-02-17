# Graph RAG Agent Network

## Overview

The **Graph RAG** registry (`graph_rag.hocon`) implements a Graph-based Retrieval-Augmented Generation (RAG) agent network for answering questions about UNFCCC climate change documents. Unlike the other registries in this project (`paris_agreement`, `kyoto_protocol`, `cop`, `unfccc`) which use hierarchical AAOSA agents with `txt_loader` toolbox tools to load raw document files, Graph RAG uses a **knowledge graph** backed by [Neo4j](https://neo4j.com/) or [FalkorDB](https://www.falkordb.com/) and powered by the [Graphiti](https://github.com/getzep/graphiti) library for hybrid (vector + keyword) search.

This approach enables:
- **Cross-document relationship traversal** (e.g., "which decision first created X?" traces backward references)
- **Temporal reasoning** across sessions and years
- **Conference-type filtering** (CMA, COP/CP, CMP, SBI, SBSTA)
- **Paragraph-level retrieval** from indexed metadata
- **Founding vs. follow-up decision classification** to answer chronological governance questions

## Architecture

```
User Query
    |
    v
+-------------------+
|     GraphRAG      |  <-- Entry-point coordinator agent (gpt-4o)
| (LLM orchestrator)|
+-------------------+
    |
    v
+-------------------+
| graph_rag_expert  |  <-- Coded tool (Python class)
| GraphSearchTool   |
+-------------------+
    |
    v
+-------------------+
|    Graphiti Core   |  <-- Hybrid search engine (vector + keyword)
+-------------------+
    |
    +-------+-------+
    |               |
    v               v
+---------+   +-----------+
|  Neo4j  |   | FalkorDB  |   <-- Graph database backends (auto-detected)
+---------+   +-----------+
        |
        v
+---------------------------+
| UNFCCC Knowledge Graph    |
| (COP, CMA, CMP, SBI,     |
|  SBSTA decisions 2014-24) |
+---------------------------+
```

### Agent Network Structure

| Agent | Type | Role |
|-------|------|------|
| `GraphRAG` | LLM agent (entry point) | Receives user queries, orchestrates search strategy, synthesizes answers with strict anti-hallucination rules |
| `graph_rag_expert` | Coded tool (`GraphSearchTool`) | Executes multi-stage graph searches, analyzes queries, filters by conference type, classifies founding/follow-up decisions |

### Comparison with Other Registries

| Feature | `paris_agreement`, `cop`, `kyoto_protocol` | `graph_rag` |
|---------|----------------------------------------------|-------------|
| LLM model | `gpt-4.1` | `gpt-4o` |
| Orchestration | AAOSA hierarchical sub-agents | Single coordinator + coded tool |
| Document access | `txt_loader` toolbox (reads raw `.txt` files) | Knowledge graph (pre-indexed entities, relationships, episodes) |
| Cross-document queries | Requires querying multiple sub-agents | Built-in relationship traversal and backward reference following |
| Temporal reasoning | Manual (agent-level session awareness) | Automatic (temporal reranking, founding/follow-up classification) |
| Conference filtering | Implicit (separate registries per conference) | Explicit (single registry filters by CMA/COP/CMP/SBI/SBSTA) |

## Configuration

### Registry File

**File:** `registries/graph_rag.hocon`

The registry is enabled in `registries/manifest.hocon`:
```hocon
{
    "graph_rag.hocon": true
}
```

### Environment Variables

The coded tools load configuration from `coded_tools/graph_rag/.env`. The following variables control behavior:

#### Database Connection (one required)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | _(none)_ | Neo4j connection URI (e.g., `neo4j+s://...`). Takes priority over FalkorDB. |
| `NEO4J_USER` | _(none)_ | Neo4j username |
| `NEO4J_PASSWORD` | _(none)_ | Neo4j password |
| `FALKORDB_HOST` | `localhost` | FalkorDB hostname. Used if `NEO4J_URI` is not set. |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `FALKORDB_USERNAME` | _(none)_ | FalkorDB username (optional) |
| `FALKORDB_PASSWORD` | _(none)_ | FalkorDB password (optional) |
| `GRAPH_NAME` | `unfccc_knowledge_graph` | FalkorDB graph name |

**Detection priority:** If `NEO4J_URI` is set, Neo4j is used. Otherwise, if `FALKORDB_HOST` is set, FalkorDB is used.

#### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider for Graphiti operations |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `LLM_API_KEY` | `OPENAI_API_KEY` | LLM API key environment variable name |
| `LLM_CHOICE` | `gpt-4.1-mini` | Model used for graph search operations |
| `INGESTION_LLM_CHOICE` | `gpt-4.1-nano` | Model used during document ingestion (faster/cheaper) |

#### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Embedding API base URL |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | Embedding API key environment variable name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimension |

#### Ingestion Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `documents` | Directory containing UNFCCC `.txt` source files |
| `BATCH_SIZE` | `10` | Number of episodes to ingest per batch |
| `MAX_EPISODES` | `0` | Maximum episodes to ingest (`0` = unlimited) |
| `CHUNK_SIZE` | `800` | Characters per chunk for document splitting |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_CHUNK_SIZE` | `1500` | Maximum characters per chunk |
| `ENABLE_DEMO_SEARCHES` | `false` | Run demo searches after ingestion |
| `VERBOSE_LOGGING` | `false` | Enable detailed logging during ingestion |

## Coded Tools

All coded tools are located in `coded_tools/graph_rag/`.

### GraphSearchTool

**File:** `coded_tools/graph_rag/graph_search_tool.py`
**Class:** `graph_rag.graph_search_tool.GraphSearchTool`
**Extends:** `CodedTool` (neuro-san)

The primary search tool used by the `graph_rag_expert` agent. Implements a multi-stage search pipeline with query analysis, conference filtering, and automatic founding/follow-up classification.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `string` | Yes | - | The search query. The tool has built-in query analysis, so the full user question should be passed. |
| `limit` | `string` | No | `10` | Maximum number of results to return |
| `search_type` | `enum` | No | `general` | Type of search (see below) |
| `query_complexity` | `enum` | No | `medium` | Search depth level (see below) |

**Search types:**

| Value | Description |
|-------|-------------|
| `general` | Searches for general facts and relationships across the knowledge graph |
| `entity` | Searches for entity nodes (countries, organizations, concepts, mechanisms) |
| `relationship` | Searches for relationship edges between entities |
| `episode` | Searches for primary source document content (episodes with exact wording) |

**Query complexity levels:**

| Value | Results | Use case |
|-------|---------|----------|
| `direct` | 1 | Specific decision, paragraph, or single fact lookup |
| `medium` | 5 | Concept definitions, moderate topics |
| `extensive` | 15 | Broad multi-faceted topics, governance evolution, timelines |

#### Query Analysis

The tool automatically analyzes each query to detect:

- **Intent:** `factual`, `requirement`, `definition`, or `relationship`
- **UNFCCC key concepts:** mitigation, adaptation, NDC, transparency, finance, loss and damage, technology transfer, capacity building, Article 6, global stocktake
- **Temporal markers:** years, conference references (CMA 3, COP 26, etc.)
- **Conference filter:** CMA (Paris Agreement), COP/CP (Convention), CMP (Kyoto Protocol), SBI, SBSTA
- **Decision references:** e.g., "Decision 3/CMA.1"
- **Paragraph references:** e.g., "para 123"
- **Temporal direction:** `earliest` (for "first/originally created") or `latest`
- **Timeline queries:** detected when asking about "all decisions", "evolution", "governance"
- **Identification queries:** when a user describes a decision by content rather than number

#### Multi-Stage Search Pipeline

The search executes in up to 4 stages:

```
Stage 1: Entity Search
    Find key concepts and entities matching the query
        |
        v
Stage 2: Episode Search
    Find primary source documents (with conference filtering)
        |
        +-- Stage 2B: Backward Reference Following
        |   Trace "Recalling decision X" links to find founding decisions
        |
        +-- Stage 2C: Topic Timeline Search
        |   Broad search for governance evolution queries
        |
        +-- Stage 2D: Founding/Follow-up Classification
            Classify decisions as creating vs. developing
        |
        v
Stage 3: Relationship Search
    Find edges connecting entities
        |
        v
Stage 4: Output Assembly
    Priority-ordered output with truncation safety (40,000 char max)
```

#### Output Structure

Results are assembled in priority order:

1. **SPECIFIC DECISION** - Exact match by decision ID (e.g., "Decision 3/CMA.1")
2. **SPECIFIC PARAGRAPHS** - Extracted paragraph content from metadata
3. **FOUNDING DECISION(S) IDENTIFIED** - Decisions with creation language (ESTABLISHES, CREATES, LAUNCHES)
4. **FOLLOW-UP DECISIONS** - Decisions that further develop/operationalize
5. **EARLIER REFERENCED DECISIONS** - Decisions traced via backward reference following
6. **COMPLETE DECISION TIMELINE** - Chronological list of all related decisions
7. **PRIMARY SOURCE DOCUMENTS** - Episode content from the knowledge graph
8. **RELEVANT ENTITIES AND CONCEPTS** - Entity nodes
9. **RELATIONSHIPS AND CONNECTIONS** - Relationship edges

#### Singleton Pattern

`GraphSearchTool` uses class-level singletons for the Graphiti instance and database driver. This means:
- The first invocation initializes the connection (which may take several seconds)
- Subsequent invocations reuse the existing connection
- The connection is shared across all agent invocations within the same process

### FalkorDBIngestionEnhanced

**File:** `coded_tools/graph_rag/falkordb_ingestion.py`
**Class:** `graph_rag.falkordb_ingestion.FalkorDBIngestionEnhanced`
**Extends:** `CodedTool` (neuro-san)

Ingests UNFCCC climate decision documents from `.txt` files into FalkorDB using the Graphiti library. This is a **one-time data preparation step**, not part of the query-time agent network.

#### Document Parsing

The ingestion tool parses UNFCCC text files with the following structure:

- **Primary sections:** Decisions and Resolutions (split by `Decision X/Y.Z` or `Resolution X/Y.Z` headers)
- **Annexes:** Separated into distinct episodes (detected by `Annex` headers)
- **Paragraphs:** Numbered paragraphs are extracted and stored in metadata for granular retrieval

#### Episode Metadata

Each ingested episode contains:

| Field | Description |
|-------|-------------|
| `name` | Episode title (e.g., "Decision 3/CMA.1 - Matters relating to the implementation") |
| `episode_body` | Full text content of the decision/annex section |
| `source_description` | Human-readable source attribution |
| `reference_time` | ISO timestamp derived from conference year |
| `metadata.conference_type` | CMA, CMP, COP, SBI, or SBSTA |
| `metadata.year` | Conference year |
| `metadata.session` | Session number |
| `metadata.location` | Conference location |
| `metadata.fccc_reference` | FCCC document symbol (e.g., FCCC/PA/CMA/2018/3/Add.1) |
| `metadata.decision_id` | Decision identifier (e.g., "3/CMA.1") |
| `metadata.annex_id` | Annex identifier if applicable |
| `metadata.paragraph_index` | JSON index mapping paragraph numbers to character offsets |

#### Checkpoint Resume

Ingestion supports checkpoint-based resumption. A checkpoint file (`.ingestion_checkpoint.txt`) tracks which episodes have been successfully added, allowing the process to resume after interruption without re-ingesting completed episodes.

#### FalkorDB Monkey-Patches

The ingestion module applies patches to three FalkorDB/Graphiti functions to prevent query timeout issues with large knowledge graphs:

- `edge_search_filter_query_constructor` - Fixes edge search filter queries
- `edge_fulltext_search` - Fixes full-text search on edges
- `FalkorDriver.execute_query` - Adds timeout handling for graph queries

### Neo4jIngestionEnhanced

**File:** `coded_tools/graph_rag/neo4j_ingestion.py`
**Class:** `graph_rag.neo4j_ingestion.Neo4jIngestionEnhanced`
**Extends:** `CodedTool` (neuro-san)

Functionally identical to `FalkorDBIngestionEnhanced` but targets Neo4j instead of FalkorDB. Uses the same document parsing, episode structure, and metadata extraction. Does not require the FalkorDB-specific monkey patches.

### Constrained Prompts

**File:** `coded_tools/graph_rag/constrained_prompts.py`

Provides modified Graphiti prompt functions used **during ingestion** to prevent hallucination when building the knowledge graph. These replace Graphiti's default entity and relationship extraction prompts.

#### `extract_text_constrained(context)`

Modified entity extraction prompt. Key constraints:

- Only extracts entities **explicitly mentioned** in the text
- **Prohibits** extracting: conference body names (COP, CMA, CMP, SBI, SBSTA), session numbers, document headers
- **Extracts:** structural references (Decision IDs, Articles, Annexes), named entities (countries, funds, organizations), climate concepts (mitigation, adaptation, NDC, finance, etc.)
- **Annex separation rule:** Annex entities belong to the annex episode, not the parent decision

#### `extract_edges_constrained(context)`

Modified relationship/fact extraction prompt. Key constraints:

- Only extracts relationships **explicitly stated** in the text
- Categorizes relationships: structural, topical, financial, governance
- Distinguishes **FOUNDING relationships** (ESTABLISHES, CREATES, LAUNCHES) from **FOLLOW-UP relationships** (FURTHER_DEVELOPS, EXPANDS, OPERATIONALIZES, REVIEWS)
- This distinction is critical for the `GraphSearchTool` to answer "which decision FIRST created X" questions

## Coordinator Agent Behavior

The `GraphRAG` coordinator agent (the entry point) implements a detailed orchestration strategy defined in its HOCON instructions. Key behaviors:

### 5-Stage Workflow

1. **Query Analysis** - Restate the objective, identify entities/years/decision IDs/conference types
2. **Knowledge Graph Exploration** - Always call `graph_rag_expert` first; treat the graph as the map of available information
3. **Synthesis** - Merge graph results into a cohesive narrative with strict per-decision attribution
4. **References** - Summarize prior decisions mentioned; call sub-agents for additional detail
5. **Completeness Check** - For multi-stage governance questions, verify all phases are covered

### Three Response Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Specific Extraction** | User asks about a specific detail (e.g., "What does para 123 say?") | Brief, quote-only answer with exact text |
| **Comprehensive Summary** | User asks what a decision is about (e.g., "Summarize Decision X") | Thorough structured list of all operative paragraphs |
| **Decision Identification** | User describes a decision by content (e.g., "Which CMA decision created a network?") | Search broadly, identify all matching decisions with IDs and summaries |

### Anti-Hallucination Rules

The coordinator enforces strict anti-hallucination rules (highest priority):

1. **Never invent decision numbers** - Only cite decision IDs that appear verbatim in search results
2. **Never mix content across decisions** - Each decision's content is attributed only to that decision
3. **Never attribute annex content to parent decisions** - Annex content is treated as separate
4. **Every claim needs a source** - Must cite decision ID and paragraph number from search results
5. **If unsure, say so** - Acknowledge gaps rather than filling them with outside knowledge

### Conference Type Filtering

| Abbreviation | Full Name | Decision Format |
|-------------|-----------|-----------------|
| CMA | Conference of the Parties serving as the meeting of the Parties to the Paris Agreement | Decision X/CMA.Y |
| COP / CP | Conference of the Parties | Decision X/CP.Y |
| CMP | Conference of the Parties serving as the meeting of the Parties to the Kyoto Protocol | Decision X/CMP.Y |
| SBI | Subsidiary Body for Implementation | _(reports, not decisions)_ |
| SBSTA | Subsidiary Body for Scientific and Technological Advice | _(reports, not decisions)_ |

These are **completely separate** decision tracks. Decision 6/CMA.5 is not the same as Decision 6/CP.25.

## Document Corpus

The knowledge graph is built from UNFCCC decision documents covering:

| Conference | Sessions | Years | Document Count |
|------------|----------|-------|----------------|
| CMA (Paris Agreement) | 1-6 | 2016-2024 | 18 files |
| CMP (Kyoto Protocol) | 10-19 | 2014-2024 | 10 files |
| COP (Conference of the Parties) | 20-29 | 2014-2024 | 22 files |
| SBI (Implementation) | 40-61 | 2014-2024 | 13 files |
| SBSTA (Scientific/Tech Advice) | 40-61 | 2014-2024 | 17 files |

Source documents are stored in `documents/` as plain text files extracted from official UNFCCC PDF reports.

## Setup and Usage

### Prerequisites

- Python 3.12 or 3.13
- A running Neo4j or FalkorDB instance with the UNFCCC knowledge graph already ingested
- OpenAI API key (for both the agent LLM and Graphiti's embedding/search operations)

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Additional packages required for graph_rag (install separately):
```bash
pip install graphiti-core neo4j  # For Neo4j backend
# or
pip install graphiti-core falkordb  # For FalkorDB backend
```

### 2. Configure Environment

Create or edit `coded_tools/graph_rag/.env` with your database credentials and API keys. See the [Environment Variables](#environment-variables) section above for all available options.

### 3. Ingest Documents (One-Time)

Before querying, the knowledge graph must be populated. This is done by running the ingestion tool, which can be invoked as a `CodedTool` through the neuro-san framework or directly.

The ingestion process:
1. Connects to the configured database (Neo4j or FalkorDB)
2. Parses all `.txt` files in the `DATA_DIR` directory
3. Splits documents into episodes (decisions, annexes, paragraphs)
4. Uses Graphiti with [constrained prompts](#constrained-prompts) to extract entities and relationships
5. Adds episodes to the knowledge graph with full metadata

Ingestion supports **checkpoint resume** - if interrupted, it will skip already-ingested episodes on restart.

### 4. Run the Agent Network

```bash
python -m run
```

Navigate to `http://localhost:4173/` and select the **graph_rag** agent network. Then ask questions about UNFCCC climate documents.

### Sample Queries

- "What does Decision 3/CMA.1 say about the Paris Agreement?"
- "Which CMA decision first created the Santiago network for loss and damage?"
- "How has the transparency framework evolved across CMA sessions?"
- "What are the main differences between the Sharm el-Sheikh decision and the Baku decision on Article 6.2?"
- "List all CMA decisions about climate finance"
- "What does paragraph 43 of Decision 2/CMA.2 establish?"

## File Reference

```
coded_tools/graph_rag/
    __init__.py                  # Package init, exports GraphSearchTool
    graph_search_tool.py         # Main search tool (~2050 lines)
    falkordb_ingestion.py        # FalkorDB ingestion tool (~1790 lines)
    neo4j_ingestion.py           # Neo4j ingestion tool (~1645 lines)
    constrained_prompts.py       # Anti-hallucination prompts for Graphiti (~394 lines)
    .env                         # Environment configuration (not committed)
    .ingestion_checkpoint.txt    # Ingestion resume checkpoint (auto-generated)

registries/
    graph_rag.hocon              # Agent network registry configuration
    manifest.hocon               # Registry manifest (enables graph_rag)
    aaosa_basic.hocon            # Shared AAOSA substitutions (included by graph_rag)
```
