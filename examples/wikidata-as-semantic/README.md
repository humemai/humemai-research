# Wikidata as Semantic Memory

This directory integrates the `wikidata` submodule as a foundation for semantic memory management in HumemAI. It simplifies and processes Wikidata JSON dumps into manageable graph structures, which can then be imported into JanusGraph while adhering to HumemAI's memory design.

## Why Use the Wikidata Submodule?

The `wikidata` submodule (available at `git@github.com:humemai/wikidata.git`) parses the large and complex Wikidata dumps into simplified JSON batches. These processed files can then be imported into JanusGraph, enabling the creation of a property graph database representing semantic memory.

Semantic memory stores knowledge about the world in a structured format and is critical for long-term knowledge retention in AI agents. Each semantic memory entry includes the `known_since` property, indicating when the knowledge was acquired.

## How Semantic Memory Differs from Episodic Memory

- **Semantic Memory**:
  - Focuses on general world knowledge.
  - Includes a `known_since` property when the information was added.
  - Derived from external knowledge bases like Wikidata.
- **Episodic Memory**:
  - Represents personal experiences or events the agent has "experienced."
  - Includes an `event_time` property, which emphasizes the temporal nature of the memory.

By maintaining both types of memories, HumemAI ensures a comprehensive memory architecture that balances general knowledge with agent-specific experiences.

## Initializing the Submodule

If you have already cloned the `humemai` repository but the `wikidata` submodule is not initialized, follow these steps:

1. Navigate to the `humemai` repository:

   ```bash
   cd /path/to/humemai
   ```

2. Initialize and update the submodule:

   ```bash
   git submodule update --init --recursive
   ```

3. Verify the `wikidata` submodule under `examples/wikidata-as-semantic/wikidata`:

   ```bash
   ls examples/wikidata-as-semantic/wikidata
   ```

This will ensure that the `wikidata` submodule is ready for use.

## How to Use the Wikidata Submodule

Read the [README.md](./wikidata/README.md) inside the submodule.
