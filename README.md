# ContextRAG

ContextRAG is a standalone, standard-library-only Python tool for enhanced web RAG (Retrieval-Augmented Generation). It is designed for exact fact retrieval with improved extraction and ranking, focusing on verifiable data points over general text.

## Features

*   **No External Dependencies:** Runs with standard Python 3.7+ libraries.
*   **Enhanced Extraction:** Extracts structured data (infoboxes) and text from HTML.
*   **Data-Focused Snippetization:** Prioritizes sentences containing measurements, dates, and verifiable facts.
*   **Smart Search:** BM25 search with boosting for exact phrases and numeric content.
*   **Privacy-Friendly:** Operates locally, no databases required.

## Installation

Simply copy `contextrag.py` to your project or path.

```bash
chmod +x contextrag.py
```

## Usage

ContextRAG operates via subcommands.

### 1. Resolve URLs (Optional)
Find URLs for a query.

```bash
./contextrag.py resolve "James Webb Telescope launch date"
```

### 2. Ingest Data
Fetch, extract, and snippetize content from a URL or query in one go.

```bash
./contextrag.py ingest "James Webb Telescope" --top 1
```

This will create a JSONL file in the `data/` directory (e.g., `data/en.wikipedia.org.jsonl`).

### 3. Compose Evidence
Search your ingested data and compose a prompt for an LLM.

```bash
./contextrag.py compose "What is the mass of the sun?" --time-scope present
```

### 4. Ask (Pipe to LLM)
Pipe the composed context to a local LLM runner (e.g., Ollama).

```bash
./contextrag.py compose "Mass of the sun" --out prompt.txt
./contextrag.py ask --runner "ollama run llama3" --in prompt.txt
```

## License

MIT
