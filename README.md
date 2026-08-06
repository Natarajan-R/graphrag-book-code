# GraphRAG — Companion Source Code

This repository contains the complete, runnable source code for the book
**_GraphRAG: Building an Intelligent Research Assistant with Knowledge Graphs_**
by Natarajan.

📖 **Get it on Amazon Kindle:** [US](https://www.amazon.com/dp/B0H3QXVSY4) · [India](https://www.amazon.in/dp/B0H3QXVSY4)

The code is organized **by chapter** — each folder holds the source files that
that chapter teaches. Some files (e.g. `hybrid_retriever.py`, `answer_generator.py`)
are reused and intentionally appear in more than one chapter folder so that each
chapter can be run on its own.

## Repository layout

| Folder | Book chapter | What it contains |
|--------|--------------|------------------|
| `chapter03/` | Ch 3 — Setting Up Your AI Laboratory | Environment validation scripts (`test_neo4j.py`, `test_ollama_*.py`, `health_check.sh`, …) |
| `chapter04/` | Ch 4 — Building the Ingestion Pipeline | The staged pipeline `a00_…`–`a05_…` plus `fix_extraction.py` |
| `chapter06/` | Ch 6 — Building the Intelligent Query System | `vector_search.py`, `graph_traversal.py`, `hybrid_retriever.py`, `answer_generator.py`, `graphrag_query.py` |
| `chapter07/` | Ch 7 — The Graph-First Approach | `graphrag_query_graph_first.py` (+ reused helpers) |
| `chapter08/` | Ch 8 — Handling Contradictions | `versioned_graph_writer.py`, `contradiction_aware_query.py`, `test_contradictions.py` |
| `chapter09/` | Ch 9 — Building a Web Interface | Flask API + web app, `templates/index.html`, `Dockerfile` |
| `chapter10/` | Ch 10 — Scaling to Production | The production system: `main.py`, `config.py`, `utils.py`, and the modular pipeline |

> Chapters 1, 2, 5, and 11 contain no source code — Chapter 5 is a hands-on
> tour of Neo4j Browser using Cypher queries printed in the book.

## Prerequisites

This project runs **100% locally** — no cloud APIs, no subscriptions.

- **Python 3.10+** (3.11 recommended)
- **Neo4j Community Edition 5.x** — running at `bolt://localhost:7687`
- **Ollama** — running at `http://localhost:11434`, with two models pulled:
  ```bash
  ollama pull qwen2.5      # entity extraction + answer generation
  ollama pull nomic-embed-text   # embeddings
  ```

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> graphrag-book-code
cd graphrag-book-code

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies (each code chapter ships a requirements.txt)
pip install -r chapter04/requirements.txt   # or the chapter you're working in

# 4. Configure your environment
cp .env.example .env            # then edit .env with your Neo4j password etc.
```

## Running the code

Work through the folders in the same order as the book. For example, the
Chapter 4 ingestion pipeline:

```bash
cd chapter04
python a05_process_document.py your_document.pdf
```

See each chapter in the book for the full explanation of every script.

---

## More books by the author

Each one is a hands-on build with its code in the open.

| Book | Amazon | Code |
|---|---|---|
| **Enterprise AI Workflow Automation: Building Resilient Agentic Systems** | [US](https://www.amazon.com/dp/B0HCZC7VCC) · [IN](https://www.amazon.in/dp/B0HCZC7VCC) | [auto-sre-graph](https://github.com/Natarajan-R/auto-sre-graph) |
| **Building a Local AI Coding Agent** | [US](https://www.amazon.com/dp/B0H8B6QXXX) · [IN](https://www.amazon.in/dp/B0H8B6QXXX) | [local-ai-coding-agent](https://github.com/Natarajan-R/local-ai-coding-agent) |
| **Agentic AI — A Hands-On Guide** | [US](https://www.amazon.com/dp/B0H6R7SZZB) · [IN](https://www.amazon.in/dp/B0H6R7SZZB) | [agentic-ai-book](https://github.com/Natarajan-R/agentic-ai-book) |

All titles → [Amazon author page](https://www.amazon.com/stores/author/B0H3T2MG83)

---

## License

See [LICENSE](LICENSE).
