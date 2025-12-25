# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-Graph-RAG is a naive implementation of Graph-based Retrieval Augmented Generation (RAG) without using external GraphRAG packages. The goal is to extract knowledge graphs from documents (papers, novels, personal statements) and enable LLM-powered retrieval using OpenAI API.

## Development Setup

This project uses **uv** for package management:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run the application
uv run python main.py

# Run tests
uv run pytest
```

## Architecture

The planned architecture follows this pipeline:

1. **Document Input** → Text Chunking
2. **Entity & Relationship Extraction** (OpenAI API)
3. **Knowledge Graph Construction**
4. **Graph Indexing & Storage**
5. **Query Processing** → Graph-based Retrieval
6. **LLM Response Generation** (OpenAI API)

## Planned Module Structure

```
mini_graph_rag/
├── chunking/          # Text chunking utilities with context overlap
├── extraction/        # Entity and relationship extraction via OpenAI API
├── graph/             # Knowledge graph construction and storage
├── retrieval/         # Graph traversal and subgraph ranking
└── llm/               # OpenAI API integration layer
```

## Key Implementation Principles

- **Naive Implementation**: Keep it simple and educational, avoid over-engineering
- **No External GraphRAG Packages**: Build from scratch using only OpenAI API for LLM capabilities
- **OpenAI API Usage**: Used for both entity/relationship extraction and response generation
- **Multi-document Support**: Handle papers, novels, personal statements with different characteristics

## Environment Configuration

Set OpenAI API key before running:
```bash
export OPENAI_API_KEY='your-api-key-here'
```
