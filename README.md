# 📄 Table-Aware Financial RAG System  
### Microsoft Annual Report – End-to-End Case Study

## Overview

This project is an end-to-end **Retrieval-Augmented Generation (RAG)** system designed to answer complex, numeric, and multi-part financial questions from long, structured documents such as **annual reports**.

Unlike naive RAG pipelines, this system explicitly handles:
- Tables vs narrative text
- Numeric and fiscal-year–sensitive queries
- Multi-part questions (e.g., revenue *and* operating income)
- Hallucination avoidance
- Retrieval explainability and debugging

The **Microsoft Annual Report** is used as a realistic stress test for retrieval quality, especially around financial statements.

---

## Why This Project Exists

Most RAG tutorials:
- Assume short, unstructured documents
- Ignore tables
- Rely solely on vector similarity
- Hallucinate missing numeric values

Real enterprise documents are:
- Long and repetitive
- Table-heavy
- Numerically sensitive
- Spread across multiple sections

This project explores **how to build a correct, production-oriented RAG system** that prioritizes **accuracy, recall, and explainability** over surface-level fluency.

---

## Key Features

- 📊 **Table-aware chunking**
  - Tables and text are extracted and indexed differently
  - Tables are preserved as atomic chunks (not arbitrarily split)

- 🔍 **Hybrid retrieval (BM25 + Vector MMR)**
  - Lexical + semantic search combined via a custom ensemble retriever

- 🔄 **Maximal Marginal Relevance (MMR)**
  - Ensures diversity across retrieved chunks
  - Prevents repetitive context from crowding out key metrics

- 🧠 **Strict hallucination control**
  - Answers are generated **only from retrieved context**
  - Partial answers are allowed
  - Missing facts are explicitly reported as `NOT FOUND`

- 🧾 **Metadata-driven logic**
  - Page numbers, table IDs, and chunk types drive:
    - Deduplication
    - Table prioritization
    - Debugging and citations

- 🔍 **Observability-first design**
  - Tracing via LangSmith
  - Retrieval inspection and context hashing

- 🚀 **API-ready architecture**
  - Offline ingestion
  - Fast online querying via FastAPI

---

## High-Level Architecture

```text
PDF
 └─ Table-aware extraction
     ├─ Text chunks
     └─ Table chunks
         └─ Text splitting (text only)
             └─ Embeddings
                 └─ Vector index (Chroma, persisted)

Lexical index
 └─ BM25 (persisted)

Query pipeline
 ├─ Query rewriting
 ├─ BM25 retrieval
 ├─ Vector retrieval (MMR)
 ├─ Deduplication
 ├─ Metadata-aware ranking
 └─ Answer generation (LLM)


