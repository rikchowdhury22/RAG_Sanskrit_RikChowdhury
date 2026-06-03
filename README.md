# RAG Sanskrit — Retrieval-Augmented Generation for Sanskrit Documents

A fully CPU-based RAG pipeline for querying Sanskrit documents. Accepts queries in Sanskrit (Devanagari script), transliterated text, or English.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                    │
│                                                          │
│  .docx / .txt  →  Loader  →  Cleaner  →  Chunker        │
│                                              │           │
│                                         TF-IDF Index     │
│                                         (index.pkl)      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│                                                          │
│  User Query  →  TF-IDF Retriever  →  Top-K Chunks       │
│                                              │           │
│                                      Prompt Builder      │
│                                              │           │
│                                      Ollama LLM          │
│                                      (TinyLlama 1.1B)    │
│                                              │           │
│                                        Response          │
└──────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| Document Loader | `code/loader.py` | Loads `.docx` and `.txt` files, preserves Devanagari Unicode |
| Preprocessor | `code/loader.py` | Removes zero-width characters, normalizes whitespace |
| Chunker | `code/chunker.py` | Paragraph-aware sliding window chunker with configurable overlap |
| Retriever | `code/retriever.py` | TF-IDF with cosine similarity via numpy — no ML frameworks needed |
| Generator | `code/generator.py` | Ollama REST API client using TinyLlama 1.1B (CPU inference) |
| Ingestion Script | `code/ingest.py` | One-time pipeline: load → chunk → index → save |
| Query Interface | `code/query.py` | Interactive CLI and single-shot query mode |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG_Sanskrit
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download from [https://ollama.com/download](https://ollama.com/download), then:

```bash
# Pull the TinyLlama model (CPU-compatible, ~600MB)
ollama pull tinyllama

# Ollama runs as a background service automatically after installation
# Verify it's running:
curl http://localhost:11434/api/tags
```

### 4. Add your Sanskrit documents

Place `.docx` or `.txt` Sanskrit documents in the `data/` directory.
The repository already includes `data/sanskrit_corpus.docx` as the default corpus.

### 5. Build the index

```bash
python code/ingest.py
```

Expected output:
```
[Step 1/3] Loading documents...
[loader] Loaded: sanskrit_corpus.docx (9064 chars)
[Step 2/3] Chunking documents...
[chunker] sanskrit_corpus.docx: 26 chunks
[Step 3/3] Building TF-IDF index...
[retriever] Index built successfully.
✓ Ingestion complete. 26 chunks indexed.
```

---

## Usage

### Interactive mode

```bash
python code/query.py
```

Then type queries at the prompt:

```
Query> कालीदासः कः?
Query> Who is Kalidasa?
Query> What happened with the bell in the forest?
```

### Single query mode

```bash
python code/query.py --query "भोजराजा कः?" --top-k 3
```

### Show retrieved chunks

```bash
python code/query.py --query "शंखनादः कः" --show-chunks
```

### Full options

```
--query       Single query (non-interactive)
--top-k       Number of chunks to retrieve (default: 3)
--model       Ollama model name (default: tinyllama)
--index-path  Path to index.pkl (default: data/index.pkl)
--show-chunks Print retrieved chunk text alongside response
```

### Custom ingestion options

```bash
python code/ingest.py --chunk-size 300 --overlap 50 --data-dir ./data
```

---

## Project Structure

```
RAG_Sanskrit/
├── code/
│   ├── loader.py       # Document loading and preprocessing
│   ├── chunker.py      # Text chunking with overlap
│   ├── retriever.py    # TF-IDF retriever (numpy-based)
│   ├── generator.py    # LLM response generation via Ollama
│   ├── ingest.py       # Ingestion pipeline entry point
│   └── query.py        # Query interface (CLI)
├── data/
│   └── sanskrit_corpus.docx   # Source Sanskrit documents
├── report/
│   └── report.pdf      # Technical report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Design Decisions

### Why TF-IDF instead of dense embeddings?

Dense embedding models (e.g., sentence-transformers) are not trained on Sanskrit text and would produce poor semantic representations for Devanagari script. TF-IDF operates at the token level, treats Devanagari Unicode characters as first-class tokens, and produces reliable retrieval results without requiring any Sanskrit-specific model.

### Why Ollama + TinyLlama?

Ollama provides a clean REST API for local LLM inference that automatically uses CPU when no GPU is detected. TinyLlama (1.1B parameters) runs comfortably within 4GB RAM and produces coherent responses on CPU within 30–60 seconds.

### Why no persistent vector database?

The corpus is small enough (~26 chunks) that TF-IDF indexing completes in under 1 second. A persistent database (e.g., ChromaDB) adds setup complexity without meaningful performance benefit at this scale. The index is saved as a pickle file and loaded in milliseconds on subsequent runs.

---

## Known Limitations

- **Script mixing**: The corpus contains Devanagari mixed with English transliteration. The tokenizer handles both but does not normalize between equivalent forms (e.g., `kAlIdAsa` ≠ `कालीदास`).
- **LLM Sanskrit comprehension**: TinyLlama has limited Sanskrit training data. It performs better on the English portions of the corpus.
- **Corpus size**: Performance is proportional to corpus size. Results improve significantly with more documents.

---

## CPU Performance (observed on Ryzen 5000 series, 16GB RAM)

| Step | Time |
|------|------|
| Ingestion (26 chunks) | < 1 second |
| Retrieval (cosine similarity) | < 50ms |
| LLM generation (TinyLlama) | 30–90 seconds |
