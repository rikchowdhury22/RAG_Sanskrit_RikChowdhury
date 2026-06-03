"""
query.py
--------
Interactive query interface for the Sanskrit RAG system.
Accepts queries in Sanskrit (Devanagari), transliterated text, or English.

Usage:
    python query.py
    python query.py --query "कालीदासः कः?" --top-k 3
    python query.py --model tinyllama --top-k 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from retriever import TFIDFRetriever
from generator import generate_response


DEFAULT_INDEX = os.path.join(os.path.dirname(__file__), "..", "data", "index.pkl")
DEFAULT_MODEL = "tinyllama"
DEFAULT_TOP_K = 3


def run_query(query: str, retriever: TFIDFRetriever, model: str, top_k: int) -> dict:
    """
    Execute a single RAG query end-to-end.

    Args:
        query: User input
        retriever: Loaded TFIDFRetriever instance
        model: Ollama model name
        top_k: Number of chunks to retrieve

    Returns:
        Dict with query, retrieved_chunks, and response
    """
    print(f"\n[Query] {query}")
    print("-" * 40)

    # Retrieve
    print(f"[Retriever] Searching top-{top_k} chunks...")
    chunks = retriever.retrieve(query, top_k=top_k)

    if not chunks or all(c["score"] == 0 for c in chunks):
        print("[Retriever] No relevant chunks found.")
        return {
            "query": query,
            "retrieved_chunks": [],
            "response": "No relevant information found in the corpus for this query."
        }

    print(f"[Retriever] Top result score: {chunks[0]['score']:.4f} | Source: {chunks[0]['source']}")

    # Generate
    print("[Generator] Generating response via LLM...")
    response = generate_response(query, chunks, model=model)

    return {
        "query": query,
        "retrieved_chunks": chunks,
        "response": response
    }


def interactive_mode(retriever: TFIDFRetriever, model: str, top_k: int):
    """Run an interactive query loop."""
    print("\n" + "=" * 50)
    print("Sanskrit RAG System — Interactive Mode")
    print("Type your query in Sanskrit (Devanagari/transliterated) or English.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if not query:
            continue

        result = run_query(query, retriever, model, top_k)

        print("\n[Response]")
        print(result["response"])
        print("\n[Retrieved Sources]")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"  {i}. {chunk['source']} (score: {chunk['score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Query the Sanskrit RAG system")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single query to run (non-interactive mode)")
    parser.add_argument("--index-path", default=DEFAULT_INDEX,
                        help="Path to the saved index.pkl")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print retrieved chunk text in output")
    args = parser.parse_args()

    index_path = os.path.abspath(args.index_path)

    if not os.path.exists(index_path):
        print(f"[Error] Index not found at: {index_path}")
        print("Run `python ingest.py` first to build the index.")
        sys.exit(1)

    # Load index
    retriever = TFIDFRetriever()
    retriever.load_index(index_path)

    if args.query:
        # Single-shot mode
        result = run_query(args.query, retriever, args.model, args.top_k)
        print("\n[Response]")
        print(result["response"])

        if args.show_chunks:
            print("\n[Retrieved Chunks]")
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                print(f"\n--- Chunk {i} (score: {chunk['score']:.4f}) ---")
                print(chunk["text"])
    else:
        # Interactive mode
        interactive_mode(retriever, args.model, args.top_k)


if __name__ == "__main__":
    main()
