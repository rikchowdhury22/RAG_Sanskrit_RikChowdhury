"""
ingest.py
---------
One-time ingestion script. Run this before querying.

Steps:
1. Load all .docx / .txt files from ../data/
2. Clean and chunk the text
3. Build TF-IDF index
4. Save index to ../data/index.pkl

Usage:
    python ingest.py
    python ingest.py --data-dir ../data --chunk-size 500 --overlap 100
"""

import argparse
import os
import sys

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_corpus, clean_text
from chunker import chunk_documents
from retriever import TFIDFRetriever


def main():
    parser = argparse.ArgumentParser(description="Ingest Sanskrit documents into RAG index")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"),
                        help="Directory containing .docx or .txt Sanskrit documents")
    parser.add_argument("--index-path", default=os.path.join(os.path.dirname(__file__), "..", "data", "index.pkl"),
                        help="Output path for the saved index")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target chunk size in characters (default: 500)")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Overlap between chunks in characters (default: 100)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    index_path = os.path.abspath(args.index_path)

    print("=" * 50)
    print("RAG Sanskrit — Ingestion Pipeline")
    print("=" * 50)
    print(f"Data directory : {data_dir}")
    print(f"Chunk size     : {args.chunk_size} chars")
    print(f"Overlap        : {args.overlap} chars")
    print(f"Index output   : {index_path}")
    print()

    # Step 1: Load documents
    print("[Step 1/3] Loading documents...")
    documents = load_corpus(data_dir)
    for doc in documents:
        doc["text"] = clean_text(doc["text"])

    # Step 2: Chunk
    print("\n[Step 2/3] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)

    # Step 3: Build and save index
    print("\n[Step 3/3] Building TF-IDF index...")
    retriever = TFIDFRetriever()
    retriever.build_index(chunks)
    retriever.save_index(index_path)

    print("\n✓ Ingestion complete.")
    print(f"  Documents loaded : {len(documents)}")
    print(f"  Total chunks     : {len(chunks)}")
    print(f"  Index saved to   : {index_path}")
    print("\nRun query.py to start querying.")


if __name__ == "__main__":
    main()
