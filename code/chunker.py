"""
chunker.py
----------
Splits documents into overlapping chunks for retrieval.
Uses a simple character-based sliding window approach that respects
sentence/paragraph boundaries — safe for Devanagari and mixed-script text.
"""

import re


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.
    Tries to split on paragraph or sentence boundaries where possible.

    Args:
        text: Input text (Devanagari, Latin, or mixed)
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    # Split on paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph keeps us under chunk_size, add it
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)

            # If paragraph itself is larger than chunk_size, split it further
            if len(para) > chunk_size:
                sub_chunks = _split_large_paragraph(para, chunk_size, overlap)
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                # Start new chunk with overlap from previous
                if chunks:
                    overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_large_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a single large paragraph by character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Chunk a list of loaded documents.

    Args:
        documents: List of {"source": str, "text": str} dicts from loader
        chunk_size: Target chunk character size
        overlap: Overlap between chunks

    Returns:
        List of {"chunk_id": str, "source": str, "text": str} dicts
    """
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}__chunk_{i}",
                "source": doc["source"],
                "text": chunk
            })
        print(f"[chunker] {doc['source']}: {len(chunks)} chunks")

    print(f"[chunker] Total chunks: {len(all_chunks)}")
    return all_chunks
