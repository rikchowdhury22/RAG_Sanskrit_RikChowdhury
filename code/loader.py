"""
loader.py
---------
Loads and preprocesses Sanskrit documents from .docx or .txt files.
Handles Devanagari script and mixed Devanagari/Latin text as-is (no transliteration normalization).
"""

import os
import re
from docx import Document


def load_docx(filepath: str) -> str:
    """Extract full text from a .docx file."""
    doc = Document(filepath)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def load_txt(filepath: str) -> str:
    """Load plain text file with UTF-8 encoding (required for Devanagari)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_document(filepath: str) -> str:
    """Auto-detect file type and load accordingly."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        return load_docx(filepath)
    elif ext == ".txt":
        return load_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .docx and .txt are supported.")


def load_corpus(data_dir: str) -> list[dict]:
    """
    Load all .docx and .txt files from a directory.
    Returns a list of dicts: [{"source": filename, "text": content}, ...]
    """
    documents = []
    supported = (".docx", ".txt")

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(supported):
            fpath = os.path.join(data_dir, fname)
            try:
                text = load_document(fpath)
                documents.append({"source": fname, "text": text})
                print(f"[loader] Loaded: {fname} ({len(text)} chars)")
            except Exception as e:
                print(f"[loader] Warning: Could not load {fname}: {e}")

    if not documents:
        raise FileNotFoundError(f"No .docx or .txt files found in: {data_dir}")

    return documents


def clean_text(text: str) -> str:
    """
    Light preprocessing:
    - Normalize whitespace
    - Remove zero-width characters common in Unicode Devanagari text
    - Preserve Devanagari script and punctuation as-is
    """
    # Remove zero-width joiners / non-joiners
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # Normalize multiple spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
