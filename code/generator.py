"""
generator.py
------------
LLM-based response generator using Ollama's local REST API.
Ollama runs models on CPU when no GPU is available — satisfies the CPU-only requirement.

Model used: tinyllama (1.1B params) — fast on CPU, fits in RAM easily.
Falls back to a context-echo response if Ollama is unavailable.
"""

import json
import requests


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tinyllama"


def _check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _pull_model_if_needed(model: str) -> None:
    """Pull model from Ollama registry if not already present."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(model in m for m in models):
            print(f"[generator] Model '{model}' not found locally. Pulling... (this may take a while)")
            pull_r = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model},
                timeout=300,
                stream=True
            )
            for line in pull_r.iter_lines():
                if line:
                    status = json.loads(line).get("status", "")
                    if status:
                        print(f"[generator] {status}")
    except Exception as e:
        print(f"[generator] Warning: Could not verify/pull model: {e}")


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build a RAG prompt from the query and retrieved context chunks.

    Prompt structure follows the standard RAG pattern:
    - System instruction: answer only from context
    - Context: retrieved chunks concatenated
    - Query: user's question
    """
    context_text = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""<|system|>
You are a precise assistant. Answer ONLY using the context below. 
If the context does not contain the answer, respond with: "Not found in the provided documents."
Do not add any information not present in the context.
<|user|>

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""
    return prompt


def generate_response(
    query: str,
    context_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 512
) -> str:
    """
    Generate a response using the Ollama LLM given retrieved context.

    Args:
        query: User's question
        context_chunks: Retrieved chunks from the retriever
        model: Ollama model name (default: tinyllama)
        temperature: Sampling temperature (low = more factual)
        max_tokens: Max tokens to generate

    Returns:
        Generated response string
    """
    if not _check_ollama_available():
        print("[generator] Warning: Ollama not available. Returning extracted context as fallback.")
        return _fallback_response(query, context_chunks)

    _pull_model_if_needed(model)
    prompt = build_prompt(query, context_chunks)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 2048
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.Timeout:
        print("[generator] Timeout waiting for Ollama. Try a smaller model or increase timeout.")
        return _fallback_response(query, context_chunks)
    except Exception as e:
        print(f"[generator] Error calling Ollama: {e}")
        return _fallback_response(query, context_chunks)


def _fallback_response(query: str, context_chunks: list[dict]) -> str:
    """
    Fallback when LLM is unavailable: return the most relevant retrieved chunk.
    This ensures the system is still usable for retrieval even without a running LLM.
    """
    if not context_chunks:
        return "No relevant context found for the query."

    best = context_chunks[0]
    return (
        f"[Retrieval-only mode — LLM unavailable]\n\n"
        f"Most relevant passage (score: {best.get('score', 0):.3f}):\n\n"
        f"{best['text']}"
    )
