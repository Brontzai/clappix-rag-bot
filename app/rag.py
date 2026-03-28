import os
import logging
from pathlib import Path
import chromadb
from openai import OpenAI
from app.config import (
    OPENAI_API_KEY, OPENAI_MODEL, CHROMA_DIR, DATA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)


def _split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks by character count."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _get_embedding(text: str) -> list[float]:
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def load_documents() -> int:
    """Load all .md and .txt files from DATA_DIR into ChromaDB.

    Returns the number of chunks indexed.
    """
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        log.warning("Data directory %s does not exist", DATA_DIR)
        return 0

    files = list(data_path.glob("*.md")) + list(data_path.glob("*.txt"))
    if not files:
        log.warning("No .md or .txt files found in %s", DATA_DIR)
        return 0

    # Clear existing data
    existing = _collection.count()
    if existing > 0:
        _collection.delete(where={"source": {"$ne": ""}})
        log.info("Cleared %d existing chunks", existing)

    all_chunks = []
    all_ids = []
    all_meta = []

    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file.stem}_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append({"source": file.name, "chunk_index": i})

    if not all_chunks:
        return 0

    # Embed and store in batches of 50
    batch_size = 50
    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        batch_texts = all_chunks[start:end]
        batch_ids = all_ids[start:end]
        batch_meta = all_meta[start:end]

        embeddings_resp = _client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )
        embeddings = [item.embedding for item in embeddings_resp.data]

        _collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    log.info("Indexed %d chunks from %d files", len(all_chunks), len(files))
    return len(all_chunks)


def search(query: str) -> list[str]:
    """Search for relevant document chunks."""
    if _collection.count() == 0:
        return []

    query_embedding = _get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    documents = results.get("documents", [[]])[0]
    return documents


def answer(query: str) -> str:
    """Search relevant docs and generate an answer with GPT."""
    relevant_chunks = search(query)

    if not relevant_chunks:
        return "I don't have enough information to answer that question. Please try rephrasing or ask something else."

    context = "\n\n---\n\n".join(relevant_chunks)

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for Clappix — an AI automation company. "
                    "Answer the user's question based ONLY on the provided context. "
                    "If the context doesn't contain the answer, say so honestly. "
                    "Be concise and professional. Respond in the same language as the question."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()
