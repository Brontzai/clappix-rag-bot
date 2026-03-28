"""
RAG движок — ядро бота.

Как работает:
1. Загружаем документы из data/ → разбиваем на чанки
2. Каждый чанк превращаем в вектор (OpenAI Embeddings)
3. Сохраняем в ChromaDB
4. Когда приходит вопрос → ищем 3 самых похожих чанка → отправляем в GPT с контекстом

TODO: добавить подержку PDF (сейчас только .md и .txt)
TODO: попробвать разные стратегии чанкинга
"""
import logging
from pathlib import Path
import chromadb
from openai import OpenAI
from app.config import (
    OPENAI_API_KEY, OPENAI_MODEL, CHROMA_DIR, DATA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

# Инициализация клиентов
_client = OpenAI(api_key=OPENAI_API_KEY)
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},  # косинусное расстояние для сравнения векторов
)


def _split_into_chunks(text: str) -> list[str]:
    """
    Разбиваем текст на куски по ~500 слов с перекрытием в 50 слов.
    Перекрытие нужно чтобы не потерять контекст на границе чанков.
    """
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
    """Получаем вектор тексат через OpenAI API."""
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def load_documents() -> int:
    """
    Загружаем все .md и .txt файлы из папки data/ в ChromaDB.
    Возвращает количество проиндексированных чанков.

    При повторном вызове — очищает старые данные и загружает заново.
    """
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        log.warning("Папка %s не найдена", DATA_DIR)
        return 0

    files = list(data_path.glob("*.md")) + list(data_path.glob("*.txt"))
    if not files:
        log.warning("Нет файлов .md/.txt в %s", DATA_DIR)
        return 0

    # Чистим старые данные перед переиндексацией
    existing = _collection.count()
    if existing > 0:
        _collection.delete(where={"source": {"$ne": ""}})
        log.info("Очистили %d старых чанков", existing)

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

    # Эмбеддим батчами по 50 (лимит API)
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

    log.info("Проиндексировали %d чанков из %d файлов", len(all_chunks), len(files))
    return len(all_chunks)


def search(query: str) -> list[str]:
    """
    Ищем релевантные чанки по вопросу.
    Возвращаем top-3 самых похожих куска текста.
    """
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
    """
    Главная функция — отвечаем на вопрос пользователя.

    1. Ищем релевантные чанки в ChromaDB
    2. Формируем промпт с контекстом
    3. Отправляем в GPT
    4. Возвращаем ответ
    """
    relevant_chunks = search(query)

    if not relevant_chunks:
        return ("У меня пока нет информации по этому вопросу. "
                "Попробуйте переформулировать или спросите что-то другое.")

    # Склеиваем найденые куски в один контекст
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
        temperature=0.3,  # низкая температура — отвечаем точнее, меньше фантазий
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()
