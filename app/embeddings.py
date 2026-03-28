from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL


_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a text string."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embedding vectors for a batch of texts."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
