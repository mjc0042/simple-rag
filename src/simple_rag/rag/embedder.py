import numpy as np

from openai import OpenAI

from simple_rag.config import OPENAI_API_KEY, EMBEDDING_MODEL

class Embedder:

    def __init__(self, client: OpenAI):
        self.client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return np.array([d.embedding for d in response.data])
