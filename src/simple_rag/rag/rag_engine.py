from openai import OpenAI

from simple_rag.config import CHAT_MODEL
from .embedder import Embedder
from .vector_store import VectorStore

class RAGEngine:

    def __init__(
        self,
        client: OpenAI,
        embedder: Embedder,
        store: VectorStore,
    ):
        self.client = client
        self.embedder = embedder
        self.store = store

    def ask(self, question: str) -> str:
        query_embedding = self.embedder.embed([question])[0]
        docs = self.store.search(query_embedding)

        context = "\n\n".join(
            f"[{d.source}]\n{d.text}" for d in docs
        )

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
        )

        return response.choices[0].message.content

class GraphRAGEngine:
    def __init__(
        self,
        client,
        entity_extractor,
        graph_retriever,
    ):
        self.client = client
        self.entity_extractor = entity_extractor
        self.retriever = graph_retriever

    def ask(self, question: str) -> str:
        extracted = self.entity_extractor.extract(question)
        entities = extracted.get("entities", [])

        docs = self.retriever.retrieve(question, entities)

        context = "\n\n".join(
            f"[{d.source}]\n{d.text}" for d in docs
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                },
            ],
        )

        return response.choices[0].message.content
