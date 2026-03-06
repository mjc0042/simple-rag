from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .embedder import Embedder
from .vector_store import VectorStore

class RAGEngine:

    def __init__(
        self,
        chat_model: Any,
        embedder: Embedder,
        store: VectorStore,
    ):
        self.chat_model = chat_model
        self.embedder = embedder
        self.store = store

    def ask(self, question: str) -> str:
        query_embedding = self.embedder.embed([question])[0]
        docs = self.store.search(query_embedding)

        context = "\n\n".join(
            f"[{d.source}]\n{d.text}" for d in docs
        )

        messages = [
            SystemMessage(content="Answer the question using only the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
        ]

        response = self.chat_model.invoke(messages)

        return response.content


class GraphRAGEngine:
    def __init__(
        self,
        chat_model: Any,
        entity_extractor,
        graph_retriever,
    ):
        self.chat_model = chat_model
        self.entity_extractor = entity_extractor
        self.retriever = graph_retriever

    def ask(self, question: str) -> str:
        extracted = self.entity_extractor.extract(question)
        entities = extracted.get("entities", [])

        docs = self.retriever.retrieve(question, entities)

        context = "\n\n".join(
            f"[{d.source}]\n{d.text}" for d in docs
        )

        messages = [
            SystemMessage(content="Answer using only the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}"),
        ]

        response = self.chat_model.invoke(messages)

        return response.content
