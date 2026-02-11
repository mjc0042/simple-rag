import faiss
import numpy as np
from .document import Document

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.documents: list[Document] = []

    def add(self, embeddings: np.ndarray, documents: list[Document]):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        return [
            self.documents[i]
            for i in indices[0]
            if i != -1
        ]

# Currently broken from pydantic v2 bug [Chroma import fails on python 3.14.2 and pydantic 2.12+ #5996]
#class ChromaVectorStore:
#
#    def __init__(self, collection_name: str = "rag_docs"):
#        self.client = chromadb.Client(
#            #Settings(anonymized_telemetry=False)
#        )
#        self.collection = self.client.create_collection(collection_name)
#
#    def add(self, embeddings, documents: list[Document]):
#        ids = [f"doc_{i}" for i in range(self.collection.count(), self.collection.count() + len(documents))]
#
#        self.collection.add(
#            ids=ids,
#            embeddings=embeddings.tolist(),
#            documents=[d.text for d in documents],
#            metadatas=[{"source": d.source} for d in documents],
#        )
#
#    def search(self, query_embedding, top_k: int = 5):
#        results = self.collection.query(
#            query_embeddings=[query_embedding.tolist()],
#            n_results=top_k,
#        )
#
#        docs = []
#        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
#            docs.append(
#                Document(
#                    text=text,
#                    source=meta["source"]
#                )
#            )
#        return docs
