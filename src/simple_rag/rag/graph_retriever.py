class GraphRetriever:
    def __init__(self, vector_store, graph, embedder):
        self.vector_store = vector_store
        self.graph = graph
        self.embedder = embedder

    def retrieve(self, question: str, entities: list[str], top_k: int = 5):
        expanded = self.graph.expand_entities(entities, hops=1)

        augmented_query = (
            question
            + "\n\nRelated concepts:\n"
            + ", ".join(expanded)
        )

        embedding = self.embedder.embed([augmented_query])[0]
        return self.vector_store.search(embedding, top_k=top_k)
