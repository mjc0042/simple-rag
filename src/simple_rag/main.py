import sys
import os
import uuid

from openai import OpenAI

from .chat_cli import ChatCLI
from .config import OPENAI_API_KEY
from .loaders.text_loader import TextLoader
from .loaders.pdf_loader import PDFLoader
from .rag.chunker import TextChunker
from .rag.knowledge_graph import KnowledgeGraph
from .rag.embedder import Embedder
from .rag.entity_extractor import EntityExtractor
from .rag.rag_engine import GraphRAGEngine, RAGEngine
from .rag.graph_retriever import GraphRetriever
from .rag.vector_store import VectorStore

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <directory>")
        sys.exit(1)

    mode = sys.argv[1].lower() if len(sys.argv) >= 3 else "rag"
    if mode not in {"rag", "graphrag", "grag"}:
        print(f"Unknown mode '{mode}', defaulting to 'rag'")
        mode = "rag"
    elif mode.startswith('g'):
        mode = "graphrag"

    directory = sys.argv[2] if len(sys.argv) == 3 else sys.argv[1]

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Dependency injection
    embedder = Embedder(client)
    chunker = TextChunker(chunk_size=800, overlap=200)
    entity_extractor = EntityExtractor(client)
    graph = KnowledgeGraph()


    loaders = [TextLoader(), PDFLoader()]

    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            for loader in loaders:
                if loader.can_load(path):
                    documents.extend(loader.load(path))

    print(f"Loaded {len(documents)} documents")

    chunks = chunker.chunk(documents)
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks...")
    embeddings = embedder.embed([c.text for c in chunks])

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    if mode == "graphrag":

        print("Building knowledge graph...")

        for chunk in chunks:
            extraction = entity_extractor.extract(chunk.text)

            graph.add_chunk_entities(
                entities=extraction.get("entities", []),
                relations=extraction.get("relations", []),
                chunk_id=str(uuid.uuid4()),
            )
            print("...")

        retriever = GraphRetriever(store, graph, embedder)
        rag = GraphRAGEngine(client, entity_extractor, retriever)
    
    else:
        rag = RAGEngine(client=client, embedder=embedder, store=store)

    ChatCLI(rag).run()

if __name__ == "__main__":
    main()
