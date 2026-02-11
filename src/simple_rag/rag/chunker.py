from typing import List
from .document import Document

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = []

        for doc in documents:
            text = doc.text
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append(
                    Document(
                        text=chunk_text,
                        source=doc.source
                    )
                )

                start += self.chunk_size - self.overlap

        return chunks
