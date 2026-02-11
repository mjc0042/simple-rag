from pypdf import PdfReader

from .base import BaseLoader
from simple_rag.rag.document import Document

class PDFLoader(BaseLoader):

    def can_load(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def load(self, path: str):
        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(text=text, source=f"{path}#page={i+1}"))
        return docs
