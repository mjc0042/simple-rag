from .base import BaseLoader
from simple_rag.rag.document import Document

class TextLoader(BaseLoader):

    def can_load(self, path: str) -> bool:
        return path.lower().endswith(".txt")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [Document(text=f.read(), source=path)]
