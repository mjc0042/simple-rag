from abc import ABC, abstractmethod
from typing import List
from simple_rag.rag.document import Document

class BaseLoader(ABC):

    @abstractmethod
    def can_load(self, path: str) -> bool:
        pass

    @abstractmethod
    def load(self, path: str) -> List[Document]:
        pass
