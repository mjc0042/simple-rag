import json
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import List

class Relation(BaseModel):
    subject: str 
    predicate: str
    object: str

class EntityExtraction(BaseModel):
    entities: List[str]
    relations: List[Relation]

class EntityExtractor:
    def __init__(self, chat_model: Any):
        self.chat_model = chat_model

    def extract(self, text: str) -> dict:
        parser = JsonOutputParser(pydantic_object=EntityExtraction)

        messages = [
            SystemMessage(
                content=(
                    "Extract named entities and relationships from the text.\n"
                    "Return JSON with keys: entities (list of strings), "
                    "relations (list of {subject, predicate, object})."
                    f"{parser.get_format_instructions()}"
                )
            ),
            HumanMessage(content=text),
        ]

        response = self.chat_model.invoke(messages)

        if not response:
            raise ValueError("Empty reponse from LLM.")

        return parser.invoke(response)
