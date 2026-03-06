import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


class EntityExtractor:
    def __init__(self, chat_model: Any):
        self.chat_model = chat_model

    def extract(self, text: str) -> dict:
        messages = [
            SystemMessage(
                content=(
                    "Extract named entities and relationships from the text.\n"
                    "Return JSON with keys: entities (list of strings), "
                    "relations (list of {subject, predicate, object})."
                )
            ),
            HumanMessage(content=text),
        ]

        response = self.chat_model.invoke(messages)

        return json.loads(response.content)
