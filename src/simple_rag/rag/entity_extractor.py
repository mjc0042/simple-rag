import json
from openai import OpenAI


class EntityExtractor:
    def __init__(self, client: OpenAI):
        self.client = client

    def extract(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract named entities and relationships from the text.\n"
                        "Return JSON with keys: entities (list of strings), "
                        "relations (list of {subject, predicate, object})."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return json.loads(response.choices[0].message.content)
