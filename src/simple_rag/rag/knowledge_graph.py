from typing import Iterable

import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_chunk_entities(
        self,
        entities: Iterable[str],
        relations: Iterable[dict],
        chunk_id: str,
    ):
        # Add entities (filter invalid)
        for entity in entities:
            if isinstance(entity, str) and entity.strip():
                self.graph.add_node(entity.strip())

        # Add relations (defensive)
        for r in relations:
            subject = r.get("subject")
            predicate = r.get("predicate")
            object_ = r.get("object")

            if not subject or not object_ or not predicate:
                continue

            if not isinstance(subject, str) or not isinstance(object_, str):
                continue

            subject = subject.strip()
            object_ = object_.strip()
            predicate = predicate.strip()

            if not subject or not object_:
                continue

            self.graph.add_edge(
                subject,
                object_,
                label=predicate,
                chunk_id=str(chunk_id),
            )

    def expand_entities(self, entities: Iterable[str], hops: int = 1) -> set[str]:
        expanded = set(entities)

        for entity in entities:
            if entity not in self.graph:
                continue

            neighbors = nx.single_source_shortest_path_length(
                self.graph, entity, cutoff=hops
            ).keys()

            expanded.update(neighbors)

        return expanded
