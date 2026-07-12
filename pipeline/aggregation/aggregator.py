from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY

class GraphAggregation:

    def _string_deduplication(self, subgraphs):
        """
        Performs string deduplication.
        """
        unique_entities = {}
        edge_map = {}
        for node in subgraphs:
            metadata = node[0].metadata
            for entity in metadata.get(KG_NODES_KEY, []):
                name = entity.name.lower().strip()
                label = entity.label.lower() if entity.label else ""
                desc = (entity.properties.get("entity_description") or entity.properties.get("relationship_description") or "")
                if name not in unique_entities:
                    unique_entities[name] = EntityNode(name=name,label=label,properties=entity.properties)
                else:
                    old_desc = (
                        unique_entities[name].properties.get("entity_description")
                        or unique_entities[name].properties.get("relationship_description")
                        or ""
                    )
                    if desc and desc not in old_desc:
                        unique_entities[name].properties["entity_description"] = (old_desc + " " + desc if old_desc else desc)

            for relation in metadata.get(KG_RELATIONS_KEY, []):
                subj = relation.source_id.lower()
                obj = relation.target_id.lower()
                rel = relation.label.lower()
                if not subj or not obj:
                    continue

                desc = relation.properties.get("relationship_description", "").strip()
                key = tuple(sorted([subj, obj]))
                if key not in edge_map:
                    edge_map[key] = {"relations": {rel},"descriptions": {desc} if desc else set()}
                else:
                    edge_map[key]["relations"].add(rel)
                    if desc:
                        edge_map[key]["descriptions"].add(desc)

        unique_edges = []
        for (subj, obj), data in edge_map.items():
            rel_concat = "|".join(sorted(data["relations"]))
            desc_concat = " | ".join(sorted(data["descriptions"]))
            unique_edges.append((subj, rel_concat, obj, desc_concat))
            
        return unique_entities, unique_edges

    def _create_final_entities(self, unique_entities):
        final_entities = {}
        for name, node in unique_entities.items():
            desc = ""
            if node.properties.get("entity_description") is not None:
                desc = node.properties["entity_description"]
            elif node.properties.get("relationship_description") is not None:
                desc = node.properties["relationship_description"]
            final_entities[name] = EntityNode(name=name,label=node.label,properties={"entity_description": desc})
        return final_entities

    def _create_final_edges(self, unique_edges):
        final_edges = {}
        for edge in unique_edges:
            edge_label = edge[1]
            if edge_label not in final_edges:
                final_edges[edge_label] = {"cluster_label": edge_label,"cluster_members": [edge],}
            else:
                members = final_edges[edge_label]["cluster_members"]
                if edge not in members:
                    members.append(edge)
        return final_edges

    def aggregate(self, subgraphs):
        """
        Aggregates the entities and edges of each subgraph.
        """
        unique_entities, unique_edges = self._string_deduplication(subgraphs)
        final_entities = self._create_final_entities(unique_entities)
        final_edges = self._create_final_edges(unique_edges)
        return final_entities, final_edges
