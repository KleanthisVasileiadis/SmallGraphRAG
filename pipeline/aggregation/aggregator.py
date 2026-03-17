from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY
from pyjedai.datamodel import Data
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import WeightedEdgePruning
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering
import pandas as pd

class GraphAggregation:

    def __init__(self, similarity_threshold=0.5):
        self.similarity_threshold = similarity_threshold

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
                    edge_map[key] = {
                        "relations": {rel},
                        "descriptions": {desc} if desc else set()
                    }
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

    def _entities_to_dataframe(self, unique_entities):
        """
        Converts entities to Dataframe.
        """
        rows = []
        for idx, (name, entity) in enumerate(unique_entities.items()):
            desc = (entity.properties.get("entity_description") or entity.properties.get("relationship_description") or "")
            rows.append({"id": str(idx), "name": entity.name, "description": desc})
        return pd.DataFrame(rows)

    def _cluster(self, df):
        """
        Clusters the entities using the pyjedAI library.
        """
        data = Data(dataset_1=df,id_column_name_1="id")
        
        bb = StandardBlocking()
        blocks = bb.build_blocks(data, attributes_1=['name'], attributes_2=['name'])

        bp = BlockPurging()
        cleaned_blocks = bp.process(blocks, data, tqdm_disable=False)

        bf = BlockFiltering(ratio=0.8)
        filtered_blocks = bf.process(cleaned_blocks, data, tqdm_disable=False)

        mb = WeightedEdgePruning(weighting_scheme='EJS')
        candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=False)

        em = EntityMatching(metric='cosine',tokenizer='char_tokenizer',vectorizer='tfidf',qgram=3,similarity_threshold=0.0)
        pairs_graph = em.predict(candidate_pairs_blocks, data, tqdm_disable=False)

        ccc = ConnectedComponentsClustering()
        return ccc.process(pairs_graph, data, similarity_threshold=self.similarity_threshold)

    def _merge_entities(self, clusters, df, unique_entities):
        """
        Merges all the entities within a cluster.
        """
        id_to_name = {int(row.id): row.name.lower().strip() for row in df.itertuples()}
        merged_entities = {}
        redirect_map = {}
        for cluster_idx, cluster in enumerate(clusters):
            canonical_id = min(cluster)
            canonical_name = id_to_name[canonical_id]

            print("\n" + "=" * 50)
            print(f"[CLUSTER {cluster_idx}]")
            print(f"Canonical ID   : {canonical_id}")
            print(f"Canonical Name : '{canonical_name}'")
            print(f"Cluster Size   : {len(cluster)}")
            print("Members:")

            canonical_entity = unique_entities[canonical_name]
            merged_desc = set()

            for eid in cluster:
                name = id_to_name[eid]
                print(f"  - {name}")

                redirect_map[name] = canonical_name
                ent = unique_entities[name]

                desc = (ent.properties.get("entity_description") or ent.properties.get("relationship_description") or "")
                if desc:
                    merged_desc.add(desc)

            if merged_desc:
                canonical_entity.properties["entity_description"] = " ".join(merged_desc)

            merged_entities[canonical_name] = canonical_entity

        print("\n" + "=" * 50)
        print(f"Total clusters: {len(clusters)}")
        return merged_entities, redirect_map


    def _redirect_edges(self, edges, redirect_map):
        """
        Redirects the edges in the clusters.
        """
        redirected = set()
        for subj, rel, obj, desc in edges:
            redirected.add((redirect_map.get(subj, subj), rel, redirect_map.get(obj, obj), desc))
        return list(redirected)

    def aggregate(self, subgraphs):
        """
        Aggregates the entities and edges of each subgraph.
        """
        unique_entities, unique_edges = self._string_deduplication(subgraphs)
        
        return unique_entities, unique_edges

        # print(f"before: {len(unique_entities)} ents, {len(unique_edges)} rels")
        # df = self._entities_to_dataframe(unique_entities)
        # clusters = self._cluster(df)
        # merged_entities, redirect_map = self._merge_entities(clusters, df, unique_entities)
        # edges = self._redirect_edges(unique_edges, redirect_map)
        # print(f"after: {len(merged_entities)} ents, {len(edges)} rels")
        # return merged_entities, edges

