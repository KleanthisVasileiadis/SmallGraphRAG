from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import ChatMessage
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
import networkx as nx
import numpy as np
import re
from pipeline.debug.debug_functions import graph_statistics

def final_graph_node(final_entities, final_edges):
    #Creates the final Graph node
    g_node = TextNode(text="",metadata={})
    existing_nodes = []
    existing_relations = []
    metadata = g_node.metadata.copy()
    entity_lookup = {}
    entity_cluster_lookup = {}
    #To each raw entity i assign to each of its members the cluster label
    for label, entity_obj in final_entities.items():
        members = entity_obj.properties.get("cluster_members", [])
        if not members:
            entity_cluster_lookup[label] = label
        else:
            for member in members:
                entity_cluster_lookup[member] = label

    #for each processed entity i create the entity node
    entity_lookup = {}
    existing_nodes = []
    for label, entity_obj in final_entities.items():
        if isinstance(entity_obj, EntityNode):
            node = entity_obj
        else: 
            node = EntityNode(name=label,properties={"cluster_members": entity_obj.get("cluster_members", [])})
        entity_lookup[label] = node
        existing_nodes.append(node)

    #For each edge
    for label, edge_info in final_edges.items():
        for member in edge_info["cluster_members"]:
            #Check if the description exists
            if len(member) == 3:
                subj, rel, obj = member
                desc = ""
            else:
                subj, rel, obj, desc = member
            subj_cluster = entity_cluster_lookup.get(subj, subj)
            obj_cluster = entity_cluster_lookup.get(obj, obj)
            subj_node = entity_lookup.get(subj_cluster)
            obj_node = entity_lookup.get(obj_cluster)
            #If the node doesnt exist i continue to the next loop
            if subj_node is None or obj_node is None:
                continue
            #Adds the final Relation item to the existing_relations list
            rel_node = Relation(
                label=label,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties={"cluster_label": label,"cluster_members": edge_info["cluster_members"],"relationship_description": desc},
            )
            existing_relations.append(rel_node)

    #And updates the nodes metadata
    g_node.metadata["KG_NODES_KEY"] = existing_nodes
    g_node.metadata["KG_RELATIONS_KEY"] = existing_relations
    print(f"Created unified graph with {len(existing_nodes)} entities and {len(existing_relations)} triples.")
    return g_node


class GraphRAGStore(SimplePropertyGraphStore):
    
    community_summary = {}
    community_tree = {} #parent -> children map
    community_parent = {} #child -> parent map
    summary_vecs = {}   

    max_cluster_size = 100

    def __init__(self, llm, embedder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.embedder = embedder
        self.node_lookup = {}
        self.community_element_summaries = {}
        self.levels = {} 
        self.node_vecs = {}

    def get_node_embeddings(self):
        if not self.node_vecs:  
            self.build_communities()
        return self.node_vecs

    def _build_node_embeddings(self):
        """
        Creates embeddings for each graph node based on its name and description.
        """
        self.node_vecs = {}
        for node_id, node in self.node_lookup.items():
            name = node.name
            desc = ""
            if node.properties.get("entity_description") is not None:
                desc = node.properties.get("entity_description")
            elif node.properties.get("relationship_description") is not None:
                desc = node.properties.get("relationship_description")

            text = f"{name}. {desc}"
            emb = self.embedder.embed(text)
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0 or np.isnan(norm):
                continue

            emb = emb / norm
            self.node_vecs[node_id] = emb

    def _build_summary_embeddings(self):
        """
        Creates embeddings for each community summary.
        """
        for cid, text in self.community_summary.items():
            emb = self.embedder.embed(text)
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0 or np.isnan(norm):
                continue
            emb = emb / norm
            self.summary_vecs[cid] = emb

    def _count_tokens(self,text):
        return len(self.llm._model.tokenize(text.encode("utf-8")))

    def _reduce_in_batches(self, texts, token_limit):
        if not texts:
            return ""

        #reserve tokens for system prompt and formatting
        safe_limit = token_limit - 1500
        total_tokens = sum(self._count_tokens(t) for t in texts)
        
        #If everything fits in one call
        if total_tokens <= safe_limit:
            return self._llm_summarize("\n".join(texts))

        reduced = []
        current_chunk = []
        current_tokens = 0
        for text in texts:
            text_tokens = self._count_tokens(text)
            #single text too large
            if text_tokens > safe_limit:
                tokens = text.split()
                truncated_text = " ".join(tokens[:safe_limit])
                summary = self._llm_summarize(truncated_text)
                reduced.append(summary)
                continue
            
            #chunk would exceed limit
            if current_tokens + text_tokens > safe_limit:
                chunk_text = "\n".join(current_chunk)
                chunk_summary = self._llm_summarize(chunk_text)
                reduced.append(chunk_summary)
                current_chunk = [text]
                current_tokens = text_tokens
            else:
                current_chunk.append(text)
                current_tokens += text_tokens

        #Final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_summary = self._llm_summarize(chunk_text)
            reduced.append(chunk_summary)
        
        #Stop recursion if only one summary remains
        if len(reduced) == 1:
            return reduced[0]

        return self._reduce_in_batches(reduced, token_limit)

    def _compute_internal_summaries(self, token_limit=4000):
        """
        Computes bottom-up summaries for all internal communities.
        """
        sorted_levels = sorted(self.levels.keys(), reverse=True)
        for level in sorted_levels:
            for cid in self.levels[level]:
                #Skips leaf communities
                if cid in self.community_summary:
                    continue

                #Gets the child summaries
                child_ids = self.community_tree.get(cid, [])
                child_texts = []
                for child in child_ids:
                    text = self.community_summary.get(child, "")
                    if text:
                        child_texts.append(text)

                #Gets the community elements
                elements = self.community_element_summaries.get(cid, [])
                element_texts = [element["text"] for element in elements]

                #Merges everything
                all_texts = child_texts + element_texts
                if not all_texts:
                    self.community_summary[cid] = ""
                    continue

                #Recursive Reduce
                summary = self._reduce_in_batches(all_texts, token_limit)
                self.community_summary[cid] = summary

    def _llm_summarize(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    f"You are an AI assistant that summarizes relationships from a knowledge graph."
                    f"\n"
                    f"Each input consists of multiple relationship records in the form:"
                    f" entity1 -> entity2 -> relation -> relationship_description"
                    f"\n"
                    f"### TASK\n"
                    f"Produce a detailed, information-rich summary that:\n"
                    f"- Mentions ALL key entities involved.\n"
                    f"- Preserves EVERY important factual detail from the relationship descriptions.\n"
                    f"- Synthesizes the meaning and significance of the relationships.\n"
                    f"- Captures nuances, context, and implications expressed in the descriptions.\n"
                    f"- Does NOT omit meaningful facts unless they are strictly redundant.\n"
                    f"- Avoid repeating the same fact more than once."
                    f"- Do NOT restate the same relationship in different wording."
                    f"- If a fact is already stated clearly, do not rephrase it again."
                    f"- The summary must be concise and non-redundant."
                    f"- Writes in clear natural language.\n"
                    f"- Is coherent, well-structured, and comprehensive.\n"
                    f"- Remains readable but prioritizes completeness over brevity.\n"
                    f"\n"
                    f"The summary should describe:\n"
                    f"- the nature of the relationships,\n"
                    f"- the purpose or role connecting the entities,\n"
                    f"- and any relevant attributes provided by the descriptions.\n"
                    f"\n"
                    f"Do NOT oversimplify. Do NOT drop significant information.\n"
                    f"\n"
                    f"---\n"
                    f"### FEW-SHOT EXAMPLES\n"
                    f"\n"
                    f"#### Example 1\n"
                    f"Input:\n"
                    f"Mars -> Sun -> orbits -> Mars completes a full orbital revolution around the Sun every 687 Earth days.\n"
                    f"Mars -> Sun -> distance -> Mars is located about 1.52 AU from the Sun.\n"
                    f"Mars -> Sun -> orbital_speed -> Mars travels at an average orbital speed of 24 km/s.\n"
                    f"Mars -> Sun -> orbit_shape -> Mars follows a slightly elliptical orbit around the Sun.\n"
                    f"\n"
                    f"Output Summary:\n"
                    f"Mars is a planet that orbits the Sun following a slightly elliptical path, traveling at an average speed of 24 km/s. "
                    f"It completes a full revolution approximately every 687 Earth days and maintains an average distance of about 1.52 AU "
                    f"from the Sun. Together, these details describe the key orbital characteristics that define Mars’ motion within the "
                    f"solar system.\n"
                    f"\n"
                    f"---\n"
                    f"### YOUR TASK\n"
                    f"Write one coherent, detailed, and information-complete summary following the rules above."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def _compute_leaf_summaries(self, token_limit=4000):
        """
        Computes summaries for all leaf-level communities.
        """
        #Finds leaf communities
        parents = set(self.community_tree.keys())
        all_clusters = set(self.community_element_summaries.keys())
        leaf_clusters = all_clusters.difference(parents)

        #Processes each leaf
        for cid in leaf_clusters:
            #Gets the element summary of the cluster
            elements = self.community_element_summaries.get(cid, [])
            element_texts = [element["text"] for element in elements]
            if not element_texts:
                self.community_summary[cid] = ""
                continue

            #Recursive reduce
            summary = self._reduce_in_batches(element_texts, token_limit)
            self.community_summary[cid] = summary


    def _get_neighbors_in_cluster(self, node_id, cluster_nodes):
        return [nbr for nbr in self.nx_graph.neighbors(node_id) if nbr in cluster_nodes]

    def _get_edge(self, a, b):
        if self.nx_graph.has_edge(a, b):
            return self.nx_graph.get_edge_data(a, b)
        return None

    def _collect_element_summaries(self):
        """
        For each community, gathers node and edge summaries.
        These summaries are just a concat of the name and the 
        properties and no llm is present yet.
        """
        self.community_element_summaries = {}
        for level, clusters in self.levels.items():
            for cluster_id, nodes in clusters.items():
                elements = []
                #Node summaries
                for node in nodes:
                    node_obj = self.node_lookup.get(node)
                    if node_obj:
                        desc = " "
                        if "entity_description" in node_obj.properties:
                            desc = node_obj.properties["entity_description"]
                        elif "relationship_description" in node_obj.properties:
                            desc = node_obj.properties["relationship_description"]
                        summary = f"Node {node_obj.name}: {desc}"
                        elements.append({"type": "node","node_id": node,"text": summary,"tokens": self._count_tokens(summary)})

                #Edge summaries
                for node in nodes:
                    for nbr in self._get_neighbors_in_cluster(node, nodes):
                        edge_data = self._get_edge(node, nbr)
                        if not edge_data:
                            continue
                        rel = edge_data.get("relationship", "")
                        desc = edge_data.get("description", "")
                        summary = f"{node} -> {nbr} : {rel} | {desc}"
                        elements.append({"type": "edge","src": node,"dst": nbr,"text": summary,"tokens": self._count_tokens(summary)})

                self.community_element_summaries[cluster_id] = elements

    def _compute_parent_child_mapping(self, levels):
        """
        Builds a parent->children and child->parent mapping
        according to the hierarchical Leiden structure.
        levels is a dict: levels[level][cluster_id] = [node_ids]
        """
        parent_map = defaultdict(list)
        child_parent = {}
        sorted_levels = sorted(levels.keys())
        #Iterates over pairs of consecutive levels
        for idx in range(len(sorted_levels) - 1):
            level = sorted_levels[idx]
            next_level = sorted_levels[idx + 1] 
            #Maps each node to its cluster at the current level
            mapping_level = {}
            for cluster_id, nodes in levels[level].items():
                for node in nodes:
                    mapping_level[node] = cluster_id

            #Maps each node to its cluster at the next level
            mapping_next_level = {}
            for cluster_id, nodes in levels[next_level].items():
                for node in nodes:
                    mapping_next_level[node] = cluster_id

            #Creates the parent-child relations
            for node_id, child_cluster in mapping_next_level.items():
                if node_id not in mapping_level:
                    continue

                parent_cluster = mapping_level[node_id]
                parent_map[parent_cluster].append(child_cluster)
                child_parent[child_cluster] = parent_cluster

        return dict(parent_map), dict(child_parent)

    def _organize_clusters_by_level(self, clusters):
        """
        Transforms hierarchical_leiden's clusters into a list of the form:
        levels[level][cluster_id] = [node_ids]
        """
        levels = defaultdict(dict)
        for item in clusters:
            level = item.level
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in levels[level]:
                levels[level][cluster_id] = []
            levels[level][cluster_id].append(node)

        return levels

    def _create_nx_graph(self):
        """
        Creates the nx graph and appends each node and edge.
        """
        nx_graph = nx.Graph()
        self.node_lookup = {}
        for node in self.graph.nodes.values():
            desc = ""
            if node.properties.get("entity_description") is not None:
                desc = node.properties.get("entity_description")
            elif node.properties.get("relationship_description") is not None:
                desc = node.properties.get("relationship_description")
            nx_graph.add_node(
                node.id,
                name=node.name,
                description=desc)
            self.node_lookup[node.id] = node
        for relation in self.graph.relations.values():
            source = relation.source_id
            target = relation.target_id
            #skip edges with missing endpoints
            if source not in self.node_lookup or target not in self.node_lookup:
                continue
            nx_graph.add_edge(source,target,relationship=relation.label,description=relation.properties.get("relationship_description", ""))

        graph_statistics(nx_graph)

        return nx_graph

    def _compute_dynamic_cluster_size(self, G):
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if num_nodes == 0:
            return 100

        density = num_edges / num_nodes
        base = int(num_nodes / (10 + density))
        max_cluster_size = max(500, min(base, 3000))
        print(f"Dynamic max_cluster_size set to: {max_cluster_size}")
        return max_cluster_size

    #Build the communities
    def build_communities(self):
        #Creates the nx graph
        self.nx_graph = self._create_nx_graph()
        print(f"NX Graph created!\n")
        #Creates the clusters using hierarchical leiden
        dynamic_size = self._compute_dynamic_cluster_size(self.nx_graph)
        clusters = hierarchical_leiden(self.nx_graph,max_cluster_size=dynamic_size)
        print(f"Leyden Clusters created!\n")
        #Organizes clusters into hierarchical levels
        self.levels = self._organize_clusters_by_level(clusters)
        print(f"Clusters organized by level completed!\n")
        #Computes parent-child mapping between levels
        self.community_tree, self.community_parent = self._compute_parent_child_mapping(self.levels)
        print("Hierarchical community detection completed:")
        for lvl in sorted(self.levels.keys()):
            num_comm = len(self.levels[lvl])
            print(f"  Level {lvl}: {num_comm} communities")

        #Summarizes the nodes and edges in each level
        self._collect_element_summaries()
        print(f"Collected element summaries!\n")
        #Summarizes the leaf communities
        self._compute_leaf_summaries(token_limit=11000) 
        print(f"Computed leaf summaries!\n")
        #Summarizes the rest of the communities
        self._compute_internal_summaries(token_limit=11000)
        print(f"computed internal summaries!\n")
        #Creates embeddings for all community summaries
        self._build_summary_embeddings()
        print(f"Builded summary embeddings")
        #Creates embeddings for all nodes
        self._build_node_embeddings()
        print("GraphRAGStore is complete!\n")

    #call the class
    def get_community_summaries(self):
        if not self.community_summary:
            print(f"Building the communities...\n")
            self.build_communities()
        return self.community_summary
