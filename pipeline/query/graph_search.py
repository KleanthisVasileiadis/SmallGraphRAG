from pipeline.extraction.prompts import KG_ENTITY_EXTRACT_TMPL
from llama_index.core.prompts import PromptTemplate
from pipeline.extraction.parse_functions import parse_en_fn
from llama_index.core.llms import LLM, ChatMessage
from pipeline.graph.graph_store import GraphRAGStore
from typing import Any
import numpy as np
from collections import deque
import re

class GraphSearch:

    graph_store: GraphRAGStore
    llm: LLM
    embedder: Any 

    def __init__(self,graph_store: GraphRAGStore,llm: LLM,embedder: Any ,similarity_top_k: int = 5,max_hops: int = 3,
        max_evidence_nodes: int = 20,similarity_threshold: float = 0.5,token_budget: int = 10000):

        self._graph_store = graph_store
        self._llm = llm
        self._embedder = embedder

        self._similarity_top_k = similarity_top_k
        self._max_hops = max_hops
        self._max_evidence_nodes = max_evidence_nodes
        self._similarity_threshold = similarity_threshold
        self._token_budget = token_budget

        self._node_embeddings = self._graph_store.get_node_embeddings()
        self._node_ids = list(self._node_embeddings.keys())

    def _extract_entities(self, query):
        #Enity extraction from the query
        if query is None:
            return []
        
        prompt = PromptTemplate(KG_ENTITY_EXTRACT_TMPL)
        for _ in range(5):
            response = parse_en_fn(self._llm.predict(prompt, text=query))
            raw_entities = [entity[0].strip() for entity in response if entity]
            query_lower = query.lower()
            seen = set()
            entities = []
            for e in raw_entities:
                key = e.lower()
                if key in query_lower and key not in seen and len(e) > 1:
                    seen.add(key)
                    entities.append(e)

            if entities:
                break

        return entities
    
    def _retrieve_query_nodes(self, query):
        #Computes the nodes most similar to the query
        query_emb = np.array(self._embedder.embed(query))
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        similarities = []
        for node_id, node_emb in self._node_embeddings.items():
            sim = np.dot(query_emb, node_emb)
            if sim >= self._similarity_threshold:
                similarities.append((node_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self._similarity_top_k]

    def _string_match_nodes(self, text):
        #String matches nodes to the entities extracted from the query
        text = text.lower()
        tokens = set(text.split())
        matches = {}
        for node_id, node in self._graph_store.node_lookup.items():
            node_name = str(getattr(node, "name", "")).lower()
            node_desc = str(getattr(node, "description", "")).lower()
            node_text = f"{node_name} {node_desc}"
            node_tokens = set(node_text.split())
            score = 0.0
            #exact match
            if text == node_name:
                score = 1.2

            #substring match
            elif text in node_text:
                score = 1.0

            #token overlap
            else:
                overlap = len(tokens & node_tokens)
                if overlap > 0:
                    score = 0.6 * (overlap / len(tokens))

            if score > 0:
                matches[node_id] = max(matches.get(node_id, 0), score)

        return [(nid, s) for nid, s in matches.items()]

    def _link_entities(self, entities, query):
        candidates = {}
        #Extracted entities similarity
        for entity in entities:
            emb = self._normalize(self._embedder.embed(entity))
            #Embedding similarity
            for node_id, node_emb in self._node_embeddings.items():
                sim = float(np.dot(emb, node_emb))
                candidates[node_id] = max(candidates.get(node_id, 0), sim)

            #String matching for each entity
            string_matches = self._string_match_nodes(entity)
            for node_id, score in string_matches:
                candidates[node_id] = max(candidates.get(node_id, 0), score)

        #Query similarity
        query_emb = self._normalize(self._embedder.embed(query))
        for node_id, node_emb in self._node_embeddings.items():
            sim = float(np.dot(query_emb, node_emb))
            candidates[node_id] = max(candidates.get(node_id, 0), sim)

        #String matching for the whole query
        query_string_matches = self._string_match_nodes(query)
        for node_id, score in query_string_matches:
            candidates[node_id] = max(candidates.get(node_id, 0), score)

        ranked = sorted(candidates.items(),key=lambda x: x[1],reverse=True)
        return ranked[:self._similarity_top_k]

    def _normalize(self, vec):
        vec = np.array(vec)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def _node_query_similarity(self, node_id, query_emb):
        node_emb = self._node_embeddings.get(node_id)
        if node_emb is None:
            return 0.0
        return float(np.dot(query_emb, node_emb))

    def _expand_subgraph(self, matched_nodes, query):

        if not matched_nodes:
            return {"nodes": set(), "edges": [], "paths": []}

        #Query embedding
        query_emb = self._normalize(self._embedder.embed(query))

        #Collects start nodes
        start_nodes = [node for node, _ in matched_nodes]

        visited = set(start_nodes)
        edges_collected = []
        paths = []
        seen_edges = set()
        queue = deque()
        for node in start_nodes:
            queue.append((node, 0, [node]))

        while queue:
            current_node, depth, path = queue.popleft()
            if depth >= self._max_hops:
                continue

            neighbors = self._graph_store.nx_graph.neighbors(current_node)
            for nbr in neighbors:
                #if neighbors similarity is greater than similarity thresshold visit him
                sim = self._node_query_similarity(nbr, query_emb)
                if sim < self._similarity_threshold:
                    continue

                edge_data = self._graph_store.nx_graph.get_edge_data(current_node, nbr)
                if edge_data is None:
                    continue

                #Deduplicates edges
                edge_key = (current_node, nbr, edge_data.get("relationship", ""))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_collected.append({
                        "source": current_node,
                        "target": nbr,
                        "relationship": edge_data.get("relationship", ""),
                        "description": edge_data.get("description", "")
                    })

                if nbr not in visited:
                    visited.add(nbr)
                    new_path = path + [nbr]
                    paths.append(new_path)
                    queue.append((nbr, depth + 1, new_path))

        return {"nodes": visited,"edges": edges_collected,"paths": paths}


    def _build_evidence_chunks(self, subgraph: dict):

        chunks = []
        #Nodes
        for node_id in subgraph["nodes"]:
            node_obj = self._graph_store.node_lookup.get(node_id)
            if not node_obj:
                continue

            name = node_obj.name
            desc = node_obj.properties.get("entity_description", "") or node_obj.properties.get("relationship_description", "")

            text = (
                f"NODE:\n"
                f"Name: {name}\n"
                f"ID: {node_id}\n"
                f"Description: {desc}"
            )

            chunks.append(text)

        #Edges
        for edge in subgraph["edges"]:
            text = (
                f"RELATIONSHIP:\n"
                f"Source: {edge['source']}\n"
                f"Target: {edge['target']}\n"
                f"Type: {edge['relationship']}\n"
                f"Description: {edge['description']}"
            )

            chunks.append(text)

        #Paths
        for path in subgraph["paths"]:
            readable_path = " -> ".join(path)
            text = f"REASONING PATH:\n{readable_path}"
            chunks.append(text)

        return chunks

    def _rank_chunks_by_query(self, chunks, query):

        query_emb = self._normalize(self._embedder.embed(query))
        scored = []
        for chunk in chunks:
            chunk_emb = self._normalize(self._embedder.embed(chunk))
            sim = float(np.dot(query_emb, chunk_emb))
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [c for c, _ in scored[:self._max_evidence_nodes]]
        return top_chunks

    def _llm_summarize(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a knowledge graph evidence summarization assistant.\n\n"

                    "You will receive graph evidence extracted from a knowledge graph.\n"
                    "The evidence may contain:\n"
                    "- NODE descriptions\n"
                    "- RELATIONSHIPS between entities\n"
                    "- REASONING PATHS connecting entities\n\n"

                    "Your task is to produce a detailed and structured summary of ALL the evidence.\n"
                    "The goal is to clearly describe the entities, their relationships, and any reasoning paths.\n\n"

                    "Important rules:\n"
                    "- DO NOT answer any question\n"
                    "- DO NOT infer conclusions beyond the evidence\n"
                    "- ONLY summarize the information present in the graph evidence\n"
                    "- Include important entities and their descriptions\n"
                    "- Include relationships between entities\n"
                    "- Explain reasoning paths when they appear\n"
                    "- Avoid repeating the same information\n"
                    "- Ignore formatting artifacts like NODE, RELATIONSHIP, or PATH labels in the final text\n"
                    "- Write a clear and detailed natural language summary\n"
                    "- Output ONLY the summary text\n\n"

                    "Below are examples of correct behavior.\n\n"

                    "Example 1\n"
                    "Graph Evidence:\n"
                    "NODE: Elon Musk — entrepreneur and business magnate.\n"
                    "NODE: SpaceX — aerospace manufacturer.\n"
                    "NODE: Tesla — electric vehicle company.\n"
                    "RELATIONSHIP: Elon Musk FOUNDED SpaceX.\n"
                    "RELATIONSHIP: Elon Musk CO-FOUNDED Tesla.\n\n"

                    "Correct Summary:\n"
                    "The evidence describes Elon Musk as an entrepreneur and business magnate. "
                    "He founded the aerospace manufacturer SpaceX and also co-founded Tesla, "
                    "a company focused on electric vehicles. The relationships indicate that "
                    "Elon Musk played a founding role in both organizations.\n\n"

                    "Example 2\n"
                    "Graph Evidence:\n"
                    "NODE: Python — a programming language widely used in data science.\n"
                    "NODE: TensorFlow — a machine learning framework.\n"
                    "RELATIONSHIP: Python USED_FOR Machine Learning.\n"
                    "RELATIONSHIP: TensorFlow IMPLEMENTED_IN Python.\n"
                    "PATH: Python → USED_FOR → Machine Learning → USES → TensorFlow.\n\n"

                    "Correct Summary:\n"
                    "The evidence describes Python as a programming language widely used in "
                    "data science and machine learning. Python is associated with machine "
                    "learning tasks, and the framework TensorFlow is implemented in Python. "
                    "The reasoning path illustrates that Python is used for machine learning "
                    "workflows in which frameworks such as TensorFlow play a role.\n\n"

                    "Remember: your job is to summarize the graph evidence in detail. "
                    "Do NOT answer any question and do NOT invent information."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"""
                Graph Evidence:
                {text}

                Write a detailed summary of the evidence.
                """,
            ),
        ]

        response = self._llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()
    
    def _count_tokens(self,text):
        return len(self._llm._model.tokenize(text.encode("utf-8")))

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

    def _summarize_evidence(self, subgraph, query):

        chunks = self._build_evidence_chunks(subgraph)
        chunks = self._rank_chunks_by_query(chunks, query)
        if not chunks:
            return ""

        print(f"len chunks: {len(chunks)}")

        evidence_summary = self.reduce_in_batches(chunks,self._token_budget)
        print(f"evidence summary: {evidence_summary}")
        return evidence_summary.strip()

    def _generate_answer(self, query, summary):
        
        if not summary.strip():
            return "Insufficient information to answer based on the available summary."

        prompt = f"""
        You are a deterministic answer verifier and synthesizer.

        You receive:
        - A user query
        - An evidence summary extracted from a knowledge graph

        ### YOUR TASK
        Produce ONE final factual answer.

        ### STRICT RULES
        - The final answer must be based ONLY on the evidence summary.
        - The summary is authoritative evidence.
        - You MUST verify that the answer is explicitly supported by the summary.
        - If the information is incomplete or slightly imprecise but explicitly stated in the summary,
        you MAY refine the answer using ONLY information explicitly present in the summary.
        - Do NOT invent or infer information not explicitly written.
        - If the summary does not explicitly support a correct answer → output EXACTLY:
        Insufficient information to answer based on the available summary.
        - Output ONLY the final factual answer.
        - No explanation. No reasoning.

        ---
        ### EXAMPLE 1
        User Query:
        Who directed Inception?

        Evidence Summary:
        The film Inception was directed by Christopher Nolan and released in 2010.

        Final Answer:
        Christopher Nolan
        ---
        ### EXAMPLE 2
        User Query:
        When was the Declaration of Independence signed?

        Evidence Summary:
        The United States Declaration of Independence was adopted on July 4, 1776 in Philadelphia.

        Final Answer:
        July 4, 1776
        ---
        ### EXAMPLE 3
        User Query:
        What is the capital of Atlantis?

        Evidence Summary:
        Atlantis is a fictional island mentioned in Plato's works.

        Final Answer:
        Insufficient information to answer based on the available summary.
        ---

        Now process the real input.

        User Query:
        {query}

        Evidence Summary:
        {summary}

        Final Answer:
        """
        prompt = PromptTemplate(prompt)
        response = self._llm.predict(prompt, query=query, summary=summary)
        response = re.split(r'("""|\'\'\'|```|#|markdown)', response)[0]
        return response.strip()

    def query(self, query):
        
        # entities = self._extract_entities(query)
        # print(f"entities: {entities}\n")
        entities = []
        most_similar_nodes = self._link_entities(entities, query)
        if not most_similar_nodes:
            print(f"No similar nodes found\n")
            return "Insufficient information to answer based on the available summary.", ""
        print(f"most_similar_nodes : {most_similar_nodes}")
        subgraph = self._expand_subgraph(most_similar_nodes, query)

        print("\n[DEBUG] FINAL SUBGRAPH")
        print("nodes:", len(subgraph["nodes"]))
        print("edges:", len(subgraph["edges"]))
        print("paths:", len(subgraph["paths"]))

        # print(f"subgraph: {subgraph}\n")
        summary = self._summarize_evidence(subgraph, query)
        # print(f"summary: {summary}\n")
        answer = self._generate_answer(query, summary)
        # print(f"anwser: {answer}\n")

        return answer, summary