#!/usr/bin/env python
# coding: utf-8

# # Dependencies

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import sys
import os

#Path to project root
PROJECT_ROOT = os.path.abspath("/home/plas_vasileiadis/GraphRAG")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("Project root added:", PROJECT_ROOT, flush=True)


# # Load and Preprocess Data

# In[ ]:


from pipeline.data.load_and_process_dataset import load_to_nodes_HotPotQA, load_to_nodes_MuSiQue, load_to_nodes_2WikiMultiHopQA

# nodes, questions, answers = load_to_nodes_HotPotQA(type="all", num=200)
nodes, questions, answers = load_to_nodes_MuSiQue(num=200)
# nodes, questions, answers = load_to_nodes_2WikiMultiHopQA(q_type="all", num=200)

# # Configure the LLM, Prompt, and GraphRAG Extractor

# ## Configure the LLM

# In[4]:


from llama_index.llms.llama_cpp import LlamaCPP
from llama_cpp import Llama
 
# llm_path = "/home/plas_vasileiadis/GraphRAG/pipeline/llm/local_models/qwen2_7B_instruct_gguf_Q8/qwen2-7b-instruct-q8_0.gguf"
llm_path = "/home/plas_vasileiadis/GraphRAG/pipeline/llm/local_models/Mistral_7B_instruct_gguf_Q8/Mistral-7B-Instruct-v0.3-Q8_0.gguf"
embedder_path = "/home/plas_vasileiadis/GraphRAG/pipeline/llm/local_embedings/bge-m3-q8_0.gguf/bge-m3-q8_0.gguf"
llm = LlamaCPP(
    model_path=llm_path,
    temperature=0.0,
    max_new_tokens=4096,
    context_window=13000,
    model_kwargs={
        "n_gpu_layers": 45,
        "n_ctx": 13000,
        "n_threads": 8,
    },
    verbose=False,
)
embedder = Llama(
    model_path=embedder_path,
    embedding=True,
    n_ctx=2048,
    n_gpu_layers=25,
    n_threads=8,
    verbose=False,
)

# ## Embedding the Chunks

# In[5]:


# from pipeline.RAG.rag_retrieval import TextChunk, embed_chunks

# chunks = []
# for i, node in enumerate(nodes):
#     text = node.text
#     doc_id = node.metadata.get("doc_id", node.node_id)
#     chunks.append(TextChunk(chunk_id=i,text=text,doc_id=doc_id))

# embed_chunks(chunks, embedder)

# ## Entities and Relationships extraction

# ### GraphRAGExtractor Class

# In[6]:


from pipeline.extraction.extractor import GraphRAGExtractor
from pipeline.extraction.prompts import KG_ENTITY_EXTRACT_TMPL, KG_RELATION_EXTRACT_TMPL                                                                                                                                                                                                                                                                                
from pipeline.extraction.parse_functions import parse_en_fn, parse_rel_fn

#instance of class GraphRAGExtractor
kg_extractor = GraphRAGExtractor(
    llm=llm,
    en_extract_prompt=KG_ENTITY_EXTRACT_TMPL,
    rel_extract_prompt=KG_RELATION_EXTRACT_TMPL,  
    parse_en_fn=parse_en_fn,
    parse_rel_fn=parse_rel_fn,
    max_paths_per_chunk=50,
)


# # Extraction ablation study

# In[ ]:


# from pipeline.debug.ablation_extraction import extraction

# extraction(kg_extractor, embedder)


# # Extracting the KG Entities and Relationships

# In[ ]:


subgraphs = []
for node in nodes:
    # print(f"NODE: {node}\n")
    #The triples here are of type node
    triples = kg_extractor([node], show_progress = True)
    subgraphs.append(triples)


# In[ ]:


# from pipeline.debug.debug_functions import debug_graph

# all_nodes = []
# all_relations = []

# for subgraph in subgraphs:
#     for text_node in subgraph:
#         all_nodes.extend(
#             text_node.metadata.get("nodes", [])
#         )
#         all_relations.extend(
#             text_node.metadata.get("relations", [])
#         )

# print(f"Total entities (raw): {len(all_nodes)}")
# print(f"Total relations (raw): {len(all_relations)}")
# print(debug_graph(all_nodes, all_relations))


# ## Entities and Relationships Aggregation

# In[ ]:


from pipeline.aggregation.aggregator import GraphAggregation
# unique_entities, unique_edges = aggregate(subgraphs)
aggregator = GraphAggregation(similarity_threshold=0.8)
unique_entities, unique_edges = aggregator.aggregate(subgraphs)


# In[ ]:


# print(f"Total entities after aggregation: {len(unique_entities)}")
# print(f"Total relations after aggregation: {len(unique_edges)}")
# print(debug_graph(unique_entities.values(), unique_edges, extr=0))


# In[ ]:


# unique_entities_name = list(unique_entities.keys())
# print(unique_entities_name)


# In[ ]:


# print(unique_edges)


# ## Entities Clustering

# In[ ]:


from pipeline.clustering.cluster_entities import EntityClusterer

entity_clusterer = EntityClusterer(llm)
final_entities = entity_clusterer.cluster(unique_entities)


# In[ ]:


# for label, entity_obj in final_entities.items():    
#     print(f"entity_obj: {entity_obj}")


# In[ ]:


# print(f"Total entities after clustering: {len(final_entities)}")
# print(f"Total relations before clustering: {len(unique_edges)}")
# print(debug_graph(final_entities.values(), unique_edges, extr=0))


# ## Relationships Clustering

# In[ ]:


from pipeline.clustering.cluster_relations import RelationClusterer

relation_clusterer = RelationClusterer(unique_edges, llm)
final_edges = relation_clusterer.cluster()


# In[ ]:


# for cluster_name, cluster_data in final_edges.items():
#     print(f"\n=== CLUSTER: {cluster_name} ===")
#     for edge in cluster_data["cluster_members"]:
#         print(" -", edge)


# In[ ]:


# print(f"Total entities after clustering: {len(final_entities)}")
# print(f"Total relations after clustering: {len(final_edges)}")
# print(debug_graph(final_entities.values(), unique_edges, extr=0))


# ## Knowledge Graph Creation

# In[ ]:


from pipeline.graph.graph_store import final_graph_node

g_node = final_graph_node(final_entities, final_edges)


# In[ ]:


# for entity in g_node.metadata["KG_NODES_KEY"]:
#     print(f"entity: {entity}\n")


# In[ ]:


# i=0
# for rel in g_node.metadata["KG_RELATIONS_KEY"]:
    
#     description=rel.properties["relationship_description"]
#     print(f"{i, rel.source_id, rel.label, rel.target_id,  description}")
#     i += 1
# # print(g_node.metadata["KG_RELATIONS_KEY"])


# In[ ]:


# def call_debug_graph_from_gnode(g_node, extr=1):

#     graph_nodes = g_node.metadata.get("KG_NODES_KEY", [])
#     graph_relations = g_node.metadata.get("KG_RELATIONS_KEY", [])

#     print("\n================ DEBUG FINAL GRAPH NODE ================\n")
#     print(f"Entities in g_node: {len(graph_nodes)}")
#     print(f"Relations in g_node: {len(graph_relations)}")

#     assert isinstance(graph_nodes, list), "KG_NODES_KEY is not a list"
#     assert isinstance(graph_relations, list), "KG_RELATIONS_KEY is not a list"

#     debug_graph(graph_nodes=graph_nodes,graph_relations=graph_relations,extr=extr)

# g_node = final_graph_node(final_entities, final_edges)

# call_debug_graph_from_gnode(g_node,extr=1)


# # Saving the Knowledge Graph

# In[ ]:


import json

graph_data = {
    "entities": [
        {
            "id": node.id,
            "name": node.name,
            "label": node.label,
            "properties": node.properties,
        }
        for node in g_node.metadata["KG_NODES_KEY"]
    ],
    "relations": [
        {
            "id": rel.id,
            "label": rel.label,
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "properties": rel.properties,
        }
        for rel in g_node.metadata["KG_RELATIONS_KEY"]
    ]
}

with open("saved_graph_200_Musique_mistral.json", "w") as f:
    json.dump(graph_data, f)


# # Loading the saved Knowledge Graph

# In[ ]:


from llama_index.core.graph_stores.types import EntityNode, Relation
import json
from llama_index.core import PropertyGraphIndex
from pipeline.graph.graph_store import GraphRAGStore

with open("saved_graph_200_Musique_mistral.json") as f:
    graph_data = json.load(f)

store = GraphRAGStore(llm=llm, embedder=embedder)

for node_data in graph_data["entities"]:
    node = EntityNode(
        name=node_data["name"],
        label=node_data["label"],
        properties=node_data["properties"]
    )
    store.graph.add_node(node)

for rel_data in graph_data["relations"]:
    rel = Relation(
        label=rel_data["label"],
        source_id=rel_data["source_id"],
        target_id=rel_data["target_id"],
        properties=rel_data["properties"]
    )
    store.graph.add_relation(rel)

index = PropertyGraphIndex(
    nodes=[],
    property_graph_store=store,
    show_progress=True,
    llm=llm
)


# ## Creating the Graph Index item

# In[ ]:


# from llama_index.core import PropertyGraphIndex
# from pipeline.graph.graph_store import GraphRAGStore

# #I manually add all the edges and nodes to the Graph store
# store = GraphRAGStore(llm=llm, embedder=embedder)
# for node in g_node.metadata["KG_NODES_KEY"]:
#     store.graph.add_node(node)
# for rel in g_node.metadata["KG_RELATIONS_KEY"]:
#     store.graph.add_relation(rel)

# #And create the graph index
# index = PropertyGraphIndex(
#     nodes=[],
#     property_graph_store=store,
#     show_progress=True,
#     llm=llm
# )


# # Querying the Graph

# In[8]:


import re
from llama_index.core.llms import ChatMessage

def final_synthesis(llm, query, community_answer, community_summary, graph_answer, graph_summary):

    prompt_system = (
        "You are a deterministic answer verifier and synthesizer.\n\n"
        "You receive:\n"
        "- A user query\n"
        "- An answer from community-based reasoning\n"
        "- An answer from graph-based reasoning\n"
        "- Their corresponding supporting summaries\n\n"

        "### YOUR TASK\n"
        "Produce ONE final factual answer.\n\n"

        "### STRICT RULES\n"
        "- Each answer must be verified against its corresponding summary.\n"
        "- Summaries are authoritative evidence.\n"

        "- If an answer is unsupported by its summary, discard it.\n"

        "- If an answer is missing or incomplete BUT the correct information\n"
        "  is explicitly present inside its summary, you MAY extract it.\n"

        "### MULTI-SUMMARY REASONING (IMPORTANT)\n"
        "- You MAY combine information from BOTH summaries ONLY IF:\n"
        "  • They refer to the SAME entities\n"
        "  • OR they describe CONNECTED facts needed to answer the query\n"
        "- This is required for multi-hop reasoning.\n"
        "- Do NOT combine unrelated summaries.\n"

        "- Do NOT invent information beyond what is written.\n"

        "- If both answers are valid and consistent → choose the more precise one.\n"

        "- If both answers are partially correct → you MAY synthesize them\n"
        "  ONLY if both summaries explicitly support the synthesis.\n"

        "- If neither summary contains enough explicit information → output EXACTLY:\n"
        "Insufficient information to answer based on the available summaries.\n"

        "- Output ONLY the final factual answer.\n"
        "- No explanation. No reasoning.\n\n"

        "### EXAMPLE (Multi-hop reasoning)\n\n"

        "User Query:\n"
        "Where was the director of Inception born?\n\n"

        "Community Summary:\n"
        "Inception is a film directed by Christopher Nolan.\n\n"

        "Graph Summary:\n"
        "Christopher Nolan was born in London.\n\n"

        "Final Answer:\n"
        "London\n\n"
    )

    prompt_user = (
        f"USER QUERY:\n{query}\n\n"

        f"GRAPH ANSWER:\n{graph_answer}\n\n"
        f"GRAPH SUMMARY:\n{graph_summary}\n\n"

        f"COMMUNITY ANSWER:\n{community_answer}\n\n"
        f"COMMUNITY SUMMARY:\n{community_summary}\n\n"

        "### FINAL ANSWER:"
    )

    messages = [
        ChatMessage(role="system", content=prompt_system),
        ChatMessage(role="user", content=prompt_user)
    ]

    response = str(llm.chat(messages)).strip()
    response = re.sub(r"^assistant:\s*", "", response)
    response = re.split(r'("""|\'\'\'|```|#|Explanation:|Reasoning:|Because|Note:)',response,flags=re.IGNORECASE)[0].strip()

    if not response:
        return "Insufficient information to answer based on the available summaries."

    return response



# In[9]:


from pipeline.query.query_engine import GraphRAGQueryEngine
from pipeline.RAG.rag_retrieval import RAGRetriever, rag_answer_query
from pipeline.query.graph_search import GraphSearch

query_engine = GraphRAGQueryEngine(graph_store=index.property_graph_store, llm=llm, embedder=embedder,chunk_token_limit=1, final_context_token_limit=11000, 
                                   shuffle_seed=42)
# rag_retriever = RAGRetriever(chunks)

graph_search = GraphSearch(graph_store=index.property_graph_store, llm=llm, embedder=embedder)


# In[10]:


llms_answers = []
i = 0;

for question in questions:
    print(f"Question[{i}]:{question}\n")
    graph_answer, graph_summary = graph_search.query(question)
    # print(f"graph answer: {graph_answer}\n")
    com_answer, com_summary = query_engine.query(question)
    # print(f"com_answer: {com_answer}\n")
    # print(f"com_summary: {com_summary}\n")
    print(f"graph answer: {graph_answer} | graph summary: {graph_summary}")
    print(f"com answer: {com_answer} | com summary: {com_summary}")
    final_answer = final_synthesis(llm,question,com_answer, com_summary,graph_answer, graph_summary)

    # llms_answers.append(graph_answer)
    llms_answers.append(final_answer)

    print(f"Llm's response[{i}]:{final_answer}\n")
    print(f"Expected answer[{i}]:{answers[i]}\n\n")
    
    i += 1

# for question in questions:
#     print(f"Question[{i}]:{question}\n")
#     top_graphrag_sim, response = query_engine.query(question)
#     rag_results = rag_retriever.retrieve(question, embedder, top_k=2)
#     top_rag_sim = rag_results[0][1]
#     print(f"GraphRAG top sim: {top_graphrag_sim} | RAG top sim: {top_rag_sim}\n")
#     # if top_graphrag_sim > top_rag_sim and "Insufficient information to answer based on the available summaries." not in response:
#     if 1 == 1:
#         llms_answers.append(response)
#         print(f"GraphRAG response[{i}]:{response}\n")
#         answer = rag_answer_query(question, rag_results, llm)
#         print(f"Would be RAG response: {answer}\n")
#         print(f"Expected answer[{i}]:{answers[i]}\n\n\n")
#     else:
#         answer = rag_answer_query(question, rag_results, llm)
#         print(f"RAG RESPONSE: {answer}\n")
#         print(f"Would be GraphRAG response: {response}\n")
#         llms_answers.append(answer)
#         print(f"Llm's response[{i}]:{answer}\n")
#         print(f"Expected answer[{i}]:{answers[i]}\n\n\n")    

#     i += 1


# Evaluation

# In[11]:


from GraphRAG.pipeline.evaluation.evaluate import f1_score, exact_match, llm_evaluation, contains_answer, hit_at_kwords

em_scores = [exact_match(la, a) for la, a in zip(llms_answers, answers)]
f1_scores = [f1_score(la, a) for la, a in zip(llms_answers, answers)]
llm_scores = llm_evaluation(llm, llms_answers, answers, questions)
contains_answer_scores = [contains_answer(la, a) for la, a in zip(llms_answers, answers)]
hit_at_k_words = [hit_at_kwords(la, a, 15) for la, a in zip(llms_answers, answers)]

print("Exact Match:", sum(em_scores)/len(em_scores))
print("F1:", sum(f1_scores)/len(f1_scores))
print(f"llm's score: {llm_scores}")
print(f"Contains answer score:", sum(contains_answer_scores)/len(contains_answer_scores))
print(f"Hit@kwords:", sum(hit_at_k_words)/len(hit_at_k_words))

