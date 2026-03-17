import json
import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def load_documents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    gold_by_doc = {}
    for sample in data:
        doc = Document(text=sample["text"],metadata={"doc_id": sample["id"]})
        documents.append(doc)
        gold_by_doc[sample["id"]] = {"entities": sample["entities"],"relations": sample.get("relationships", [])}

    return documents, gold_by_doc

def convert_to_nodes(documents):
    splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"nodes: {len(nodes)}\n")
    return nodes

def gold_entities_for_node(node, gold_by_doc):
    doc_id = node.metadata["doc_id"]
    node_text = node.text.lower()
    gold_entities = []
    for ent in gold_by_doc[doc_id]["entities"]:
        if ent["text"].lower() in node_text:
            gold_entities.append(ent["text"].lower().strip())

    return set(gold_entities)

def gold_relations_for_node(node, gold_by_doc):
    doc_id = node.metadata["doc_id"]
    node_text = node.text.lower()
    gold_relations = set()
    for rel in gold_by_doc[doc_id]["relations"]:
        s = rel["source_entity"].lower().strip()
        r = rel["relation"].lower().strip()
        t = rel["target_entity"].lower().strip()
        if s in node_text and t in node_text:
            gold_relations.add((s, r, t))

    return gold_relations

def predicted_entities_from_node(text_node):
    predicted = set()
    entity_nodes = text_node.metadata.get("nodes", [])
    for ent in entity_nodes:
        if hasattr(ent, "name"):
            predicted.add(ent.name.lower().strip())

    return predicted

def predicted_relations_from_node(text_node):
    predicted = set()
    relations = text_node.metadata.get("relations", [])
    for rel in relations:
        s = rel.source_id.lower().strip()
        r = rel.label.lower().strip()
        t = rel.target_id.lower().strip()
        predicted.add((s, r, t))

    return predicted

def gold_coverage(predicted, gold):

    if not gold:
        return 1.0, 1

    covered = len(predicted & gold)
    coverage_ratio = covered / len(gold)
    binary_containment = 1 if covered == len(gold) else 0
    return coverage_ratio, binary_containment

def evaluate_sets(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def embed(texts, embedder):

    def normalize(vec: np.ndarray):
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return None
        return vec / norm

    embs = []
    for t in texts:
        try:
            e = embedder.embed(t)
            e = np.array(e, dtype=np.float32)
            e = normalize(e)
            embs.append(e)
        except RuntimeError:
            continue
    return embs

#If at least one gold matches the predicted with a similarity higher than threshold it returns 1
def semantic_gold_coverage(predicted ,gold, embedder, threshold: float = 0.8):

    if not gold:
        return 1.0, 1

    gold_list = [g for g in gold if isinstance(g, str) and g.strip()]
    pred_list = [p for p in predicted if isinstance(p, str) and p.strip()]

    if not gold_list or not pred_list:
        return 0.0, 0

    gold_embs = embed(gold_list, embedder)
    pred_embs = embed(pred_list, embedder)

    if not gold_embs or not pred_embs:
        return 0.0, 0

    matched = 0
    for g_emb in gold_embs:
        sims = [float(np.dot(g_emb, p_emb)) for p_emb in pred_embs]
        if max(sims) >= threshold:
            matched += 1

    coverage_ratio = matched / len(gold_embs)
    binary_hit = 1 if matched == len(gold_embs) else 0

    return coverage_ratio, binary_hit

def extraction(kg_extractor, embedder):
    dataset_paths = [
        "/home/plas_vasileiadis/GraphRAG/pipeline/data/ablation_datasets/extraction/entity_relation_extraction_gpt.json",
        "/home/plas_vasileiadis/GraphRAG/pipeline/data/ablation_datasets/extraction/entity_relation_extraction_hard_short.json",
        "/home/plas_vasileiadis/GraphRAG/pipeline/data/ablation_datasets/extraction/entity_relation_extraction_hard_long.json"
    ]

    all_results = []
    for path in dataset_paths:
        print(f"\nProcessing dataset: {path}")

        documents, gold_by_doc = load_documents(path)

        #Convert to nodes
        nodes = convert_to_nodes(documents)
        print(f"nodes: {nodes}")

        extracted_nodes = kg_extractor(nodes)

        ent_p, ent_r, ent_f1 = [], [], []
        rel_p, rel_r, rel_f1 = [], [], []
        ent_cov, ent_bin = [], []
        rel_cov, rel_bin = [], []
        ent_sem_cov, ent_sem_bin = [], []
        rel_sem_cov, rel_sem_bin = [], []
        i = 0
        for original_node, extracted_node in zip(nodes, extracted_nodes):
            #Gold and predicted
            gold_entities = gold_entities_for_node(original_node, gold_by_doc)
            gold_relations = gold_relations_for_node(original_node, gold_by_doc)
            pred_entities = predicted_entities_from_node(extracted_node)
            pred_relations = predicted_relations_from_node(extracted_node)
            # if i % 50 == 0:
            if i == i:
                print(f"entities gold: {gold_entities}")
                print(f"entities pred: {pred_entities}")
                print(f"relations gold: {gold_relations}")
                print(f"relations pred: {pred_relations}")

            #Valuation
            p, r, f1 = evaluate_sets(pred_entities, gold_entities)
            ent_p.append(p)
            ent_r.append(r)
            ent_f1.append(f1)
            p, r, f1 = evaluate_sets(pred_relations, gold_relations)
            rel_p.append(p)
            rel_r.append(r)
            rel_f1.append(f1)
            cov, bin_hit = gold_coverage(pred_entities, gold_entities)
            ent_cov.append(cov)
            ent_bin.append(bin_hit)
            cov, bin_hit = gold_coverage(pred_relations, gold_relations)
            rel_cov.append(cov)
            rel_bin.append(bin_hit)
            sem_cov, sem_bin = semantic_gold_coverage(pred_entities,gold_entities,embedder=embedder,threshold=0.5)
            ent_sem_cov.append(sem_cov)
            ent_sem_bin.append(sem_bin)
            sem_cov, sem_bin = semantic_gold_coverage(pred_relations,gold_relations,embedder=embedder,threshold=0.5)
            rel_sem_cov.append(sem_cov)
            rel_sem_bin.append(sem_bin)
            i += 1

        print("\nENTITY EXTRACTION")
        print(f"Precision: {sum(ent_p)/len(ent_p):.3f} | "
            f"Recall: {sum(ent_r)/len(ent_r):.3f} | "
            f"F1: {sum(ent_f1)/len(ent_f1):.3f}"
        )
        print(
            f"Gold Coverage: {sum(ent_cov)/len(ent_cov):.3f} | "
            f"All-Gold-Hit Rate: {sum(ent_bin)/len(ent_bin):.3f}"
        )
        print(
            f"Semantic Gold Coverage: {sum(ent_sem_cov)/len(ent_sem_cov):.3f} | "
            f"Semantic All-Gold-Hit Rate: {sum(ent_sem_bin)/len(ent_sem_bin):.3f}"
        )
        print("\nRELATION EXTRACTION")
        print(
            f"Precision: {sum(rel_p)/len(rel_p):.3f} | "
            f"Recall: {sum(rel_r)/len(rel_r):.3f} | "
            f"F1: {sum(rel_f1)/len(rel_f1):.3f}"
        )
        print(
            f"Gold Coverage: {sum(rel_cov)/len(rel_cov):.3f} | "
            f"All-Gold-Hit Rate: {sum(rel_bin)/len(rel_bin):.3f}"
        )
        print(
            f"Semantic Gold Coverage: {sum(rel_sem_cov)/len(rel_sem_cov):.3f} | "
            f"Semantic All-Gold-Hit Rate: {sum(rel_sem_bin)/len(rel_sem_bin):.3f}"
        )
        break