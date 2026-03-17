import re
from typing import Any

# cluster_pattern = re.compile(r'^entity:\s*(.+?)\s*(?=$|\nentity:|\n\.)', re.MULTILINE)
cluster_pattern = re.compile(r'^\s*entity:\s*(.+?)\s*(?=$|\n\s*entity:|\n\s*\.)', re.MULTILINE)

edge_cluster_pattern = re.compile(r'^\s*relation:\s*(.+?)\s*(?=$|\n\s*relation:|\n\s*\.)', re.MULTILINE)

response_pattern = re.compile(r'answer\s*:\s*"(yes|no)"', re.IGNORECASE)

label_pattern = re.compile(r'answer\s*:\s*(.+)', re.IGNORECASE)

batch_cluster = re.compile(r"\[\s*((?:yes|no)(?:\s*,\s*(?:yes|no))*)\s*\]?",re.IGNORECASE)

def parse_cluster_fn(response_str: str) -> Any:
    return re.findall(cluster_pattern, response_str)

def parse_edge_cluster_fn(response_str: str) -> Any:
    return re.findall(edge_cluster_pattern, response_str)

def parse_response_fn(response_str: str) -> Any:
    responses = re.findall(response_pattern, response_str) 
    if not responses:
        return "no"
    last_response = responses[-1].lower()
    if last_response == "yes":
        return "yes"
    else: 
        return "no"

def parse_label_fn(response_str: str) -> Any:
    labels = re.findall(label_pattern, response_str)
    label = None
    for l in labels:
        label = l
        break
    return label

def parse_entity_batch_cluster(response_str: str):
    block_pattern = re.compile(r'ENTITIES:\s*(.*?)\n\s*\.',re.DOTALL)
    match = block_pattern.search(response_str)
    if not match:
        return {}

    block = match.group(1)
    line_pattern = re.compile(r'^\s*(.*?)\s*:\s*(.*?)\s*$', re.MULTILINE)

    assignments = {}
    for entity, cluster in line_pattern.findall(block):
        assignments[entity.strip()] = cluster.strip()

    return assignments

def parse_relation_batch_cluster(response_str: str):
    block_pattern = re.compile(r'RELATIONS:\s*(.*?)\n\s*\.',re.DOTALL)
    match = block_pattern.search(response_str)
    if not match:
        return {}

    block = match.group(1)
    line_pattern = re.compile(r'^\s*(.*?)\s*:\s*(.*?)\s*$', re.MULTILINE)

    assignments = {}
    for relation, cluster in line_pattern.findall(block):
        assignments[relation.strip()] = cluster.strip()

    return assignments