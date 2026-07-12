import re
from typing import Any

entity_pattern = re.compile(
    r'entity_name:\s*(.*?)\s*entity_type:\s*(.*?)\s*entity_description:\s*(.*?)(?=\n\s*entity_name:|\n\s*RELATIONSHIPS:|\n\s*\.)',
    re.DOTALL
)

relationship_pattern = re.compile(
    r'source_entity:\s*(.*?)\s*'
    r'target_entity:\s*(.*?)\s*'
    r'relation:\s*(.*?)\s*'
    r'relationship_description:\s*(.*?)(?=(?:\nsource_entity:)|(?:^\s*\.\s*$)|$)',
    re.DOTALL | re.MULTILINE
)

def parse_en_fn(response_str: str) -> Any:
    return re.findall(entity_pattern, response_str)

def parse_rel_fn(response_str: str) -> Any:
    return re.findall(relationship_pattern, response_str)