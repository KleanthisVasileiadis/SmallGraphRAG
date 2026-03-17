import re
from typing import Any

#Captures pattern in the form: entity_name:{whitespace(s)}(text){whitespace(s)}entity_type:{whitespace(s)}(text){whitespace(s)}entity_description:
#{whitespace(s)}(text)
#and it ends when it encounters either "entity_name", "RELATIONSHIPS" or "\n.". 
# entity_pattern = re.compile(
#     r'entity_name:\s*(.+?)\s*entity_type:\s*(.+?)\s*entity_description:\s*(.+?)(?=\nentity_name:|\nRELATIONSHIPS:|\n\.)',
#     re.DOTALL
# )

entity_pattern = re.compile(
    r'entity_name:\s*(.*?)\s*entity_type:\s*(.*?)\s*entity_description:\s*(.*?)(?=\n\s*entity_name:|\n\s*RELATIONSHIPS:|\n\s*\.)',
    re.DOTALL
)

#Captures pattern in the form: source_entity:{whitespace(s)}(text){whitespace(s)}target_entity:{whitespace(s)}(text){whitespace(s)}relation:
#{whitespace(s)}(text){whitespace(s)}relationship_description:{whitespace(s)}(text)
#and it ends when it encounters either "source_entity" or "\n.".
# relationship_pattern = re.compile(
#     r'source_entity:\s*(.+?)\s*target_entity:\s*(.+?)\s*relation:\s*(.+?)\s*relationship_description:\s*(.+?)(?=(?:\r?\nsource_entity:)|(?:\r?\n\.)|$)',
#     re.DOTALL
# )

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