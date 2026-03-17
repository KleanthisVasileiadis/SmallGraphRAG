import nest_asyncio
nest_asyncio.apply()
from typing import List, Callable, Optional, Union
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode
from pipeline.extraction.prompts import KG_ENTITY_EXTRACT_TMPL, KG_RELATION_EXTRACT_TMPL
from typing import Any, List
import asyncio
import re

class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.
 
    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.
 
    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """
 
    llm: LLM
    en_extract_prompt: PromptTemplate
    rel_extract_prompt: PromptTemplate
    parse_en_fn: Callable
    parse_rel_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
 
    def __init__(
        self,
        llm: Optional[LLM] = None,
        en_extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        rel_extract_prompt: Optional[Union[str, PromptTemplate]] = None,

        parse_en_fn: Callable = default_parse_triplets_fn,
        parse_rel_fn: Callable = default_parse_triplets_fn,
        # max_paths_per_chunk: int = 10,
        max_paths_per_chunk: int = 50,
        num_workers: int = 4,) -> None:
        """Init params."""
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        #Setting the embeding model to a local one
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        if isinstance(en_extract_prompt, str):
            en_extract_prompt = PromptTemplate(en_extract_prompt)
        
        if isinstance(rel_extract_prompt, str):
            rel_extract_prompt = PromptTemplate(rel_extract_prompt)
        
        #Initialize parameters to the superclass TransformComponent
        super().__init__(
            llm=llm or Settings.llm,
            en_extract_prompt=en_extract_prompt or KG_ENTITY_EXTRACT_TMPL,
            rel_extract_prompt=rel_extract_prompt or KG_RELATION_EXTRACT_TMPL,
            parse_en_fn=parse_en_fn,
            parse_rel_fn=parse_rel_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )
 
    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    #Call the class GraphRAGExtractor as a function and run async the acall() function.
    def __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    #Extract triples from a node
    async def _aextract(self, node: BaseNode) -> BaseNode:

        ENTITY_BATCH_SIZE = 25
        """Extract triples from a node."""
        assert hasattr(node, "text")
        #Node's text, i.e. chunked document
        text = node.text
        try:
            # print(f"node_id: {node.node_id}\n")
            # patience = 0 
            # max_iter = 2 
            # entities = [] 
            # #Entities Extraction 
            # while patience < max_iter: 
            #     llm_en_response = await self.llm.apredict( 
            #         self.en_extract_prompt, 
            #         text=text, 
            #         max_knowledge_triplets=self.max_paths_per_chunk
            #         ) 
                
            #     entities = self.parse_en_fn(llm_en_response)
            #     if entities: 
            #         break 
            #     patience += 1 
            def entity_in_text(entity_name, text):
                pattern = rf'\b{re.escape(entity_name)}\b'
                return re.search(pattern, text) is not None

            sentences = re.split(r'(?<=[.!?])\s+', text)
                
            entities_dict = {}
            relations_dict = {} 
            entity_merge_count = 0
            relation_merge_count = 0
            unknown_entity_relations = 0
            raw_relation_count = 0
            for i in range(len(sentences)):
                                
                window = sentences[i:i+2]
                window_text = " ".join(window).strip()

                if not window_text:
                    continue

                # print(f"\n=== WINDOW {i} ===")
                # print(window_text)

                #Entity extraction
                patience = 0
                max_iter = 2
                local_entities = []
                while patience < max_iter:
                    llm_en_response = await self.llm.apredict(
                        self.en_extract_prompt,
                        text=window_text,
                        max_knowledge_triplets=self.max_paths_per_chunk,
                    )

                    parsed = self.parse_en_fn(llm_en_response)
                    if parsed:
                        local_entities = parsed
                        break

                    patience += 1

                if not local_entities:
                    continue

                #Local dedup inside window
                local_seen = set()
                cleaned_local_entities = []
                for name, etype, desc in local_entities:
                    key = (name.strip(), etype.strip())
                    if key not in local_seen:
                        local_seen.add(key)
                        cleaned_local_entities.append((name.strip(), etype.strip(), desc.strip()))

                print(f"Local unique entities: {len(cleaned_local_entities)}")

                #Global entity aggregation
                for name, etype, desc in cleaned_local_entities:
                    key = (name, etype)

                    if key not in entities_dict:
                        entities_dict[key] = set()

                    before_size = len(entities_dict[key])
                    entities_dict[key].add(desc)
                    after_size = len(entities_dict[key])

                    if after_size > before_size:
                        if before_size > 0:
                            entity_merge_count += 1

                #Need at least 2 entities for relation extraction
                if len(cleaned_local_entities) < 2:
                    continue

                #Relation extraction
                blocks = []
                for name, etype, desc in cleaned_local_entities:
                    blocks.append(
                        f"entity_name: {name}\n"
                        f"entity_type: {etype}\n"
                        f"entity_description: {desc}\n"
                    )

                entities_str = "\n".join(blocks)

                llm_rel_response = await self.llm.apredict(
                    self.rel_extract_prompt,
                    text=window_text,
                    entities=entities_str,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )

                parsed_relations = self.parse_rel_fn(llm_rel_response)
                if not parsed_relations:
                    continue
                    
                # for relation in parsed_relations:
                #     print(f"relation: {relation}\n")

                # print(f"Raw relations extracted: {len(parsed_relations)}")

                raw_relation_count += len(parsed_relations)

                #Global relation aggregation
                for s, o, r, d in parsed_relations:

                    s_clean = s.strip()
                    o_clean = o.strip()
                    r_clean = r.strip()
                    d_clean = d.strip()

                    key = (s_clean, o_clean, r_clean)

                    if key not in relations_dict:
                        relations_dict[key] = set()

                    before_size = len(relations_dict[key])
                    relations_dict[key].add(d_clean)
                    after_size = len(relations_dict[key])

                    if after_size > before_size:
                        if before_size > 0:
                            relation_merge_count += 1


            #Final validation
            valid_entity_names = set(name for (name, _) in entities_dict.keys())

            final_relations = []

            for (s, o, r), desc_set in relations_dict.items():

                if s not in valid_entity_names or o not in valid_entity_names:
                    unknown_entity_relations += 1
                    # print(f"Relation with unknown entity removed:")
                    # print(f"   ({s}) -[{r}]-> ({o})")
                    continue

                combined_desc = " | ".join(sorted(desc_set))
                final_relations.append((s, o, r, combined_desc))

            #Build final entity list
            entities = []
            for (name, etype), desc_set in entities_dict.items():
                combined_desc = " | ".join(sorted(desc_set))
                entities.append((name, etype, combined_desc))

            relations = final_relations


            #Debug summary
            print("\n================ FINAL SUMMARY ================")
            print(f"Total unique entities: {len(entities)}")
            print(f"Entity description merges: {entity_merge_count}")

            print(f"\nTotal raw relations extracted: {raw_relation_count}")
            print(f"Total unique relation triples: {len(relations)}")
            print(f"Relation description merges: {relation_merge_count}")

            print(f"Relations removed: {unknown_entity_relations}")
            print("================================================\n")

        except ValueError:
            entities = []
            relations = []
            
        #String normilization function
        def canon(string):
            if not string:
                return ""
            string = string.strip().lower()
            if (string.startswith('"') and string.endswith('"')) or (string.startswith("'") and string.endswith("'")):
                string = string[1:-1]
            return string.strip()

        #Removes all previous nodes and relations metadata and makes a new copy
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        #For every entity, it creates a new node and it appends it to the existing nodes list
        for entity, entity_type, description in entities:
            metadata["entity_description"] = description
            entity_node = EntityNode(name=canon(entity), label=entity_type, properties=metadata)
            existing_nodes.append(entity_node)
        
        #Another metadata copy
        metadata = node.metadata.copy()
        #For every relationship, it creates the subject and object node, creates their relationship 
        #and appends everything to their corresponding lists
        for subj, obj, rel, description in relations:
            subj_node = EntityNode(name=canon(subj), properties=metadata)
            obj_node = EntityNode(name=canon(obj), properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=canon(rel),
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )
            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)
        #Puts back the lists into their corresponding metadata and returns the node
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    #For every node it async calls aextract() function
    async def acall(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))
        #Waits for all the jobs, that were called above, to finish.
        return await run_jobs(jobs,workers=self.num_workers,show_progress=show_progress,desc="Extracting paths from text")