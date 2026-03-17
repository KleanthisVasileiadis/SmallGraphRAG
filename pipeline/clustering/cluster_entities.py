from pipeline.clustering.prompts import(
    default_entity_clustering_prompt,
    default_entity_cluster_judge_prompt,
    default_entity_cluster_label_prompt,
    default_remaining_entities_cluster_prompt,
    default_entity_cluster_merge_prompt
)
from pipeline.clustering.parse_functions import(
    parse_cluster_fn as default_parse_cluster_fn,
    parse_response_fn as default_parse_response_fn,
    parse_label_fn as default_parse_label_fn,
    parse_entity_batch_cluster as default_parse_batch_cluster
)
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.prompts import PromptTemplate

class EntityClusterer:
    def __init__(self, llm, entity_clustering_prompt=default_entity_clustering_prompt, entity_cluster_judge_prompt=default_entity_cluster_judge_prompt,
                 entity_cluster_label_prompt=default_entity_cluster_label_prompt, remaining_entities_cluster_prompt=default_remaining_entities_cluster_prompt, 
                 parse_cluster_fn=default_parse_cluster_fn, cluster_merge_prompt=default_entity_cluster_merge_prompt, parse_response_fn=default_parse_response_fn, 
                 parse_label_fn=default_parse_label_fn, parse_batch_cluster=default_parse_batch_cluster, batch_size=25, max_iter=2, exemplar_size=10, 
                 max_merge_passes=4):
        
        self.llm = llm
        self.entity_clustering_prompt = PromptTemplate(entity_clustering_prompt)
        self.entity_cluster_judge_prompt = PromptTemplate(entity_cluster_judge_prompt)
        self.entity_cluster_label_prompt = PromptTemplate(entity_cluster_label_prompt)
        self.cluster_merge_prompt = PromptTemplate(cluster_merge_prompt)
        self.remaining_entities_cluster_prompt = PromptTemplate(remaining_entities_cluster_prompt)  

        self.parse_cluster_fn = parse_cluster_fn
        self.parse_response_fn = parse_response_fn
        self.parse_label_fn = parse_label_fn
        self.parse_batch_cluster = parse_batch_cluster

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.exemplar_size = exemplar_size
        self.max_merge_passes = max_merge_passes

        self.micro_clusters = []
        self.leftovers = []
        self.clusters = []
        self.unique_entities_name = []

    def cluster_single_batch(self, batch):
        """
        Performs micro-clustering on a batch of entities using LLM
        proposals and validation steps. 
        """
        remaining = batch.copy()
        batch_clusters = []
        patience = 0
        while remaining and patience < self.max_iter:
            
            #Potential cluster
            llm_cluster = self.llm.predict(prompt=self.entity_clustering_prompt, entities=remaining)
            # print(f"llm micro cluster: {llm_cluster}\n")
            llm_cluster = self.parse_cluster_fn(llm_cluster)
            # print(f"llm micro cluster parse: {llm_cluster}\n")
            
            #Cleaning the clustered entities, removing duplicates and filtering non-existing entities
            if llm_cluster:
                llm_cluster = [entity.strip() for entity in llm_cluster if entity.strip()]
                llm_cluster = list(dict.fromkeys(llm_cluster))
                llm_cluster = [entity for entity in llm_cluster if entity in remaining]
            if not llm_cluster or len(llm_cluster) <= 1:
                patience += 1
                continue
            
            #Judging the cluster
            judge = self.llm.predict(prompt=self.entity_cluster_judge_prompt, entities=llm_cluster)
            # print(f"llm response: {judge}\n")
            judge = self.parse_response_fn(judge)
            # print(f"llm response parse: {judge}\n")

            if judge != "yes":
                patience += 1
                continue

            #Labeling the cluster
            label = self.llm.predict(prompt=self.entity_cluster_label_prompt, entities=llm_cluster)
            # print(f"llm label: {label}\n")
            label = self.parse_label_fn(label)
            # print(f"llm label parse: {label}\n")
            
            if label is None:
                patience += 1
                continue

            batch_clusters.append({"label": label,"members": llm_cluster})
            # print(f"Unique entities before: {len(self.unique_entities_name)}\n")
            self.unique_entities_name = [entity for entity in self.unique_entities_name if entity not in llm_cluster]
            # print(f"Unique entities after: {len(self.unique_entities_name)}\n")
            remaining = [entity for entity in remaining if entity not in llm_cluster]
            patience = 0

        return batch_clusters, remaining
    
    def local_clustering(self):
        """
        Batches the entity list and calls cluster_single_batch()
        function for every batch.
        """
        #Creating the batches
        batches = []
        for i in range(0, len(self.unique_entities_name), self.batch_size):
            batch = self.unique_entities_name[i : i + self.batch_size]
            batches.append(batch)

        #Clustering for each batch
        all_leftovers = []
        for idx, batch in enumerate(batches):
            print(f"[Batch {idx+1}/{len(batches)}] size={len(batch)}")
            batch_clusters, batch_left = self.cluster_single_batch(batch)
            self.micro_clusters.extend(batch_clusters)
            all_leftovers.extend(batch_left)

        print(f"local clustering finished: {len(self.micro_clusters)} micro-clusters, "f"{len(all_leftovers)} leftover entities")


        self.clusters = [{"label": cluster["label"], "members": cluster["members"]} for cluster in self.micro_clusters]

    def get_exemplars(self, members):
        if len(members) <= self.exemplar_size:
            return members
        return members[:self.exemplar_size]

    def ask_merge(self, clusterA, clusterB):
        examples_A = "\n".join(clusterA)
        examples_B = "\n".join(clusterB)

        response = self.llm.predict(prompt=self.cluster_merge_prompt, cluster_a=examples_A, cluster_b=examples_B).strip().lower()
        # print(f"llm response: {response}\n")  
        response = self.parse_response_fn(response)
        # print(f"llm response parse: {response}\n")
        if response == "yes":
            return True
        return False

    def global_clustering(self):
        """
        Checks every cluster with every other and tries to 
        concatenate them into a bigger one, if they are 
        semanticly similar.
        """
        clusters = [{"label": cluster["label"], "members": cluster["members"]} for cluster in self.micro_clusters]
        # print(f"clusters: {len(clusters)} start: {clusters}\n")
        passes = 0
        stable = False
        while not stable and passes < self.max_merge_passes:
            passes += 1
            stable = True
            to_remove = set()
            for i in range(len(clusters)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(clusters)):
                    if j in to_remove:
                        continue
                    exA = self.get_exemplars(clusters[i]["members"])
                    exB = self.get_exemplars(clusters[j]["members"])
                    # print(f"asking with exA: {exA} and exB: {exB}\n")
                    if self.ask_merge(exA, exB):
                        clusters[i]["members"] = list(dict.fromkeys(clusters[i]["members"] + clusters[j]["members"]))
                        # print(f"new cluster: {clusters[i]['label']}:{clusters[i]['members']}\n")
                        to_remove.add(j)
                        stable = False

            clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in to_remove]

        # print(f"clusters: {len(clusters)} end: {clusters}\n")
        self.clusters = [{"label": cluster["label"], "members": cluster["members"]} for cluster in clusters]

    def expand_clusters(self):
        """
        Batching the remaining entities, those without a cluster,
        and checking if they can be added to an existing one.
        """
        if not self.unique_entities_name:
            return

        for i in range(0, len(self.unique_entities_name), self.batch_size):
            entities_batch = self.unique_entities_name[i:i+self.batch_size]

            for j in range(0, len(self.clusters), self.batch_size):
                clusters_batch = self.clusters[j:j+self.batch_size]

                cluster_desc_str = ""
                for cluster in clusters_batch:
                    exemplars = ", ".join(self.get_exemplars(cluster["members"]))
                    cluster_desc_str += f"- {cluster['label']}: {exemplars}\n"

                # print(f"entities_batch: {entities_batch}\n")
                response = self.llm.predict(prompt=self.remaining_entities_cluster_prompt,clusters=cluster_desc_str,batch=entities_batch)
                # print(f"response: {response}\n")
                assignments = self.parse_batch_cluster(response)
                # print(f"assignments: {assignments}\n")

                for entity, target in assignments.items():
                    if entity in self.unique_entities_name and target != "none":
                        for cluster in self.clusters:
                            if cluster["label"] == target:
                                # print(f"to append {entity} to {target}\n")
                                cluster["members"].append(entity)
                                self.unique_entities_name.remove(entity)
                                break

    def create_final_entities(self, unique_entities):
        """
        Transforms every entity and batch into EntityNode.
        """
        final_entities = {}
        for cluster in self.clusters:
            desc = ""
            for name, node in unique_entities.items():
                if name in cluster["members"]:
                    if node.properties.get('entity_description') is not None:
                        desc += f" {node.properties.get('entity_description')}" 
                    elif node.properties.get('relationship_description') is not None:
                        desc += f" {node.properties.get('relationship_description')}" 
                    else:
                        desc += f""
            final_entities[cluster["label"]] = EntityNode(name=cluster["label"], properties={"cluster_members": cluster["members"], "entity_description": desc})
        for entity in self.unique_entities_name:
            if entity not in final_entities:
                final_entities[entity] = unique_entities[entity]
        print(f"Total number of entities after clustering: {len(final_entities)}")
        return final_entities

    def cluster(self, unique_entities):
        self.unique_entities_name = list(unique_entities.keys()).copy()
        # print(f"Local Clustering Starting...\n")
        # self.local_clustering()
        # # print(f"Global Clustering Starting...\n")
        # # self.global_clustering()
        # print(f"Cluster Expansion starting...\n")
        # self.expand_clusters()
        print(f"Final Entities Are Being Created...\n")
        return self.create_final_entities(unique_entities)
