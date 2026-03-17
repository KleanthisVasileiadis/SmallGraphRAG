from pipeline.clustering.prompts import(
    default_edge_clustering_prompt,
    default_edge_clustering_judge_prompt,
    default_edge_cluster_label_prompt,
    default_edge_cluster_merge_prompt,
    default_remaining_edges_cluster_prompt)

from pipeline.clustering.parse_functions import(
    parse_edge_cluster_fn as default_parse_edge_cluster_fn,
    parse_response_fn as default_parse_response_fn,
    parse_label_fn as default_parse_label_fn,
    parse_relation_batch_cluster as default_parse_batch_cluster
)
from llama_index.core.prompts import PromptTemplate

class RelationClusterer:
    def __init__(self, unique_edges, llm, edge_clustering_prompt=default_edge_clustering_prompt, edge_clustering_judge_prompt=default_edge_clustering_judge_prompt,
                 edge_cluster_label_prompt=default_edge_cluster_label_prompt, remaining_edges_cluster_prompt=default_remaining_edges_cluster_prompt, edge_cluster_merge_prompt=default_edge_cluster_merge_prompt,
                 parse_edge_cluster_fn=default_parse_edge_cluster_fn, parse_response_fn=default_parse_response_fn, parse_label_fn=default_parse_label_fn, 
                 parse_batch_cluster=default_parse_batch_cluster, max_iter=2, batch_size=25, exemplar_size=10, max_merge_passes=4):

        self.edge_clustering_prompt = PromptTemplate(edge_clustering_prompt)
        self.edge_clustering_judge_prompt = PromptTemplate(edge_clustering_judge_prompt)
        self.edge_cluster_label_prompt = PromptTemplate(edge_cluster_label_prompt)
        self.edge_cluster_merge_prompt = PromptTemplate(edge_cluster_merge_prompt)
        self.remaining_edges_cluster_prompt = PromptTemplate(remaining_edges_cluster_prompt)

        self.parse_edge_cluster_fn = parse_edge_cluster_fn
        self.parse_response_fn = parse_response_fn
        self.parse_label_fn = parse_label_fn
        self.parse_batch_cluster = parse_batch_cluster

        self.unique_edges = unique_edges.copy()
        self.unique_predicates = [edge[1] for edge in unique_edges]
        self.llm = llm
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.exemplar_size = exemplar_size
        self.max_merge_passes = max_merge_passes

        self.edge_clusters = []
        self.leftovers = []

    def cluster_single_batch(self, batch_predicates):
        """
        Performs micro-clustering on a batch of predicates using LLM
        proposals and validation steps. 
        """
        remaining = batch_predicates.copy()
        batch_clusters = []
        patience = 0
        while remaining and patience < self.max_iter:
                
            llm_cluster = self.llm.predict(prompt=self.edge_clustering_prompt, relations=remaining)
            # print(f"llm_cluster: {llm_cluster}\n")
            llm_cluster = self.parse_edge_cluster_fn(llm_cluster)
            # print(f"llm_cluster parse: {llm_cluster}\n")
            if llm_cluster:
                llm_cluster = [edge.strip() for edge in llm_cluster if edge.strip()]
                llm_cluster = list(dict.fromkeys(llm_cluster))
                llm_cluster = [edge for edge in llm_cluster if edge in remaining]
            if not llm_cluster or len(llm_cluster) <= 0:
                patience += 1
                continue

            judge = self.llm.predict(prompt=self.edge_clustering_judge_prompt, edges=llm_cluster)
            # print(f"llm judge: {judge}\n")
            judge = self.parse_response_fn(judge)
            # print(f"llm judge parse: {judge}\n")
            if judge != "yes":
                patience += 1
                continue

            label = self.llm.predict(prompt=self.edge_cluster_label_prompt, edges=llm_cluster)
            # print(f"llm label: {label}\n")
            label = self.parse_label_fn(label)
            # print(f"llm label parse: {label}\n")
            if label is None:
                patience += 1
                continue

            members = [edge for edge in self.unique_edges if edge[1] in llm_cluster]
            if len(members) == 0:
                patience += 1
                continue

            batch_clusters.append({"label": label, "members": members})
            # print(f"len members: {len(members)}\n")
            # print(f"unique_edges before: {len(self.unique_edges)}\n")
            #Edw nomizw oti prepei kai apta batches na afairw ta edges giati blepw polles fores ta idia
            self.unique_edges = [edge for edge in self.unique_edges if edge not in members]
            self.unique_predicates = [edge[1] for edge in self.unique_edges]
            remaining = [edge for edge in remaining if edge not in llm_cluster]
            # print(f"unique_edges after: {len(self.unique_edges)}\n")
            patience = 0

        return batch_clusters, remaining

    def local_clustering(self):
        """
        Batches the predicates list and calls cluster_single_batch()
        function for every batch.
        """
        #create batches of predicates
        self.unique_predicates = list(set(self.unique_predicates))
        batches = [self.unique_predicates[i:i + self.batch_size]
                for i in range(0, len(self.unique_predicates), self.batch_size)]

        all_leftover_preds = []
        for idx, batch in enumerate(batches):
            print(f"[Predicate Batch {idx+1}/{len(batches)}] size={len(batch)}")
            # print(f"batch: {batch}\n")
            batch_clusters, batch_left = self.cluster_single_batch(batch)

            self.edge_clusters.extend(batch_clusters)
            all_leftover_preds.extend(batch_left)

        self.leftovers = all_leftover_preds
        print(f"local clustering finished: {len(self.edge_clusters)} micro-clusters, {len(self.leftovers)} leftover predicates")

    def get_exemplars(self, members):
        if len(members) <= self.exemplar_size:
            return members
        return members[:self.exemplar_size]

    def ask_merge(self, clusterA, clusterB):
        examples_A = "\n".join(clusterA)
        examples_B = "\n".join(clusterB)

        response = self.llm.predict(prompt=self.edge_cluster_merge_prompt, cluster_a=examples_A, cluster_b=examples_B)
        response = self.parse_response_fn(response)
        # print(f"llm merge judge parse: {response}\n")
        if response == "yes":
            return True
        return False

    def global_clustering(self):
        """
        Checks every cluster with every other and tries to 
        concatenate them into a bigger one, if they are 
        semanticly similar.
        """
        clusters = [{"label": cluster["label"], "members": cluster["members"]} for cluster in self.edge_clusters]
        print(f"clusters: {len(clusters)} start: {clusters}\n")
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

                    exA = self.get_exemplars([edge[1] for edge in clusters[i]["members"]])
                    exB = self.get_exemplars([edge[1] for edge in clusters[j]["members"]])
                    # print(f"asking merge with exA: {exA} and exB: {exB}\n")
                    if self.ask_merge(exA, exB):
                        # merge j into i
                        clusters[i]["members"] = list(dict.fromkeys(clusters[i]["members"] + clusters[j]["members"]))
                        print(f"new cluster: {clusters[i]['label']}:{clusters[i]['members']}\n")
                        to_remove.add(j)
                        stable = False

            clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in to_remove]

        self.edge_clusters = [{"label": cluster["label"], "members": cluster["members"]} for cluster in clusters]
        print(f"clusters: {len(self.edge_clusters)} end: {self.edge_clusters}\n")

    def expand_clusters(self):
        """
        Batching the remaining predicates, those without a cluster,
        and checking if they can be added to an existing one.
        """
        if not self.unique_edges:
            return

        for i in range(0, len(self.unique_edges), self.batch_size):
            batch_triples = self.unique_edges[i:i + self.batch_size]
            batch_preds = [edge[1] for edge in batch_triples]

            for j in range(0, len(self.edge_clusters), self.batch_size):
                clusters_batch = self.edge_clusters[j:j + self.batch_size]
                
                #Cluster description as string <label>: <members>
                cluster_desc_str = ""
                for cluster in clusters_batch:
                    exemplars = ", ".join(self.get_exemplars([edge[1] for edge in cluster["members"]]))
                    cluster_desc_str += f"- {cluster['label']}: {exemplars}\n"

                # print(f"relations_batch: {batch_preds}\n")
                response = self.llm.predict(prompt=self.remaining_edges_cluster_prompt, clusters=cluster_desc_str, batch=batch_preds)
                # print(f"response: {response}\n")
                assignments = self.parse_batch_cluster(response)
                # print(f"assignments: {assignments}\n")

                #Mapping each edge to its corresponding batch based on the LLM
                for triple in batch_triples:
                    pred = triple[1]
                    assigned = assignments.get(pred, None)
                    if assigned and assigned != "none":
                        #finds the global cluster by label and appends the edge
                        for cluster in self.edge_clusters:
                            if cluster["label"] == assigned:
                                if triple not in cluster["members"]:
                                    # print(f"to append {triple} to {cluster}\n")
                                    cluster["members"].append(triple)
                                break
                        #removes triples from unique_edges
                        if triple in self.unique_edges:
                            self.unique_edges.remove(triple)

        #Update predicates list
        self.unique_predicates = [edge[1] for edge in self.unique_edges]

    def create_final_edges(self):
        """
        Adds every relation and batch into final_edges dictionary.
        """
        final_edges = {}
        for cluster in self.edge_clusters:
            label = cluster["label"]
            members = cluster["members"]
            final_edges[label] = {"cluster_label": label, "cluster_members": members}

        for edge in self.unique_edges:
            edge_label = edge[1]
            if edge_label not in final_edges:
                final_edges[edge_label] = {"cluster_label": edge_label, "cluster_members": [edge]}
            else:
                members = final_edges[edge_label]["cluster_members"]
                if edge not in members:
                    members.append(edge)

        print(f"Total number of edges after clustering: {len(final_edges)}")
        return final_edges

    def cluster(self):
        # print("Initial predicate micro-clustering starting...\n")
        # self.local_clustering()
        # # print("Global predicate clustering starting...\n")
        # # self.global_clustering()
        # print("Edge cluster expansion starting...\n")
        # self.expand_clusters()
        print("Creating final edges...\n")
        return self.create_final_edges()
