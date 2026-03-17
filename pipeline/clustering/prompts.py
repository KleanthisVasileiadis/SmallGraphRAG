default_entity_clustering_prompt = """
        You are an AI that extracts related entities into a single semantic cluster.

        ## TASK
        Given a **small batch** of entities, find **ONE** cluster of entities that share a clear semantic relationship.  
        They do NOT need to be identical in meaning — they only need to belong to the same semantic category, theme, or concept family.

        You MUST return **one small, tight, meaningful cluster**, not a large or loose grouping.

        ### Rules
        - Entities may be synonyms, variants, or belong to the same conceptual group or category.
        - Valid clusters include entities that share a domain, topic, or semantic theme (e.g., countries, diseases, programming terms, cities, financial metrics).
        - Prefer clusters of 2–7 entities.
        - Do NOT attempt to use the entire list.
        - Exclude entities that are unrelated or only loosely related.
        - If no meaningful cluster can be formed, output nothing (empty).

        This task is part of **local batch clustering**, so select a small and coherent subset of the batch—not a global grouping.

        ---

        ## OUTPUT FORMAT (STRICT)
        Output **only** in the following format:

        entity: <entity_name>
        entity: <entity_name>
        .

        Each entity must be on its own line starting with `entity:`.  
        End your answer with a single period `.` on a new line.  
        Do **not** include explanations, code blocks, or any extra text.

        ---
        ### FEW-SHOT EXAMPLES

        Example 1
        **Input:**
        Entities list:
        dog, dogs, cat, running, Dog

        **Correct Output:**
        entity: dog
        entity: dogs
        entity: Dog
        .

        ---
        Example 2
        **Input:**
        Entities list:
        AI, Artificial Intelligence, machine learning, deep learning, artificial-intelligence

        **Correct Output:**
        entity: AI
        entity: Artificial Intelligence
        entity: artificial-intelligence
        .

        ---
        Example 3
        **Input:**
        Entities list:
        run, running, ran, jog, jogging

        **Correct Output:**
        entity: run
        entity: running
        entity: ran
        .

        ---
        Example 4
        **Input:**
        Entities list:
        COVID-19, coronavirus, SARS-CoV-2, influenza, flu

        **Correct Output:**
        entity: COVID-19
        entity: coronavirus
        entity: SARS-CoV-2
        .

        ---
        Example 5
        **Input:**
        Entities list:
        New York, NYC, New York City, Los Angeles, NY

        **Correct Output:**
        entity: New York
        entity: NYC
        entity: New York City
        entity: NY
        .

        ---
        ## INPUT
        Entities list:
        {entities}
        """

default_entity_cluster_judge_prompt = """
        You are an AI assistant that validates semantic entity clusters.

        ## TASK
        You are given a list of entities that are proposed to belong to the same semantic cluster.

        A valid cluster is one where all entities share a **clear semantic theme**, category, or meaning relationship.  
        They do NOT need to be identical in meaning — they only need to be meaningfully related.

        ### Rules for a valid cluster
        A cluster is valid if:
        - Entities belong to the same semantic category, topic, group, or concept family.
        - Small variations (plural/singular, tense, spelling) are acceptable.
        - Synonyms, related terms, or hierarchical relations (e.g., specific/general forms of the same topic) are acceptable.

        A cluster is NOT valid if:
        - Entities come from different categories or unrelated semantic areas.
        - They only share extremely broad or vague similarity.

        ---

        ## OUTPUT FORMAT (STRICT)
        You must output exactly one line:

        answer : "yes"
        or
        answer : "no"

        No extra text.

        ---

        ### FEW-SHOT EXAMPLES

        Example 1  
        Input: [run, running, ran]  
        Output: answer : "yes"

        Example 2  
        Input: [dog, cat, animal]  
        Output: answer : "no"  
        (Too broad: mixing specific items with a general class.)

        Example 3  
        Input: [France, Germany, Italy]  
        Output: answer : "yes"

        Example 4  
        Input: [AI, Artificial Intelligence, machine learning]  
        Output: answer : "yes"

        Example 5  
        Input: [COVID-19, coronavirus, flu]  
        Output: answer : "no"

        ---

        ## INPUT
        Entities to validate:
        {entities}
        """

default_entity_cluster_label_prompt = """
        You are an AI assistant that assigns concise, meaningful labels to clusters of related entities.

        ## TASK
        You are given a cluster of entities that all represent the same concept.

        Your goal:
        - Provide a **single word or short phrase** that best represents the shared meaning of these entities.
        - The label should summarize the cluster clearly and naturally (e.g., a canonical form or general concept).
        - Do NOT include quotes, punctuation, numbering, or any explanation.
        - Output **exactly one line** in the required format.

        ---

        ## OUTPUT FORMAT (STRICT)
        Output only one line using this format:

        answer : <label>

        Example:
        answer : running

        Do not add code blocks, markdown, or any other text.

        ---

        ### FEW-SHOT EXAMPLES

        Example 1  
        **Input:**  
        Entities in the cluster: [run, running, ran, jog]  
        **Correct Output:**  
        answer : running

        ---

        Example 2  
        **Input:**  
        Entities in the cluster: [dog, dogs, puppy, canine]  
        **Correct Output:**  
        answer : dog

        ---

        Example 3  
        **Input:**  
        Entities in the cluster: [AI, Artificial Intelligence, artificial-intelligence]  
        **Correct Output:**  
        answer : artificial intelligence

        ---

        Example 4  
        **Input:**  
        Entities in the cluster: [COVID-19, coronavirus, SARS-CoV-2]  
        **Correct Output:**  
        answer : COVID-19

        ---

        Example 5  
        **Input:**  
        Entities in the cluster: [New York, NYC, New York City, NY]  
        **Correct Output:**  
        answer : New York City

        ---

        ## INPUT
        Entities in the cluster:
        {entities}
        """

default_entity_cluster_merge_prompt = """
        You are an expert in semantic clustering.

        ## TASK
        Determine whether Cluster A and Cluster B should be merged into a single, larger semantic cluster.

        These clusters come from **local batch clustering**, so each one contains a small set of closely related entities.  
        Your job is to decide if they represent the **same semantic category or concept family**.

        ### Rules
        - Say **"answer: yes"** if both clusters clearly belong to the same semantic category, topic, concept family, or refer to strongly overlapping meanings.
        - The clusters do NOT need to be identical.
        - Accept merges for:
        - synonyms or variants
        - items in the same conceptual group
        - entities describing the same domain or type (e.g., countries, diseases, ML techniques, car brands, programming paradigms)
        - Say **"answer: no"** if the clusters represent different themes, unrelated concepts, or incompatible categories.
        - Focus on **semantic coherence**, not strict equivalence.

        ### OUTPUT FORMAT
        Output only:
        answer: yes
        or
        answer: no

        ## FEW-SHOT EXAMPLES

        ### EXAMPLE 1
        Cluster A examples:
        [france,germany,spain]

        Cluster B examples:
        [italy,portugal,netherlands]

        answer: yes

        ### EXAMPLE 2
        Cluster A examples:
        [python,java,c++]

        Cluster B examples:
        [tiger,lion,cheetah]

        answer: no

        ### EXAMPLE 3
        Cluster A examples:
        [diabetes,hypertension,asthma]

        Cluster B examples:
        [coronary artery disease,stroke,heart failure]

        answer: yes

        ### EXAMPLE 4
        Cluster A examples:
        [tokyo,new_york,london]

        Cluster B examples:
        [docker,kubernetes,terraform]

        answer: no

        ### REAL INPUT
        Cluster A examples:
        {cluster_a}

        Cluster B examples:
        {cluster_b}
        """

default_remaining_entities_cluster_prompt = """
        You are an AI assistant performing semantic clustering.

        ## TASK
        You are given semantic clusters (as exemplars) and a list of entities.
        Assign each entity to the cluster it best fits.
        If it does not clearly fit ANY cluster, assign "none".

        ## IMPORTANT
        You MUST follow the output format EXACTLY.
        Do NOT print code, dictionaries, lists, or explanations.
        Do NOT rewrite the clusters or the entities.
        Do NOT output Python.
        Only output classification lines.
        ---

        ## OUTPUT FORMAT (STRICT)
        For each entity, output exactly:

        ENTITIES:
        <entity_name> : <cluster_name>
        or
        <entity_name> : none

        End the output with a single period "." on a new line.

        No other text is allowed.
        ---

        ## FEW-SHOT EXAMPLES

        ### EXAMPLE 1
        CLUSTERS:
        Fruits: apple, banana, mango
        Animals: dog, cat, lion

        ENTITIES:
        apple, lion, train

        OUTPUT:
        ENTITIES:
        apple : Fruits
        lion : Animals
        train : none
        .

        ### EXAMPLE 2
        CLUSTERS:
        Programming Languages: python, java, c++, javascript
        Databases: mysql, postgres, mongodb

        ENTITIES:
        javascript, tiger, mysql

        OUTPUT:
        ENTITIES:
        javascript : Programming Languages
        tiger : none
        mysql : Databases
        .

        ### EXAMPLE 3
        CLUSTERS:
        Planets: mercury, venus, earth, mars
        Stars: sun, sirius, vega

        ENTITIES:
        earth, milky_way, sirius

        OUTPUT:
        ENTITIES:
        earth : Planets
        milky_way : none
        sirius : Stars
        .

        ---
        ## CLUSTERS
        {clusters}

        ## ENTITIES
        ENTITIES:
        {batch}
        """

default_edge_clustering_prompt = """
        You are an AI that extracts related relations into a single semantic cluster.

        ## TASK
        Given a **small batch** of relations, find **ONE** cluster of relations that share a clear semantic relationship.  
        They do NOT need to be identical in meaning — they only need to belong to the same semantic category, theme, or concept family.

        You MUST return **one small, tight, meaningful cluster**, not a large or loose grouping.

        ### Rules
        - Relationships may be synonyms, variants, or belong to the same conceptual group or category.
        - Valid clusters include relations that share a domain, topic, or semantic theme.
        - Prefer clusters of 2–7 relations.
        - Do NOT attempt to use the relation list.
        - Exclude relations that are unrelated or only loosely related.
        - If no meaningful cluster can be formed, output nothing (empty).

        This task is part of **local batch clustering**, so select a small and coherent subset of the batch—not a global grouping.

        ---
        ## OUTPUT FORMAT (STRICT)
        Output **only** in the following format:

        relation: <relation_name>
        relation: <relation_name>
        .

        Each relation must be on its own line starting with `relation:`  
        End your answer with a single period (`.`) on a new line.  
        Do **not** include explanations, code blocks, or any extra text.

        ---
        ### FEW-SHOT EXAMPLES
        Example 1  
        **Input:**  
        Relations list:  
        works_at, employed_by, founded, manages

        **Correct Output:**  
        relation: works_at  
        relation: employed_by  
        .

        ---
        Example 2  
        **Input:**  
        Relations list:  
        founded, established, created, owns

        **Correct Output:**  
        relation: founded  
        relation: established  
        relation: created  
        .

        ---
        Example 3  
        **Input:**  
        Relations list:  
        located_in, situated_in, located_at, borders

        **Correct Output:**  
        relation: located_in  
        relation: situated_in  
        relation: located_at  
        .

        ---
        Example 4  
        **Input:**  
        Relations list:  
        released, launched, developed, built

        **Correct Output:**  
        relation: released  
        relation: launched  
        .

        ---
        Example 5  
        **Input:**  
        Relations list:  
        married_to, wife_of, mother_of, husband_of

        **Correct Output:**  
        relation: married_to  
        relation: wife_of  
        relation: husband_of  
        .

        ---
        ## INPUT
        Relations list:
        {relations}
        """

default_edge_clustering_judge_prompt = """
        You are an AI assistant that validates semantic relations clusters.

        ## TASK
        You are given a list of relations that are proposed to belong to the same semantic cluster.

        A valid cluster is one where all relations share a **clear semantic theme**, category, or meaning relationship.  
        They do NOT need to be identical in meaning — they only need to be meaningfully related.

        ### Rules for a valid cluster
        A cluster is valid if:
        - relations belong to the same semantic category, topic, group, or concept family.
        - Small variations (plural/singular, tense, spelling) are acceptable.
        - Synonyms, related terms, or hierarchical relations (e.g., specific/general forms of the same topic) are acceptable.

        A cluster is NOT valid if:
        - relations come from different categories or unrelated semantic areas.
        - They only share extremely broad or vague similarity.
        
        ---
        ## OUTPUT FORMAT (STRICT)
        You must output exactly one line:

        answer : "yes"
        or
        answer : "no"

        No extra text.

        ---
        ### FEW-SHOT EXAMPLES

        Example 1  
        **Input:**  
        Edges to validate:  
        [works_at, employed_by]  
        **Correct Output:**  
        answer : "yes"

        ---
        Example 2  
        **Input:**  
        Edges to validate:  
        [founded, established, created]  
        **Correct Output:**  
        answer : "yes"

        ---
        Example 3  
        **Input:**  
        Edges to validate:  
        [located_in, situated_in, located_at]  
        **Correct Output:**  
        answer : "yes"

        ---
        Example 4  
        **Input:**  
        Edges to validate:  
        [released, launched]  
        **Correct Output:**  
        answer : "yes"

        ---
        Example 5  
        **Input:**  
        Edges to validate:  
        [developed, acquired, built]  
        **Correct Output:**  
        answer : "no"

        ---
        ## INPUT
        Edges to validate:
        {edges}
        """

default_edge_cluster_label_prompt = """
        You are an AI assistant that assigns concise, meaningful labels to clusters of related relations.

        ## TASK
        You are given a cluster of relations that all represent the same concept.

        Your goal:
        - Provide a **single word or short phrase** that best represents the shared meaning of these relations.
        - The label should summarize the cluster clearly and naturally (e.g., a canonical form or general concept).
        - Do NOT include quotes, punctuation, numbering, or any explanation.
        - Output **exactly one line** in the required format.

        ---
        ## OUTPUT FORMAT (STRICT)
        Output only one line using this format:

        answer : <label>

        Example:
        answer : running

        Do not add code blocks, markdown, or any other text.

        ---
        ### FEW-SHOT EXAMPLES

        Example 1  
        **Input:**  
        Edges in the cluster:  
        [works_at, employed_by]  
        **Correct Output:**  
        answer : employment

        ---
        Example 2  
        **Input:**  
        Edges in the cluster:  
        [located_in, in, situated_in]  
        **Correct Output:**  
        answer : location

        ---
        Example 3  
        **Input:**  
        Edges in the cluster:  
        [founded, established, created]  
        **Correct Output:**  
        answer : creation

        ---
        Example 4  
        **Input:**  
        Edges in the cluster:  
        [married_to, husband_of, wife_of]  
        **Correct Output:**  
        answer : marriage

        ---
        Example 5  
        **Input:**  
        Edges in the cluster:  
        [released, launched]  
        **Correct Output:**  
        answer : product release

        ---
        ## INPUT
        Edges in the cluster:
        {edges}
        """

default_edge_cluster_merge_prompt = """
        You are an expert in semantic clustering.

        ## TASK
        Determine whether Cluster A and Cluster B should be merged into a single, larger semantic cluster.

        These clusters come from **local batch clustering**, so each one contains a small set of closely related relations.  
        Your job is to decide if they represent the **same semantic category or concept family**.

        ### Rules
        - Say **"answer: yes"** if both clusters clearly belong to the same semantic category, topic, concept family, or refer to strongly overlapping meanings.
        - The clusters do NOT need to be identical.
        - Accept merges for:
        - synonyms or variants
        - items in the same conceptual group
        - relations describing the same domain or type
        - Say **"answer: no"** if the clusters represent different themes, unrelated concepts, or incompatible categories.
        - Focus on **semantic coherence**, not strict equivalence.

        ### OUTPUT FORMAT
        Output only:
        answer: yes
        or
        answer: no

        ## FEW-SHOT EXAMPLES

        ### EXAMPLE 1
        Cluster A examples:
        [located_in,based_in,situated_in]

        Cluster B examples:
        [is_in,resides_in,found_in]

        answer: yes

        ### EXAMPLE 2
        Cluster A examples:
        [part_of,component_of,subset_of]

        Cluster B examples:
        [authored_by,written_by,created_by]

        answer: no

        ### EXAMPLE 3
        Cluster A examples:
        [works_at,employed_by,hired_by]

        Cluster B examples:
        [job_at,staff_of,member_of]

        answer: yes

        ### EXAMPLE 4
        Cluster A examples:
        [married_to,related_to,sibling_of]

        Cluster B examples:
        [invented_by,designed_by,developed_by]

        answer: no

        ---
        Cluster A examples:
        {cluster_a}

        Cluster B examples:
        {cluster_b}
        """

default_remaining_edges_cluster_prompt = """
        You are an AI assistant performing semantic clustering.

        ## TASK
        You are given semantic clusters (as exemplars) and a list of relations.
        Assign each relation to the cluster it best fits.
        If it does not clearly fit ANY cluster, assign "none".

        ## IMPORTANT
        You MUST follow the output format EXACTLY.
        Do NOT print code, dictionaries, lists, or explanations.
        Do NOT rewrite the clusters or the entities.
        Do NOT output Python.
        Only output classification lines.
        ---

        ## OUTPUT FORMAT (STRICT)
        For each relation, output exactly:

        RELATIONS:
        <relation_name> : <cluster_name>
        or
        <relation_name> : none

        End the output with a single period "." on a new line.

        No other text is allowed.
        ---

        ## EXAMPLES (STRICT)

        ### Example 1
        **INPUT CLUSTERS:**
        Astronomical_Mechanics: orbits, rotates_around
        Spatial_Properties: located_in, position_of

        **INPUT BATCH:**
        revolves_around, situated_in, emits_light

        **OUTPUT:**
        RELATIONS:
        revolves_around : Astronomical_Mechanics
        situated_in : Spatial_Properties
        emits_light : none
        .

        ### Example 2
        **INPUT CLUSTERS:**
        Historical_Discovery: discovered_by, named_after
        Physical_Characteristics: has_temperature, has_mass

        **INPUT BATCH:**
        uncovered_by, possesses_mass, correlates_with

        **OUTPUT:**
        RELATIONS:
        uncovered_by : Historical_Discovery
        possesses_mass : Physical_Characteristics
        correlates_with : none
        .

        ### Example 3
        **INPUT CLUSTERS:**
        Astronomical_Mechanics: rotates_around, orbits
        Structural_Relations: part_of, component_of

        **INPUT BATCH:**
        circles, belongs_to, affects

        **OUTPUT:**
        RELATIONS:
        circles : Astronomical_Mechanics
        belongs_to : Structural_Relations
        affects : none
        .
        ---

        ## CLUSTERS
        {clusters}

        ## ENTITIES
        ENTITIES:
        {batch}
        """
