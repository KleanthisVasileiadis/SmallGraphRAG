KG_ENTITY_EXTRACT_TMPL = """
        You are an AI specialized in **Knowledge Graph entity extraction**.

        ## TASK
        Given a piece of natural language text, extract **ALL entities that appear EXPLICITLY in the text**.

        This is a **pure surface-form extraction task**:
        - Do NOT decide importance
        - Do NOT filter entities
        - Do NOT normalize names
        - Do NOT infer missing information

        If an entity appears in the text, you MUST extract it.

        ## WHAT COUNTS AS AN ENTITY
        Extract entities such as (but not limited to):
        - People
        - Organizations
        - Locations (countries, cities, regions)
        - Products
        - Works (books, movies, games, albums)
        - Events
        - Materials
        - Scientific or technical terms
        - Named concepts or titles
        - Dates and numbers IF explicitly mentioned

        When in doubt, **EXTRACT the entity**.

        ## CRITICAL RULES
        - The input text is **natural language**, not code.
        - Copy entity names **exactly as they appear** in the text.
        - Do NOT merge entities.
        - Do NOT explain or justify.
        - Do NOT extract relations.
        - Do NOT invent entities.
        - You MUST output **ONLY** entity blocks in the strict format below.
        - Descriptions must be **short** and derived ONLY from the text wording.
        - If the text gives very little information, write a minimal description reflecting that.

        ---

        ## OUTPUT FORMAT (STRICT)
        Output **ONLY** in the following format:

        ENTITIES:
        entity_name: <string>
        entity_type: <string>
        entity_description: <string>

        (repeat the above block for every entity)

        End your entire output with a single period `.` on its own line.
        Do **not** include explanations, code blocks, or any extra text.
        ---

        ## FEW-SHOT EXAMPLES

        Example 1  
        **Input:**  
        "Elon Musk founded SpaceX in 2002."

        **Correct Output:**  
        ENTITIES:
        entity_name: Elon Musk
        entity_type: Person
        entity_description: Founder of SpaceX.

        entity_name: SpaceX
        entity_type: Organization
        entity_description: Company founded by Elon Musk.

        entity_name: 2002
        entity_type: Date
        entity_description: Year when SpaceX was founded.
        .
        ---

        Example 2  
        **Input:**  
        "Barack Obama served as the 44th President of the United States."

        **Correct Output:**  
        ENTITIES:
        entity_name: Barack Obama
        entity_type: Person
        entity_description: Served as the 44th President of the United States.

        entity_name: 44th President
        entity_type: Title
        entity_description: Presidential role held by Barack Obama.

        entity_name: United States
        entity_type: Country
        entity_description: Country where Barack Obama served as President.
        .
        ---

        Example 3  
        **Input:**  
        "Apple released the iPhone 15 with a new titanium frame."

        **Correct Output:**  
        ENTITIES:
        entity_name: Apple
        entity_type: Organization
        entity_description: Company that released the iPhone 15.

        entity_name: iPhone 15
        entity_type: Product
        entity_description: Device released by Apple.

        entity_name: titanium
        entity_type: Material
        entity_description: Material used for the phone frame.
        .
        ---

        Example 4  
        **Input:**  
        "The Eiffel Tower is located in Paris."

        **Correct Output:**  
        ENTITIES:
        entity_name: Eiffel Tower
        entity_type: Landmark
        entity_description: Structure located in Paris.

        entity_name: Paris
        entity_type: City
        entity_description: City where the Eiffel Tower is located.
        .
        ---

        Example 5  
        **Input:**  
        "OpenAI developed ChatGPT based on GPT architecture."

        **Correct Output:**  
        ENTITIES:
        entity_name: OpenAI
        entity_type: Organization
        entity_description: Organization that developed ChatGPT.

        entity_name: ChatGPT
        entity_type: Model
        entity_description: AI model developed by OpenAI.

        entity_name: GPT
        entity_type: Architecture
        entity_description: Architecture used as the basis for ChatGPT.
        .
        ---

        ## INPUT
        {text}
        """

KG_RELATION_EXTRACT_TMPL = """
        You are an AI specialized in **local Knowledge Graph relation extraction**.

        ## TASK
        Given:
        1. A short natural-language text passage (1–2 consecutive sentences), and
        2. A list of extracted entities that appear in this passage,

        extract **ALL explicit relationships stated directly in the text**
        between the provided entities.

        You must:
        - Consider ONLY the entities provided
        - Extract relations ONLY if both entities explicitly appear in the text
        - Ignore any entities not listed

        This is a **pure surface-level extraction task**:
        - Do NOT infer
        - Do NOT reason
        - Do NOT generalize
        - Do NOT use world knowledge
        - Do NOT add missing information

        If no explicit relation exists between any of the listed entities,
        output **NONE**.

        ---

        ## WHAT COUNTS AS A RELATION

        A relation exists ONLY if the text explicitly:
        - Describes an action, role, attribute, location, ownership, composition, origin, or association
        - Directly connects two listed entities using a verb, preposition, or fixed phrase

        Do NOT extract implicit or background relations.

        ---

        ## RELATION CONSTRAINTS

        - Use ONLY entities from the provided entity list
        - Predicate must be **1–3 words**
        - Predicate must closely match wording in the text
        - Do NOT normalize predicates
        - Do NOT invent semantic meaning
        - relationship_description must restate the fact exactly as written

        ---

        ### OUTPUT FORMAT (STRICT)

        If at least one relation exists, output:

        RELATIONSHIPS:
        source_entity: <string>
        target_entity: <string>
        relation: <string>
        relationship_description: <string>

        (repeat for each relation)

        End the output with a single period `.` on a new line.

        If NO relation exists, output exactly:

        NONE
        .

        ---

        ## FEW-SHOT EXAMPLES

        Example 1  
        Text:
        "Barack Obama studied at Harvard University."

        Entities:
        entity_name: Barack Obama
        entity_type: Person
        entity_description: Studied at Harvard University.

        entity_name: Harvard University
        entity_type: Organization
        entity_description: University where Barack Obama studied.

        Correct Output:
        RELATIONSHIPS:
        source_entity: Barack Obama
        target_entity: Harvard University
        relation: studied at
        relationship_description: Barack Obama studied at Harvard University.
        .

        ---

        Example 2  
        Text:
        "Barack Obama studied at Harvard University. He later became president."

        Entities:
        entity_name: Barack Obama
        entity_type: Person
        entity_description: Studied at Harvard University and later became president.

        entity_name: Harvard University
        entity_type: Organization
        entity_description: University where Barack Obama studied.

        entity_name: president
        entity_type: Title
        entity_description: Role that Barack Obama later became.

        Correct Output:
        RELATIONSHIPS:
        source_entity: Barack Obama
        target_entity: Harvard University
        relation: studied at
        relationship_description: Barack Obama studied at Harvard University.
        .

        ---

        Example 3  
        Text:
        "Apple released the iPhone 15 in 2023."

        Entities:
        entity_name: Apple
        entity_type: Organization
        entity_description: Company that released the iPhone 15.

        entity_name: iPhone 15
        entity_type: Product
        entity_description: Product released by Apple.

        entity_name: 2023
        entity_type: Date
        entity_description: Year when the iPhone 15 was released.

        Correct Output:
        RELATIONSHIPS:
        source_entity: Apple
        target_entity: iPhone 15
        relation: released
        relationship_description: Apple released the iPhone 15 in 2023.
        .
        ---

        ## INPUT
        Text:
        {text}

        Entities:
        {entities}
"""
