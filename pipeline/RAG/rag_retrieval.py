import numpy as np
import re
from llama_index.core.prompts import PromptTemplate

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return None
    return vec / norm

class TextChunk:
    def __init__(self, chunk_id, text, doc_id, metadata=None):
        self.chunk_id = chunk_id
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.embedding = None

def embed_chunks(chunks, embedder):
    """
    Computes and stores normalized embeddings for each text chunk.
    """
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    for chunk, emb in zip(chunks, embeddings):
        emb = np.array(emb, dtype=np.float32)
        emb = normalize(emb)
        if emb is None:
            continue
        chunk.embedding = emb

class RAGRetriever:
    def __init__(self, chunks):
        self.chunks = [chunk for chunk in chunks if chunk.embedding is not None]
        self.embeddings = np.stack([chunk.embedding for chunk in self.chunks])

    def retrieve(self, query: str, embedder, top_k: int = 5):
        #Embeds and normalizes the query
        qenc = embedder.embed(query)
        qvec = np.array(qenc, dtype=np.float32)
        qvec = normalize(qvec)
        if qvec is None:
            return []

        #Cosine similarity via dot product
        scores = np.dot(self.embeddings, qvec)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

def rag_retrieve(query, retriever: RAGRetriever, embedder, top_k=5):
    return retriever.retrieve(query, embedder, top_k)

def build_rag_context(rag_results, max_chars=20000):
    """
    Returns a single context string for the LLM.
    """
    context_parts = []
    total_chars = 0
    for chunk, _ in rag_results:
        text = chunk.text.strip()
        if not text:
            continue
        if total_chars + len(text) > max_chars:
            break
        context_parts.append(text)
        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts)

def rag_answer_query(query, rag_results, llm):
    """
    Uses retrieved chunks to answer the query.
    """
    context = build_rag_context(rag_results)

    prompt = f"""
    You are a deterministic answer extraction system.

    ## ROLE
    Your role is NOT to explain, summarize, or reason.
    Your role is to EXTRACT the shortest possible correct answer from the provided context.

    ## INPUT
    - A question
    - A set of short factual text snippets (context)

    ## TASK
    Return ONLY the minimal answer string that directly answers the question.

    ## HARD OUTPUT CONSTRAINTS (STRICT)
    - Output MUST be as short as possible.
    - Prefer:
    • a single word
    • or a short phrase
    • or a list of names separated by commas
    - DO NOT:
    • restate or paraphrase the question
    • include explanations or context
    • use full sentences unless absolutely unavoidable
    • include punctuation other than commas
    - If the answer is a number, date, or name, output ONLY that value.
    - If the answer cannot be found explicitly in the context, output EXACTLY:
    Insufficient information to answer based on the available summaries.

    ## NORMALIZATION RULES
    - Remove determiners (e.g., "the", "a", "an").
    - Prefer canonical names over descriptions.
    - If multiple forms exist, choose the shortest correct one.

    ## EXAMPLES

    Example 1:
    Context:
    ---
    Microsoft was founded by Bill Gates and Paul Allen.
    ---
    Question:
    Who founded Microsoft?
    Answer:
    Bill Gates, Paul Allen

    Example 2:
    Context:
    ---
    This document discusses methods for document clustering.
    ---
    Question:
    When was GraphRAG introduced?
    Answer:
    Insufficient information to answer based on the available summaries.

    ## CONTEXT
    ---
    {context}
    ---

    ## QUESTION
    {query}

    ## FINAL ANSWER
    """.strip()

    prompt = PromptTemplate(prompt)
    response = llm.predict(prompt, context=context, query=query)
    
    #Removes label if exists
    response = re.sub(r"^\s*RAG answer\s*:\s*", "", response, flags=re.IGNORECASE)
    #Splits into lines
    lines = response.splitlines()
    for line in lines:
        line = line.strip()
        #Skips empty lines
        if not line:
            continue
        #Stops if markdown or garbage starts
        if re.match(r"(#+|---+)", line):
            break
        #First valid line is the answer
        return line

    return response