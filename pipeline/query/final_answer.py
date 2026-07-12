import re
from llama_index.core.llms import ChatMessage

def final_synthesis(llm, query, community_answer, community_summary, graph_answer, graph_summary):
    prompt_system = (
        "You are a deterministic answer verifier and synthesizer.\n\n"
        "You receive:\n"
        "- A user query\n"
        "- An answer from community-based reasoning\n"
        "- An answer from graph-based reasoning\n"
        "- Their corresponding supporting summaries\n\n"

        "### YOUR TASK\n"
        "Produce ONE final factual answer.\n\n"

        "### STRICT RULES\n"
        "- Each answer must be verified against its corresponding summary.\n"
        "- Summaries are authoritative evidence.\n"

        "- If an answer is unsupported by its summary, discard it.\n"

        "- If an answer is missing or incomplete BUT the correct information\n"
        "  is explicitly present inside its summary, you MAY extract it.\n"

        "### MULTI-SUMMARY REASONING (IMPORTANT)\n"
        "- You MAY combine information from BOTH summaries ONLY IF:\n"
        "  -They refer to the SAME entities\n"
        "  -OR they describe CONNECTED facts needed to answer the query\n"
        "- This is required for multi-hop reasoning.\n"
        "- Do NOT combine unrelated summaries.\n"

        "- Do NOT invent information beyond what is written.\n"

        "- If both answers are valid and consistent → choose the more precise one.\n"

        "- If both answers are partially correct → you MAY synthesize them\n"
        "  ONLY if both summaries explicitly support the synthesis.\n"

        "- If neither summary contains enough explicit information → output EXACTLY:\n"
        "Insufficient information to answer based on the available summaries.\n"

        "- Output ONLY the final factual answer.\n"
        "- No explanation. No reasoning.\n\n"

        "### EXAMPLE (Multi-hop reasoning)\n\n"

        "User Query:\n"
        "Where was the director of Inception born?\n\n"

        "Community Summary:\n"
        "Inception is a film directed by Christopher Nolan.\n\n"

        "Graph Summary:\n"
        "Christopher Nolan was born in London.\n\n"

        "Final Answer:\n"
        "London\n\n"
    )
    prompt_user = (
        f"USER QUERY:\n{query}\n\n"

        f"GRAPH ANSWER:\n{graph_answer}\n\n"
        f"GRAPH SUMMARY:\n{graph_summary}\n\n"

        f"COMMUNITY ANSWER:\n{community_answer}\n\n"
        f"COMMUNITY SUMMARY:\n{community_summary}\n\n"

        "### FINAL ANSWER:"
    )
    messages = [
        ChatMessage(role="system", content=prompt_system),
        ChatMessage(role="user", content=prompt_user)
    ]

    response = str(llm.chat(messages)).strip()
    response = re.sub(r"^assistant:\s*", "", response)
    response = re.split(r'("""|\'\'\'|```|#|Explanation:|Reasoning:|Because)',response,flags=re.IGNORECASE)[0].strip()
    if not response:
        return "Insufficient information to answer based on the available summaries."

    return response