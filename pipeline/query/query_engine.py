from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.llms import ChatMessage
from pipeline.graph.graph_store import GraphRAGStore
from typing import List, Dict, Any
import re
import numpy as np

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM
    embedder: Any
    final_context_token_limit: int = 3000

    def __init__(self, graph_store: GraphRAGStore, llm: LLM, embedder: Any, *, final_context_token_limit: int = 3000):
        super().__init__(graph_store=graph_store, llm=llm, embedder=embedder, final_context_token_limit=final_context_token_limit)
        self.embedder = embedder

    def _final_answer_from_selected(self, selected: List[Dict[str, Any]], query: str) -> str:
        """
        Builds a final prompt from the intermediate answers and the query,
        and produces the final answer.
        """
        answers_text = []
        for it_answers in selected:
            cid = it_answers["community_id"]
            ans = it_answers["answer"]
            summary = it_answers["summary"]
            answers_text.append(f"COMMUNITY_ID: {cid}\nANSWER: {ans}\nORIGINAL SUMMARY (verification only):\n{summary}")

        combined = "\n\n".join(answers_text)
        prompt_system = (
        "You are a deterministic answer verifier and selector.\n\n"
        "You receive:\n"
        "- A user query\n"
        "- Extracted answers from community summaries\n"
        "- The original summaries\n\n"

        "### YOUR TASK\n"
        "Select the correct answer AND the exact summary that supports it.\n\n"

        "### STRICT RULES\n"
        "- The final answer must be based on ONE of the extracted answers.\n"
        "- You MUST verify each extracted answer against its summary.\n"
        "- If an extracted answer is slightly imprecise but the correct information\n"
        "  is explicitly written in the same summary, you may refine it using ONLY\n"
        "  that summary.\n"
        "- Do NOT combine multiple summaries.\n"
        "- Do NOT invent information.\n"
        "- If no summary explicitly supports a correct answer → output EXACTLY:\n"
        "Insufficient information to answer based on the available summaries.\n"
        "- Output MUST follow EXACTLY this format:\n\n"

        "FINAL ANSWER:\n"
        "<answer>\n\n"
        "SELECTED SUMMARY:\n"
        "<summary>\n"
        )
        prompt_user = (
            f"USER QUERY:\n{query}\n\n"
            f"INTERMEDIATE ANSWERS (sorted by helpfulness):\n{combined}\n\n"
            "### NOW PRODUCE THE FINAL ANSWER BELOW.\n"
            "REMEMBER: Output ONLY the final factual answer and its corresponding summary."
        )
        messages = [
            ChatMessage(role="system", content=prompt_system),
            ChatMessage(role="user", content=prompt_user)
        ]
        response = str(self.llm.chat(messages)).strip()
        response = re.sub(r"^assistant:\s*", "", response)
        if not response:
            return "Insufficient information to answer based on the available summaries.", ""

        response = re.split(r'("""|\'\'\'|```|#|Explanation:|Reasoning:|Because)',response,flags=re.IGNORECASE)[0].strip()
        if response.startswith("Insufficient information"):
            return "Insufficient information to answer based on the available summaries.", ""
        
        answer_match = re.search(r"FINAL ANSWER:\s*(.*?)\s*SELECTED SUMMARY:",response,re.DOTALL)
        summary_match = re.search(r"SELECTED SUMMARY:\s*(.*)",response,re.DOTALL)
        if not answer_match and not summary_match:
            return "Insufficient information to answer based on the available summaries.", ""
        elif not answer_match:
            selected_summary = summary_match.group(1).strip()    
            return "Insufficient information to answer based on the available summaries.", selected_summary
        elif not summary_match:
            final_answer = answer_match.group(1).strip()
            return final_answer,""

        final_answer = answer_match.group(1).strip()
        selected_summary = summary_match.group(1).strip()
        return final_answer, selected_summary

    def _reduce_to_final_context(self, intermediate_answers: List[Dict[str, Any]], token_limit: int) -> List[Dict[str, Any]]:
        """
        Sorts intermediate answers by score descending, then adds them 
        in the final answer until the token limit is reached.
        """
        #Sort
        ordered = sorted(intermediate_answers, key=lambda x: x["score"], reverse=True)
        selected = []
        used_tokens = 0
        for item in ordered:
            tokens = self._count_tokens(
                f"COMMUNITY_ID:{item['community_id']}\n"
                f"ANSWER:{item['answer']}\n"
                f"SUMMARY:{item['summary']}"
            )
            #Token limit is reached
            if used_tokens + tokens > token_limit:
                #Stops adding further answers
                break
            if item["score"] <= 0:
                #Skips non useful ones
                continue
            selected.append(item)
            used_tokens += tokens

        return selected

    def _map_summary(self, cid: str, summary: str, query: str) -> Dict[str, Any]:
        """
        Processes one community summary and produces an intermediate answer to the query.
        Returns:
        {"community_id": cid, "answer": str, "score": int, "summary": str}
        """

        prompt_system = (
            "You are a deterministic evaluator.\n"
            "You receive ONE community summary and a user query.\n\n"
            "Your task:\n"
            "1) Produce a short factual answer STRICTLY based on the summary.\n"
            "2) Produce a helpfulness score 0–100 reflecting how much the summary helps answer the query.\n\n"
            "RULES:\n"
            "- Use ONLY information explicitly in the summary.\n"
            "- If the answer is not explicitly stated then output EXACTLY: insufficient information\n"
            "- The answer must be ONLY the fact itself.\n"
            "- Score 0 = irrelevant. Score 100 = perfectly relevant.\n"
            "- Follow the exact output format.\n"
        )
        prompt_user = (
            f"USER QUERY:\n{query}\n\n"
            f"COMMUNITY_ID: {cid}\n"
            f"SUMMARY:\n{summary}\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "ANSWER: <short factual answer OR 'insufficient information'>\n"
            "SCORE: <integer 0-100>"
            "### FEW-SHOT EXAMPLES\n"
            "\n"
            "Example 1:\n"
            "Summary: The group studies renewable energy and solar panel efficiency.\n"
            "Query: What does the group study?\n"
            "Expected:\n"
            "COMMUNITY_ID: 12\n"
            "ANSWER: renewable energy and solar panel efficiency\n"
            "SCORE: 90\n\n"

            "Example 2:\n"
            "Summary: This community explores quantum computing and its applications in cryptography.\n"
            "Query: What applications does the community explore?\n"
            "Expected:\n"
            "COMMUNITY_ID: A\n"
            "ANSWER: cryptography\n"
            "SCORE: 100\n\n"

            "Example 3:\n"
            "Summary: The community researches deep-sea ecosystems and marine biodiversity.\n"
            "Query: Who founded the community?\n"
            "Expected:\n"
            "COMMUNITY_ID: 7\n"
            "ANSWER: insufficient information\n"
            "SCORE: 0\n\n"
            "### NOW PRODUCE THE OUTPUT FOR THE COMMUNITY\n"
            "Follow the template EXACTLY."
        )
        messages = [
            ChatMessage(role="system", content=prompt_system),
            ChatMessage(role="user", content=prompt_user)
        ]
        
        response = str(self.llm.chat(messages)).strip()
        ans_match = re.search(r"ANSWER:\s*(.+)", response)
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        answer = ans_match.group(1).strip() if ans_match else "insufficient information"
        score = int(score_match.group(1)) if score_match else 0
        return {"community_id": cid,"answer": answer,"score": score,"summary": summary}

    def _count_tokens(self,text):
        return len(self.llm._model.tokenize(text.encode("utf-8")))
    
    def _filter_summaries_with_embeddings(self, summaries: list, query: str, top_k: int = 10):
        """
        Filter the given summaries using cosine similarity with precomputed 
        embeddings stored in graph_store.summary_vecs[cid].
        Returns the top_k most relevant summaries.
        """
        if not summaries:
            return []

        #Embeds the query
        qenc = self.embedder.embed(query)
        qvec = np.array(qenc, dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0 or np.isnan(qnorm):
            return summaries[:top_k]
        
        #Normalizes the query
        qvec = qvec / qnorm
        scored = []
        #Computes similarity for the given summary cids
        for cid, text in summaries:
            svec = self.graph_store.summary_vecs.get(cid)
            if svec is None:
                continue
            #Cosine similarity
            score = float(np.dot(qvec, svec))
            scored.append((cid, text, score))

        if not scored:
            return []

        #Sorts by similarity
        def score_key(item):
            return item[2]

        scored.sort(key=score_key, reverse=True)
        #Returns top-k
        top_n = min(top_k, len(scored))
        results = []
        for cid, text, score in scored[:top_n]:
            results.append((cid, text))

        return results

    def _get_all_summaries(self):
        """
        Returns all community summaries stored in GraphRAGStore.
        """
        summaries = self.graph_store.get_community_summaries()
        result = []
        for cid, info in summaries.items():
            result.append((cid, info))
        return result

    def custom_query(self, query_str: str) -> str:
        #Gets all summaries
        summaries = self._get_all_summaries()
        if not summaries:
            return "Insufficient information to answer based on the available summaries.", ""

        #Filters summaries with embeddings
        summaries_filtered = self._filter_summaries_with_embeddings(summaries=summaries, query=query_str, top_k=10)
        if not summaries_filtered:
            return "Insufficient information to answer based on the available summaries.", ""

        #Collects the intermediate answers
        intermediate_answers = []
        for cid, summary_text in summaries_filtered:
            result = self._map_summary(cid, summary_text, query_str)
            if result["score"] > 0:
                intermediate_answers.append(result)
        if not intermediate_answers:
            return "Insufficient information to answer based on the available summaries.", ""
        
        #Selects best intermediate answers until token limit
        selected = self._reduce_to_final_context(intermediate_answers, self.final_context_token_limit)
        if not selected:
            return "Insufficient information to answer based on the available summaries.", ""

        #Final answer
        final_answer, final_summary = self._final_answer_from_selected(selected, query_str)
        return final_answer, final_summary