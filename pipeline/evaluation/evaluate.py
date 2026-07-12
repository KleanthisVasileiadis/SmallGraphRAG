import re

from llama_index.core.prompts import PromptTemplate


class Evaluator:
    def __init__(self, llm=None):
        self.llm = llm
        self.llm_prompt = PromptTemplate(
            """You are an expert evaluator. Your role is to determine whether the model's answer
            is semantically equivalent to the gold answer AND correctly answers the query.

            You must perform a BINARY judgment.

            Rules:
            - Output **1** if:
                • The model answer has the same meaning as the gold answer, AND
                • It correctly answers the query.
            - Output **0** if:
                • The meaning differs, OR
                • The answer does not address the query, OR
                • It contains incorrect or unrelated information.

            Notes:
            - Wording does NOT matter, meaning does.
            - Paraphrases → 1
            - Same entity/person/concept but different phrasing → 1
            - Verbose but equivalent → 1
            - Factual deviation → 0
            - Wrong entity → 0
            - Not answering the query → 0

            Your output MUST follow the format:

            answer: <0 or 1>

            Few-shot examples:

            Example 1:
            Query: "Who developed the theory of relativity?"
            Gold: "Albert Einstein"
            LLM: "The scientist Albert Einstein"
            answer: 1

            Example 2:
            Query: "What is the capital of France?"
            Gold: "Paris"
            LLM: "Paris, the capital of France"
            answer: 1

            Example 3:
            Query: "Which mountain is the tallest in the world?"
            Gold: "Mount Everest"
            LLM: "K2"
            answer: 0

            Example 4:
            Query: "When was Google founded?"
            Gold: "1998"
            LLM: "In the late 1990s"
            answer: 1

            Example 5:
            Query: "What is the closest planet to the Sun?"
            Gold: "Mercury"
            LLM: "Venus"
            answer: 0

            Query:
            {query}

            Gold answer:
            {gold_ans}

            Model answer:
            {llm_ans}

            Provide your judgment in the format:

            answer: <0 or 1>
            """)

    def _normalize_answer(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    def exact_match(self, llm_answers, gold_answers):
        matches = []
        for llm_ans, gold_ans in zip(llm_answers, gold_answers):
            matches.append(int(self._normalize_answer(llm_ans)== self._normalize_answer(gold_ans)))

        return sum(matches) / len(matches)

    def f1_score(self, llm_answers, gold_answers):
        scores = []
        for llm_ans, gold_ans in zip(llm_answers, gold_answers):
            llm_tokens = self._normalize_answer(llm_ans).split()
            gold_tokens = self._normalize_answer(gold_ans).split()
            gold_len = len(gold_tokens)
            common = []
            remaining_gold = gold_tokens.copy()
            for token in llm_tokens:
                if token in remaining_gold:
                    common.append(token)
                    remaining_gold.remove(token)

            if len(common) == 0:
                scores.append(0.0)
                continue

            precision = len(common) / len(llm_tokens)
            recall = len(common) / gold_len
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

        return sum(scores) / len(scores)

    def contains_answer(self, llm_answers, gold_answers):
        scores = []
        for llm_ans, gold_ans in zip(llm_answers, gold_answers):
            scores.append(int(self._normalize_answer(gold_ans) in self._normalize_answer(llm_ans)))

        return sum(scores) / len(scores)

    def hit_at_kwords(self, llm_answers, gold_answers, k=10):
        scores = []
        for llm_ans, gold_ans in zip(llm_answers, gold_answers):
            tokens = self._normalize_answer(llm_ans).split()
            prefix = " ".join(tokens[:k])
            scores.append(int(self._normalize_answer(gold_ans) in prefix))

        return sum(scores) / len(scores)

    def llm_score(self, llm_answers, gold_answers, queries):
        if self.llm is None:
            raise ValueError("Evaluator requires an LLM.")

        results = []
        for llm_ans, gold_ans, query in zip(llm_answers, gold_answers, queries):
            response = self.llm.predict(prompt=self.llm_prompt, query=query, gold_ans=gold_ans, llm_ans=llm_ans,)
            match = re.search(r"answer:\s*\{?([01])\}?", response)
            if match:
                score = int(match.group(1))
            else:
                score = 0

            results.append(score)

        return sum(results) / len(results)

    def evaluate(self, llm_answers, gold_answers, queries, k=10):
        results = {"exact_match": self.exact_match(llm_answers,gold_answers),
                   "f1": self.f1_score(llm_answers,gold_answers),
                   "contains_answer": self.contains_answer(llm_answers,gold_answers),
                   f"hit@{k}": self.hit_at_kwords(llm_answers,gold_answers,k)}

        if self.llm is not None:
            results["llm_score"] = self.llm_score(llm_answers,gold_answers,queries)

        return results