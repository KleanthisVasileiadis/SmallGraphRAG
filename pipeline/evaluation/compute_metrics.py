import re
from llama_index.core.prompts import PromptTemplate

#Normalizing the output
def normalize_answer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def exact_match(llms_answer, gold_answer):
    return normalize_answer(llms_answer) == normalize_answer(gold_answer)

def f1_score(llms_answer, gold_answer):
    #Tokenizing the answers
    llm_tokens = normalize_answer(llms_answer).split()
    gold_tokens = normalize_answer(gold_answer).split()

    len_gold_tokkens = len(gold_tokens)

    common_tokens = []
    #Keeping the common words of both answers
    for token in llm_tokens:
        if token in gold_tokens:
            common_tokens.append(token)
            gold_tokens.remove(token)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(llm_tokens)
    recall = len(common_tokens) / len_gold_tokkens
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def llm_evaluation(llm, llm_answers, gold_answers, queries):

    results = []
    i = 0
    for llm_ans, gold_ans, query in zip(llm_answers, gold_answers, queries):
        
        prompt = f"""
        You are an expert evaluator. Your role is to determine whether the model's answer 
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

        """

        prompt_temp = PromptTemplate(prompt)

        response = llm.predict(prompt=prompt_temp, query=query, gold_ans=gold_ans, llm_ans=llm_ans)
        # print(f"llm response: {response}\n")
        match = re.search(r"answer:\s*\{?([01])\}?", response)
        if match:
            val = int(match.group(1))
        else:
            val = 0 
        print(f"result[{i}] = {val}")
        results.append(val)
        i += 1

    return sum(results) / len(results)

def contains_answer(answer, gold_answer):
    return int(normalize_answer(gold_answer) in normalize_answer(answer))

def hit_at_kwords(answer, gold_answer, k):
    tokens = normalize_answer(answer).split()
    prefix = " ".join(tokens[:k])
    return int(normalize_answer(gold_answer) in prefix)