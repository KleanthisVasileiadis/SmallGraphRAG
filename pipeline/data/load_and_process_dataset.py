from datasets import load_from_disk
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from datasets import concatenate_datasets

def convert_to_docs(texts):
    #Convert data into LlamaIndex Document objects
    documents = [Document(text=text) for text in texts]
    return documents

def convert_to_nodes(documents):
    splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Number Of Nodes: {len(nodes)}\n")
    return nodes

def load_HotPotQA(type, num=1):
    dataset = load_from_disk("/home/plas_vasileiadis/GraphRAG/pipeline/data/raw_datasets/HotPotQA_fullwiki")
    if type == "all":
        per_type = num // 3
        remainder = num % 3
        counts = {"easy": per_type,"medium": per_type,"hard": per_type}
        for t in ["easy", "medium", "hard"][:remainder]:
            counts[t] += 1

        subsets = []
        for t in ["easy", "medium", "hard"]:
            subset_t = dataset.filter(lambda x, t=t: x["level"] == t).select(range(counts[t]))
            subsets.append(subset_t)

        subset = concatenate_datasets(subsets).shuffle(seed=42)
    else:
        subset = dataset.filter(lambda x: x["level"] == type).select(range(num))

    questions = []
    answers = []
    texts = []
    easy = 0 
    medium = 0
    hard = 0
    for item in subset:
        questions.append(item["question"])
        answers.append(item["answer"])
        if item["level"] == "easy":
            easy += 1
        elif item["level"] == "medium":
            medium += 1
        else:
            hard += 1

        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        text = ""
        for title, sentence in zip(titles, sentences):
            text += f"{title}: " + " ".join(sentence)

        texts.append(text)

    print(f"easy: {easy} | medium: {medium} | hard: {hard}\n")
    return texts, questions, answers

def load_to_nodes_HotPotQA(type, num):
    texts, questions, answers = load_HotPotQA(type, num)
    documents = convert_to_docs(texts)
    nodes = convert_to_nodes(documents)
    return nodes, questions, answers

def load_MuSiQue(num=1, answerable_only=True):
    dataset = load_from_disk("/home/plas_vasileiadis/GraphRAG/pipeline/data/raw_datasets/MuSiQue")
    if answerable_only:
        dataset = dataset.filter(lambda x: x["answerable"])

    subset = dataset.select(range(min(num, len(dataset))))
    questions = []
    answers = []
    texts = []
    for item in subset:
        questions.append(item["question"])
        answers.append(item["answer"])
        text = ""
        for paragraph in item["paragraphs"]:
            title = paragraph["title"]
            paragraph_text = paragraph["paragraph_text"]
            text += f"{title}: {paragraph_text}\n"
        
        texts.append(text)

    print(f"Loaded {len(texts)} examples")
    return texts, questions, answers


def load_to_nodes_MuSiQue(num):
    texts, questions, answers = load_MuSiQue(num)
    documents = convert_to_docs(texts)
    nodes = convert_to_nodes(documents)
    return nodes, questions, answers

def load_2WikiMultiHopQA(q_type="all", num=1):

    dataset = load_from_disk("/home/plas_vasileiadis/GraphRAG/pipeline/data/raw_datasets/2WikiMultiHopQA")
    if q_type == "all":
        subset = dataset.select(range(min(num, len(dataset))))
    else:
        subset = dataset.filter(lambda x: x["type"] == q_type).select(range(num))

    questions = []
    answers = []
    texts = []
    type_counter = {}
    for item in subset:
        questions.append(item["question"])
        answers.append(item["answer"])
        curr_type = item["type"]
        if curr_type not in type_counter:
            type_counter[curr_type] = 0

        type_counter[curr_type] += 1
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        text = ""
        for title, sentence_list in zip(titles, sentences):
            text += (f"{title}: "+ " ".join(sentence_list)+ "\n")

        texts.append(text)
    print("Question types:")
    for k, v in type_counter.items():
        print(f"{k}: {v}")

    print(f"\n")
    return texts, questions, answers

def load_to_nodes_2WikiMultiHopQA(q_type, num):
    texts, questions, answers = load_2WikiMultiHopQA(q_type,num)
    documents = convert_to_docs(texts)
    nodes = convert_to_nodes(documents)
    return nodes, questions, answers