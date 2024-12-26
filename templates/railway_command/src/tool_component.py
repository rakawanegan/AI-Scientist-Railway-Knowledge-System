import io
import os
import re
import subprocess
import sys
from functools import wraps
from glob import glob

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel


class ToolUsageTracker:
    def __init__(self):
        self.counts = {}

    def increment(self, tool_name: str):
        if tool_name not in self.counts:
            self.counts[tool_name] = 0
        self.counts[tool_name] += 1

    def get_counts(self):
        return self.counts


# グローバルな変数ではなく、インスタンスを作成して、それを利用する
tool_usage_tracker = ToolUsageTracker()


def count_tool_usage(tool_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tool_usage_tracker.increment(tool_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MakeRailwayKnowledgePrompt:
    def __init__(self, k: int) -> None:
        self.k = k
        knowledge = ""
        for p_file in sorted(glob("data/*.md")):
            knowledge += self._load_markdown_file(p_file)
        self.sentences = knowledge.split("\n")
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        if os.path.exists("./cache/railway-knowledge-index.faiss"):
            self.index = faiss.read_index("./cache/railway-knowledge-index.faiss")
        else:
            embeddings = self._embed_text(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(self.index, "./cache/railway-knowledge-index.faiss")

    def _load_markdown_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def get_text(self) -> str:
        return "\n".join(self.sentences)

    def get_basis(self, query: str) -> str:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(
            query_embedding, self.k
        )  # first output is distance
        basis = " ".join(
            [self.sentences[i] for i in indices[0] if i < len(self.sentences)]
        )
        return basis

    def make_prompt(self, query: str) -> str:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQuery: {query}\nAnswer:"
        return prompt


class MakeRailwayIncidentPrompt:
    def __init__(self, k: int) -> None:
        self.k = k
        p_file = "./data/RailwayIncidentCase.csv"
        self.sentences = self._load_csv_file(p_file)
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        if os.path.exists("./cache/railway-incident-index.faiss"):
            self.index = faiss.read_index("./cache/railway-incident-index.faiss")
        else:
            embeddings = self._embed_text(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(self.index, "./cache/railway-incident-index.faiss")

    def _load_csv_file(self, file_path: str) -> list[str]:
        df = pd.read_csv(file_path, sep="\t", encoding="utf-16")
        texts = list()
        for i, row in df.iterrows():
            row = str(dict(row))
            row = re.sub(r"\{|\}", "", row)  # strip {}
            row = re.sub(r"'", '"', row)  # replace ' with "
            texts.append(row)
        return texts

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def get_text(self) -> str:
        return "\n".join(self.sentences)

    def get_basis(self, query: str) -> str:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(
            query_embedding, self.k
        )  # first output is distance
        basis = " ".join(
            [self.sentences[i] for i in indices[0] if i < len(self.sentences)]
        )
        return basis

    def make_prompt(self, query: str) -> str:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQuery: {query}\nAnswer:"
        return prompt


def search_word(word, mrkp):
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    matched_sentences = [sent for sent in mrkp.sentences if pattern.search(sent)]
    return "\n".join(matched_sentences)


def get_knowledge(query, mrkp):
    basis = mrkp.get_basis(query)
    return basis


def get_incident(query, mrip):
    basis = mrip.get_basis(query)
    return basis


@count_tool_usage("DuckDuckGoSearch")
def duckduckgo_search(query: str) -> str:
    search_url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "pretty": 1,
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    search_results = response.json()
    related_topics = search_results.get("RelatedTopics", [])
    formatted_results = ""
    for topic in related_topics[:5]:
        if "Text" in topic and "FirstURL" in topic:
            formatted_results += f"Title: {topic['Text']}\nURL: {topic['FirstURL']}\n\n"
    return formatted_results or "No relevant results found."


@count_tool_usage("Code")
def execute_code(code: str):
    match = re.search(
        r"```(python|sh|bash|zsh|fish|cmd|powershell)\n(.*?)```", code, re.DOTALL
    )
    if not match:
        return "Error: No supported code block found."

    code_type = match.group(1)
    code_to_execute = match.group(2)

    if code_type == "python":
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code_to_execute)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {e}"
        finally:
            sys.stdout = stdout
    else:
        shell = code_type in ["sh", "bash", "zsh", "fish"]
        executable = {"cmd": "cmd.exe", "powershell": "powershell.exe"}.get(code_type)
        try:
            result = subprocess.run(
                code_to_execute,
                shell=shell,
                capture_output=True,
                text=True,
                executable=executable,
            )
            output = (
                result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            )
        except Exception as e:
            output = f"Error: {e}"

    return output


def load_func_dict(mrkp, mrip) -> dict[str]:
    @count_tool_usage("RailwayIncidentSearcher")
    def railway_incident_searcher(query: str):
        return get_incident(query, mrip)

    @count_tool_usage("RailwayTechnicalStandardSearcher")
    def railway_technical_standard_searcher(query: str):
        return get_knowledge(query, mrkp)

    @count_tool_usage("RailwayTechnicalKeywordSearcher")
    def railway_technical_keyword_searcher(word: str):
        return search_word(word, mrkp)

    return {
        "PythonShellExecutor": execute_code,
        "RailwayIncidentSearcher": railway_incident_searcher,
        "RailwayTechnicalStandardSearcher": railway_technical_standard_searcher,
        "RailwayTechnicalKeywordSearcher": railway_technical_keyword_searcher,
        "DuckDuckGoWebSearcher": duckduckgo_search,
    }
