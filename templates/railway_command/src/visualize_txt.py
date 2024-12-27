import os
import re
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from langchain_openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_embedding(text, model="text-embedding-3-small"):
    client = OpenAI()
    text = text.replace("\n", " ")
    if len(text) == 0:
        return [0] * 1536
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def preprocess_and_save_embeddings(
    data, filename="railway-knowledge-text-embedding-3-small.npy"
):
    data = np.array(data)
    if len(data.shape) == 3:
        n, _, dim = data.shape
        data = data.reshape((n, dim))
    np.save(filename, data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(data)
    return embedding


def plot_umap(embedding):
    cmap_bounds = [0, 42, 92, 163, 178, len(embedding)]
    segment_names = [
        "Section 1 (1-42)",
        "Section 2 (43-92)",
        "Section 4 (93-163)",
        "Section 5 (164-178)",
        "Section 9 (179+)",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(cmap_bounds, cmap.N)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=list(range(len(embedding))),
        cmap=cmap,
        norm=norm,
        s=5,
    )
    cbar = plt.colorbar(scatter, ticks=[21, 67, 127, 170, 189])
    cbar.ax.set_yticklabels(segment_names)

    plt.title("UMAP Projection with Custom Band Names")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()


def read_markdown_as_plain_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    content = re.sub(r"#\s+", "", content)
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"\*(.*?)\*", r"\1", content)
    content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
    content = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", content)
    content = re.sub(r"`(.*?)`", r"\1", content)
    content = re.sub(r"\n+", "\n", content).strip()
    return content


def tokenize_japanese(text):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [
        token.surface
        for token in tokens
        if token.part_of_speech.split(",")[0] == "名詞"
    ]
    tokens = [token for token in tokens if not re.search(r"\d", token)]
    return " ".join(tokens)


def extract_keywords_with_tfidf(text, top_n=5):
    tokenized_text = tokenize_japanese(text)
    documents = [tokenized_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    tfidf_scores = pd.DataFrame({"word": feature_names, "value": scores})
    return tfidf_scores.sort_values(by="value", ascending=False).head(top_n)


def extract_keywords_with_bm25(text, top_n=5):
    tokenized_text = tokenize_japanese(text).split()
    documents = [tokenized_text]
    bm25 = BM25Okapi(documents)
    scores_dict = defaultdict(float)
    for word in tokenized_text:
        scores_dict[word] += bm25.get_scores([word])[0]
    bm25_scores = pd.DataFrame(list(scores_dict.items()), columns=["word", "value"])
    return bm25_scores.sort_values(by="value", ascending=False).head(top_n)


def generate_wordcloud(text, extract_keywords, top_n=50):
    p_font = os.path.abspath("../../font_assets/NotoSansCJKjp-Regular.otf")
    top_keywords = extract_keywords(text, top_n)
    keyword_dict = dict(zip(top_keywords["word"], top_keywords["value"]))
    wordcloud = WordCloud(
        font_path=p_font, background_color="white", width=800, height=400
    ).generate_from_frequencies(keyword_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return top_keywords


def analyze_technical_terms(
    p_railway_txt, p_another_domain_txts, extract_keywords_list
):
    p_font = os.path.abspath("../font_assets/NotoSansCJKjp-Regular.otf")
    railway_txt = read_markdown_as_plain_text(p_railway_txt)
    dict_tech_terms = dict()

    for extract_keywords in extract_keywords_list:
        set_railway_top_keywords = set(extract_keywords(railway_txt, top_n=50)["word"])
        list_set_another_domain_top_keywords = list()
        for p_another_domain_txt in p_another_domain_txts:
            another_domain_txt = read_markdown_as_plain_text(p_another_domain_txt)
            list_set_another_domain_top_keywords.append(
                set(extract_keywords(another_domain_txt, top_n=50)["word"])
            )
        set_another_domain_top_keywords = set.union(
            *list_set_another_domain_top_keywords
        )
        tech_terms = set_railway_top_keywords - set_another_domain_top_keywords
        dict_tech_terms[extract_keywords.__name__] = tech_terms

    intersection_count = len(set.intersection(*dict_tech_terms.values()))
    union_count = len(set.union(*dict_tech_terms.values()))

    print(f"Number of Common Technical Terms (Intersection): {intersection_count}")
    print(f"Total Unique Technical Terms (Union): {union_count}")

    wordcloud = WordCloud(
        font_path=p_font, background_color="white", width=800, height=400
    ).generate_from_frequencies(
        {tech_term: 1 for tech_term in set.union(*dict_tech_terms.values())}
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
