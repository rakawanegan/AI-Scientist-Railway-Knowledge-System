import os
import os.path as osp
import json
import random
import re
from typing import Dict, List
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

class AccidentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with necessary configurations."""
        self.regulation_summary = self._load_summary("summary.txt")

    def _load_summary(self, filepath: str) -> List[str]:
        """Load regulation summary from a text file."""
        with open(filepath, 'r') as file:
            return file.readlines()

    def identify_relevant_regulations(self, accident_report: str) -> str:
        """Identify the relevant regulation for a given accident report."""
        # Placeholder for AI-based identification logic
        for regulation in self.regulation_summary:
            if any(keyword in accident_report for keyword in regulation.split()):
                return regulation.strip()
        return "No matching regulation found."

    def analyze_accident(self, accident_report: str, regulation_content: str) -> str:
        """Analyze the accident and provide insights."""
        # Placeholder for AI-based analysis logic
        analysis = f"Analyzing the report: {accident_report[:50]}\n"
        analysis += f"Matched Regulation: {regulation_content}\n"
        analysis += "Proposed Improvement: Enhance safety protocols related to identified issues."
        return analysis


def extract_total_reports(content: str) -> int:
    """Extract the total number of reports from the content."""
    matches = re.findall(r'\*\*Report\*\*', content)
    return len(matches)


def get_random_report(content: str, total_reports: int) -> str:
    """Select a random accident report from the list."""
    reports = content.split("\n\n")
    return random.choice(reports) if reports else ""


def display_report_summary(report: str):
    """Display a summary of the selected accident report."""
    summary = re.search(r'Title: (.+?)\nDescription: (.+)', report)
    if summary:
        print(f"Title: {summary.group(1)}")
        print(f"Description: {summary.group(2)}")
    else:
        print("Unable to parse the report.")


def visualize_text_keywords(texts: List[str]):
    """Visualize keywords from text using WordCloud and PCA."""
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Generate WordCloud
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().sum(axis=0)
    word_freq = dict(zip(feature_array, tfidf_scores))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Keyword Visualization via WordCloud")
    plt.show()

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Plot PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, edgecolors='k')
    plt.title("TF-IDF Features Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def analyze_random_accident():
    """Analyze a randomly selected accident report."""
    with open("accident-report.md", "r") as file:
        content = file.read()

    total_reports = extract_total_reports(content)
    if total_reports == 0:
        print("No reports found.")
        return

    selected_report = get_random_report(content, total_reports)
    display_report_summary(selected_report)

    analyzer = AccidentAnalyzer()
    relevant_regulation = analyzer.identify_relevant_regulations(selected_report)

    with open(f"{relevant_regulation}.md", "r") as reg_file:
        regulation_content = reg_file.read()

    analysis = analyzer.analyze_accident(selected_report, regulation_content)
    print("\nAnalysis Result:\n", analysis)

    # Visualization Step
    print("\nVisualizing Report Content...")
    visualize_text_keywords([selected_report, regulation_content])


if __name__ == "__main__":
    analyze_random_accident()
