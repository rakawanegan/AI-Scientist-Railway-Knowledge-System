import csv
from typing import List, Dict
import json
import os.path as osp
from pymupdf4llm import to_markdown

def read_related_indices(index_file: str) -> List[Dict[str, str]]:
    """
    Reads and parses the Index.csv file containing related document indices.

    Args:
        index_file (str): Path to the Index.csv file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with keys 'Path', 'Title', and 'Summary'.

    Expected CSV format:
    Path,Title,Summary
    /path/to/document1.pdf,Title of Document 1,Summary of Document 1
    /path/to/document2.pdf,Title of Document 2,Summary of Document 2
    """
    indices = []
    try:
        with open(index_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'Path' in row and 'Title' in row and 'Summary' in row:
                    indices.append({"Path": row['Path'], "Title": row['Title'], "Summary": row['Summary']})
    except FileNotFoundError:
        print(f"Error: {index_file} not found.")
    except Exception as e:
        print(f"An error occurred while reading {index_file}: {e}")
    return indices

def select_documents(indices: List[Dict[str, str]], keyword: str) -> List[Dict[str, str]]:
    """
    Filters the indices to find documents matching the given keyword.

    Args:
        indices (List[Dict[str, str]]): List of document indices.
        keyword (str): Keyword to filter by.

    Returns:
        List[Dict[str, str]]: Filtered list of indices matching the keyword.
    """
    return [index for index in indices if keyword.lower() in index['Summary'].lower()]

def read_pdf_content(pdf_path: str) -> str:
    """
    Reads the content of a PDF file using pymupdf4llm.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content of the PDF in markdown format.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            return to_markdown(pdf_file)
    except FileNotFoundError:
        print(f"Error: PDF file {pdf_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading {pdf_path}: {e}")
    return ""

def generate_ideas(base_dir: str, filtered_indices: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generate new ideas based on the filtered documents.

    Args:
        base_dir (str): Base directory to save ideas.
        filtered_indices (List[Dict[str, str]]): Filtered document indices.

    Returns:
        List[Dict[str, str]]: Generated ideas with additional metadata.
    """
    ideas = []
    for index in filtered_indices:
        pdf_content = read_pdf_content(index['Path'])
        if pdf_content:
            idea = {
                "Path": index['Path'],
                "Title": index['Title'],
                "Summary": index['Summary'],
                "ExtractedContent": pdf_content,  # Full extracted content
            }
            ideas.append(idea)
    # Save ideas to a JSON file
    ideas_file = osp.join(base_dir, "generated_ideas.json")
    with open(ideas_file, "w", encoding="utf-8") as f:
        json.dump(ideas, f, indent=4, ensure_ascii=False)
    print(f"Ideas saved to {ideas_file}")
    return ideas

def check_idea_novelty(ideas: List[Dict[str, str]], base_dir: str, client=None, model=None, max_num_iterations=10) -> List[Dict[str, str]]:
    """
    Check the novelty of each idea without relying on external APIs.

    Args:
        ideas (List[Dict[str, str]]): List of generated ideas.
        base_dir (str): Base directory where results are saved.
        client: Not used, included for compatibility.
        model: Not used, included for compatibility.
        max_num_iterations (int): Number of iterations for novelty checks (for compatibility).

    Returns:
        List[Dict[str, str]]: Ideas annotated with their novelty status.
    """
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Title']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                # Simulated novelty check logic based on text analysis
                extracted_content = idea.get("ExtractedContent", "")
                summary = idea.get("Summary", "")

                if len(extracted_content.split()) > 100 and summary not in extracted_content:
                    novel = True
                    break

            except Exception as e:
                print(f"Error during novelty check: {e}")
                continue

        idea["novel"] = novel

    # Save updated ideas
    ideas_file = osp.join(base_dir, "ideas_with_novelty.json")
    with open(ideas_file, "w", encoding="utf-8") as f:
        json.dump(ideas, f, indent=4, ensure_ascii=False)
    print(f"Updated ideas with novelty saved to {ideas_file}")
    return ideas

if __name__ == "__main__":
    INDEX_FILE = "data/related_search/Index.csv"
    SEARCH_KEYWORD = "machine learning"  # Example keyword
    BASE_DIR = "data/output"

    # Read the related indices
    related_indices = read_related_indices(INDEX_FILE)

    # Display all indices
    if related_indices:
        print("All Related Indices:")
        for index in related_indices:
            print(f"Path: {index['Path']} | Title: {index['Title']} | Summary: {index['Summary']}")

        # Select documents based on keyword
        filtered_indices = select_documents(related_indices, SEARCH_KEYWORD)
        print("\nFiltered Indices:")
        for index in filtered_indices:
            print(f"Path: {index['Path']} | Title: {index['Title']} | Summary: {index['Summary']}")

        # Generate ideas from filtered indices
        generated_ideas = generate_ideas(BASE_DIR, filtered_indices)
        print("\nGenerated Ideas:")
        for idea in generated_ideas:
            print(json.dumps(idea, indent=4, ensure_ascii=False))

        # Check novelty of generated ideas
        novel_ideas = check_idea_novelty(generated_ideas, BASE_DIR)
        print("\nNovelty-Checked Ideas:")
        for idea in novel_ideas:
            print(json.dumps(idea, indent=4, ensure_ascii=False))
    else:
        print("No related indices found.")
