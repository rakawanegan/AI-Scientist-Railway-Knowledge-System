import csv
from typing import List, Dict, Union
import json
import os.path as osp
from pymupdf4llm import to_markdown

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS


idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

以下は、これまでに生成されたアイデアです：

'''
{prev_ideas_string}
'''

提供されたコードを使って実行可能な、次に影響力があり創造的な研究実験や方向性のアイデアを考えてください。
追加のリソースやデータセットにはアクセスできないことに注意してください。
どのアイデアも特定のトレーニングデータセットやモデルに過剰適合せず、広い意味での意義を持つようにしてください。

以下の形式で回答してください：

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

<THOUGHT> では、まずアイデアの直感や動機について簡単に説明してください。
アイデアの高レベルな計画、必要な設計選択、理想的な実験結果を詳述してください。
また、このアイデアが既存のものとどのように異なるかを正当化してください。

<JSON> では、次のフィールドを含むJSON形式で新しいアイデアを記載してください：
- "Name": アイデアの簡潔な記述子。小文字、スペースなし、アンダースコアを許可。
- "Title": アイデアのタイトル。レポート作成で使用されます。
- "Experiment": 実装の概要。例：追加または変更が必要な関数、結果の取得方法、など。
- "Interestingness": 1から10の範囲で評価（最低から最高）。
- "Feasibility": 1から10の範囲で評価（最低から最高）。
- "Novelty": 1から10の範囲で評価（最低から最高）。

評価は慎重かつ現実的に行ってください。
このJSONは自動的に解析されるため、フォーマットが正確であることを確認してください。
{num_reflections}回のラウンドでアイデアを繰り返し検討することができますが、すべてのラウンドを使用する必要はありません。
"""

idea_reflection_prompt = """ラウンド {current_round}/{num_reflections}。
あなたが作成したアイデアの質、新規性、実現可能性を慎重に考慮してください。
重要だと思う他の要素も評価に含めてください。
アイデアが明確で簡潔であることを確認し、JSONが正しい形式であることを確保してください。
物事を過度に複雑にしないでください。
次の試行では、アイデアを改良し改善するように努めてください。
重大な問題がない限り、元のアイデアの趣旨を守ってください。

以前と同じ形式で回答してください：
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

改善すべき点がない場合は、思考の後に前回のJSONを正確に繰り返し、「I am done」を思考の最後に記載し、JSONの前に記載してください。
変更を加えない場合にのみ「I am done」を含めてください。"""




def search_for_papers(index_file: str) -> List[Dict[str, str]]:
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


def generate_ideas(base_dir: str, client=None, model=None, skip_generation=False, max_num_generations=10, num_reflections=5) -> List[Dict[str, str]]:

    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

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
    related_indices = search_for_papers(INDEX_FILE)

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
