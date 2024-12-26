import re

import matplotlib.pyplot as plt
import pandas as pd
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from .tool_component import (
    MakeRailwayIncidentPrompt,
    MakeRailwayKnowledgePrompt,
    load_func_dict,
    tool_usage_tracker,
)
from .utils import load_criteria_with_weights, load_prompt, load_tool


def make_agent(p_tool_configs, p_react_prompt, k, llm):
    mrkp = MakeRailwayKnowledgePrompt(k=k)
    mrip = MakeRailwayIncidentPrompt(k=k)

    func_dict = load_func_dict(mrkp, mrip)
    tools = [load_tool(p_tool_config, func_dict) for p_tool_config in p_tool_configs]
    react_prompt = load_prompt(p_react_prompt)
    prompt = PromptTemplate(
        input_variables=["task", "tools"],
        template=react_prompt,
    )
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        early_stopping_method="generate",
    )
    return agent


def evaluate_by_llm_with_criteria(
    pred, answer, llm, question=None, criteria_with_weights=None
):
    """
    LLMでの評価を実行し、指定された重みに基づいてスコアを計算する関数。

    Parameters:
    pred (str): 採点対象の答案。
    answer (str): 正解または模範解答。
    llm (LLM): 使用する言語モデル。
    question (str, optional): 問題文。デフォルトはNone。
    criteria_with_weights (list of dict): 評価基準、重み、説明を含むリスト。
        例: [{"name": "妥当性", "weight": 0.3, "description": "詳細な説明文"}, ...]

    Returns:
    dict: 評価結果を含む辞書。
    """
    if criteria_with_weights is None:
        raise ValueError("評価基準と重みを含むリストまたは辞書を指定してください。")

    # Normalize weights
    total_weight = sum(item["weight"] for item in criteria_with_weights)
    for item in criteria_with_weights:
        item["weight"] /= total_weight

    base_prompt_template = load_prompt("./docs/eval_base_prompt.md")

    scores = dict()
    feedbacks = list()

    for item in criteria_with_weights:

        prompt_template = base_prompt_template.format(
            criterion_description=item["description"]
        )

        if question is not None:
            prompt = PromptTemplate(
                input_variables=["pred", "answer", "question"],
                template=f"""
                問題: "{{question}}"
                答案: "{{pred}}"
                解答: "{{answer}}",
                {prompt_template}
                """,
            )
        else:
            prompt = PromptTemplate(
                input_variables=["pred", "answer"],
                template=f"""
                答案: "{{pred}}"
                解答: "{{answer}}",
                {prompt_template}
                """,
            )

        # Use the LLM to evaluate for the current name
        chain = LLMChain(llm=llm, prompt=prompt)
        feedback = chain.run(
            pred=pred, answer=answer, question=question if question else None
        )
        print(feedback.replace("\n", ""))
        suggest_score = re.search(r"【([\d.]+)】", feedback)
        score = float(suggest_score.group(1)) if suggest_score else "N/A"
        scores[item["name"]] = score
        feedbacks.append(feedback)

    # Calculate the weighted average score
    weighted_scores = [
        item["weight"] * scores[item["name"]]
        for item in criteria_with_weights
        if isinstance(scores[item["name"]], (int, float))
    ]
    final_score = sum(weighted_scores) if weighted_scores else 0

    # Compile final feedback
    final_feedback = "\n".join(feedbacks)
    final_feedback = final_feedback.replace("\n\n", "\n")
    scores.update(
        {"final_feedback": final_feedback, "final_score": round(final_score, 2)}
    )
    return scores


def plot_tools_count(current_tool_usage_counts: dict):
    current_tool_usage_counts = {
        k: v / sum(current_tool_usage_counts.values())
        for k, v in current_tool_usage_counts.items()
    }
    tool_usage_df = pd.DataFrame(
        list(current_tool_usage_counts.items()), columns=["Tool", "Usage_Count"]
    )

    plt.figure(figsize=(10, 6))
    plt.bar(tool_usage_df["Tool"], tool_usage_df["Usage_Count"])
    plt.xlabel("Tool")
    plt.ylabel("Usage Ratio")
    plt.title("Usage Ratio of Each Tool")
    plt.show()


def main():
    question = ""
    answer = ""

    k = 3
    p_tool_configs = [
        "./docs/tools/code.yaml",
        "./docs/tools/incident_rag.yaml",
        "./docs/tools/knowledge_rag.yaml",
        "./docs/tools/raw_text.yaml",
        "./docs/tools/search.yaml",
    ]
    p_eval_configs = [
        "./docs/evals/accuracy.yaml",
        "./docs/evals/calculation.yaml",
        "./docs/evals/evidence.yaml",
        "./docs/evals/expertise.yaml",
        "./docs/evals/expression.yaml",
        "./docs/evals/relevance.yaml",
    ]
    p_react_prompt = "./docs/react_base_prompt.md"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = make_agent(p_tool_configs, p_react_prompt, k, llm)
    criteria_with_weights = load_criteria_with_weights(p_eval_configs)

    prediction = agent.run(question)
    print(prediction)

    eval_result = evaluate_by_llm_with_criteria(
        prediction,
        answer,
        llm,
        question=question,
        criteria_with_weights=criteria_with_weights,
    )
    print(eval_result)

    # ツール使用状況の可視化
    current_tool_usage_counts = tool_usage_tracker.get_counts()
    if len(current_tool_usage_counts) > 0:
        plot_tools_count(current_tool_usage_counts)


if __name__ == "__main__":
    main()
