import os
import time
from contextlib import contextmanager

import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from .main import evaluate_by_llm_with_criteria, make_agent, plot_tools_count
from .tool_component import tool_usage_tracker
from .utils import load_criteria_with_weights


def setup_environment(k, tool_configs, eval_configs, react_prompt):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = make_agent(tool_configs, react_prompt, k, llm)
    criteria_with_weights = load_criteria_with_weights(eval_configs)
    return agent, criteria_with_weights


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def inference(agent, question):
    try:
        output = agent.run(question)
    except Exception as e:
        output = f"Error: {e}"
    return output


def evaluate_responses(agent, criteria_with_weights, query, answer):
    with timer("inference"):
        predict = inference(agent, query)

    dict_eval = evaluate_by_llm_with_criteria(
        predict,
        answer,
        llm=agent.llm,
        question=query,
        criteria_with_weights=criteria_with_weights,
    )
    return predict, dict_eval


def batch_inference(
    agent, criteria_with_weights, input_file, output_file, all_run=True
):
    df = pd.read_csv(input_file)
    df_answer = list()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if not all_run:
            break
        row = dict(row)
        query = row["問題"]
        answer = row["答え"]
        predict = inference(agent, query)
        dict_eval = evaluate_by_llm_with_criteria(
            pred=predict,
            answer=answer,
            llm=agent.llm,
            criteria_with_weights=criteria_with_weights,
        )
        print(dict_eval)
        dict_eval.update(
            {
                "answer": predict,
            }
        )
        row.update(dict_eval)
        df_answer.append(row)
    df_answer = pd.DataFrame(df_answer)
    df_answer.to_csv(output_file, index=False)
    return df_answer


def visualize_tool_usage():
    current_tool_usage_counts = tool_usage_tracker.get_counts()
    if len(current_tool_usage_counts) > 0:
        plot_tools_count(current_tool_usage_counts)


def main():
    # Configuration
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

    # Environment setup
    agent, criteria_with_weights = setup_environment(
        k, p_tool_configs, p_eval_configs, p_react_prompt
    )

    # Single query evaluation
    query = """
状況：
あなたは鉄道会社の保守管理責任者です。石勝線での列車脱線事故を受けて、同様の事故を防ぐための点検体制の見直しを任されました。

設問：
省令第87条（施設及び車両の保全）に基づき、減速機の吊りピンについて、どのような点検項目と頻度を設定すべきですか？具体的に3つ以上挙げてください。
点検時に「軽微な摩耗」が見つかった場合、運行継続の判断基準をどのように設定しますか？省令の安全確保の観点から説明してください。
"""

    answer = """
点検は3段階で実施することが望ましいです。
まず、毎日の運行前には基本的な日常点検を行います。これには吊りピンの位置確認、ナットの緩みチェック、異音や異常振動の確認が含まれます。
次に、月次の定期点検では、デジタルノギスを使用した吊りピンの摩耗量測定、割りピンの状態確認、ナットの締め付けトルクチェック、そして周辺部品との干渉がないかの確認を行います。
さらに、3ヶ月ごとの精密検査では、非破壊検査による吊りピン内部の亀裂検査、溝付き六角ナットの詳細な摩耗状態検査、減速機取付部全体の応力測定を実施します。
"""

    predict, dict_eval = evaluate_responses(agent, criteria_with_weights, query, answer)
    print(predict)
    print()

    for k, v in dict_eval.items():
        print(f"{k} {v}")
        print()

    # Batch inference
    batch_file = "data/Question.csv"
    output_file = "data/Answer.csv"
    df_answer = batch_inference(agent, criteria_with_weights, batch_file, output_file)

    print(df_answer)

    # Visualize tool usage
    visualize_tool_usage()
