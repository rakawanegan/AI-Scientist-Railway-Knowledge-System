import os
import re

import autogen
import pandas as pd
import yaml
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from .main import evaluate_by_llm_with_criteria
from .utils import load_criteria_with_weights


def initialize_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def load_yaml_config(file_path: str) -> dict[str, str]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"警告: ファイルが見つかりません: {file_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"警告: YAMLの読み込みエラー: {e}")
        return {}


def setup_agents(p_eval_configs, llm_config):
    criterions = [load_yaml_config(path) for path in p_eval_configs]
    agents = [
        autogen.AssistantAgent(
            name=criterion["name"],
            llm_config=llm_config,
            system_message=(
                f"あなたは採点者です。"
                f" この基準の重みは{criterion['weight']}です。以下の説明に従い、公平かつ正確に採点を行ってください。\n\n"
                f"説明: {criterion['description']}"
            ),
        )
        for criterion in criterions
    ]
    group_chat = autogen.GroupChat(
        agents=agents, messages=list(), max_round=len(agents) + 5
    )
    return autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)


def inference(manager, question):
    try:
        manager.initiate_chat(message=question, recipient=manager.groupchat.agents[0])
        output = manager.groupchat.messages[-1]["content"]
    except Exception as e:
        output = f"Error: {e}"
    return output


def evaluate_by_multiagent_llm(manager, pred, answer, question=None):
    prompt_template = """
    # 評価基準
    問題と解答を参照しながら採点対象を採点してください。
    各エージェントに適切に採点業務を割り振って、最終的な点数を出力してください。
    例えば、計算箇所がない問題に対して数値計算を評価するエージェントに採点業務を振る必要はないです。
    また、各項目について0〜5点のスコアを付け、理由を添えてください。
    スコアについて、0点は全く正確でない場合、3点は一部正確だが誤りも含む場合、5点は完全に正確である場合とします。

    # 無視事項
    以下の内容については採点不要です。
    - 他の評価基準
    - 問題文や解答の情報

    # 注意点
    データが見つからなかったという文言やエラー文が含まれている文章はスコアを0点にしてください。
    スコアの箇所は必ず【】で囲って明示してください。スコア以外の部分には【】を使用しないでください。
    """

    if question:
        base_prompt = PromptTemplate(
            input_variables=["pred", "answer", "question"],
            template=f"""
            問題: "{{question}}"
            採点対象: "{{pred}}"
            解答: "{{answer}}",
            {prompt_template}
            """,
        )
    else:
        base_prompt = PromptTemplate(
            input_variables=["pred", "answer"],
            template=f"""
            採点対象: "{{pred}}"
            解答: "{{answer}}",
            {prompt_template}
            """,
        )
    prompt = base_prompt.format(pred=pred, answer=answer, question=question)
    return inference(manager, prompt)


def evaluate_by_llm(llm, criterions, pred, answer, question):
    description = "\n".join(
        [f'- {criterion["description"]}' for criterion in criterions]
    )
    base_template = f"""
    # 評価基準
    問題と解答を参照しながら採点してください。
    採点基準については以下の通りです。
    {description}
    最終的な採点結果は【スコア】点のように角括弧で囲って出力してください。
    """
    prompt = PromptTemplate(
        input_variables=["pred", "answer", "question"],
        template=base_template
        + """
        問題: "{{question}}"
        採点対象: "{{pred}}"
        解答: "{{answer}}",
        """,
    )
    prompt = prompt.format(pred=pred, answer=answer, question=question)
    return llm.predict(prompt)


def process_dataframe(df, evaluate_function, output_file, **kwargs):
    predict_cols = ["GPT4", "GPTs-markdown", "Rinna+GPT4", "BERT+GPT4", "BERT+Rinna"]
    df_answer = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        query = row["問題"]
        answer = row["答え"]
        for predict_col in predict_cols:
            predict = row[predict_col]
            output = evaluate_function(
                pred=predict, answer=answer, question=query, **kwargs
            )
            suggest_score = re.search(r"【([\d.]+)】", output)
            score = float(suggest_score.group(1)) if suggest_score else "N/A"
            row[predict_col] = score
        df_answer.append(row)
    df_answer = pd.DataFrame(df_answer)
    df_answer.to_csv(output_file, index=False)
    return df_answer


# Example usage of functions
def main():
    llm = initialize_llm()
    llm_config = {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
        ]
    }
    criteria_with_weights = load_criteria_with_weights(
        [
            "./docs/evals/accuracy.yaml",
            "./docs/evals/calculation.yaml",
            "./docs/evals/evidence.yaml",
            "./docs/evals/expertise.yaml",
            "./docs/evals/expression.yaml",
            "./docs/evals/relevance.yaml",
        ]
    )
    manager = setup_agents(
        [
            "./docs/evals/accuracy.yaml",
            "./docs/evals/calculation.yaml",
            "./docs/evals/evidence.yaml",
            "./docs/evals/expertise.yaml",
            "./docs/evals/expression.yaml",
            "./docs/evals/relevance.yaml",
        ],
        llm_config,
    )

    df = pd.read_csv("data/Result.csv")

    print("Processing GenAI...")
    process_dataframe(
        df,
        lambda **kwargs: evaluate_by_llm(llm, criteria_with_weights, **kwargs),
        "data/Answer-GenAI.csv",
    )

    print("Processing Multi-Agent...")
    process_dataframe(
        df,
        lambda **kwargs: evaluate_by_multiagent_llm(manager, **kwargs),
        "data/Answer-MultiAgent.csv",
    )

    print("Processing Agentless...")
    process_dataframe(
        df,
        lambda **kwargs: evaluate_by_llm_with_criteria(
            **kwargs, llm=llm, criteria_with_weights=criteria_with_weights
        ),
        "data/Answer-Agentless.csv",
    )
