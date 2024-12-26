import os
import random
import re

import pandas as pd
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI


def extract_total_reports(content):
    """報告書の総件数を抽出"""
    match = re.search(r"全(\d+)件の報告書", content)
    if match:
        return int(match.group(1))
    return 0


def get_random_report(content, total_reports):
    """ランダムな報告書を1件取得"""
    reports = content.split("\n## ")[1:]
    selected_report = random.choice(reports)

    report_number = selected_report.split("\n")[0].strip()
    if report_number.startswith("報告書番号: "):
        report_number = report_number[len("報告書番号: ") :]

    print(f"選択された報告書: {report_number} (全{total_reports}件中)")
    full_report = f"## 報告書番号: {selected_report}"
    return full_report


def display_report_summary(report):
    """報告書の内容をわかりやすく表示"""
    print("\n" + "=" * 50)
    print("選択された報告書の内容")
    print("=" * 50)

    report_number = re.search(r"報告書番号: (.+?)\n", report)
    if report_number:
        print(f"\n📄 報告書番号: {report_number.group(1)}")

    print("\n📌 基本情報:")
    basic_info = {
        "発生年月日": None,
        "区分": None,
        "発生場所": None,
        "事業者": None,
        "都道府県": None,
    }
    for key in basic_info.keys():
        match = re.search(f"\\*\\*{key}\\*\\*: (.+?)\\n", report)
        if match:
            print(f"- {key}: {match.group(1)}")

    print("\n🚨 事故詳細:")
    accident_details = {
        "事故等種類": None,
        "踏切区分": None,
        "人の死傷": None,
        "死傷者数": None,
    }
    for key in accident_details.keys():
        match = re.search(f"\\*\\*{key}\\*\\*: (.+?)\\n", report)
        if match:
            print(f"- {key}: {match.group(1)}")

    print("\n📝 概要:")
    if "#### 概要" in report:
        overview = report.split("#### 概要")[1].split("####")[0].strip()
        print(overview)

    print("\n🔍 原因:")
    if "#### 原因" in report:
        cause = report.split("#### 原因")[1].split("###")[0].strip()
        print(cause)

    print("\n" + "=" * 50 + "\n")


class AccidentAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def identify_relevant_regulations(self, accident_report):
        """事故報告書に関連する省令ファイルを特定"""
        try:
            with open("docs/summary.txt", "r", encoding="utf-8") as f:
                syourei_content = f.read()

            prompt = f"""
            以下の事故報告書を分析し、最も関連性の高い規定ファイルを特定してください。

            事故報告書:
            {accident_report}

            規定ファイルの候補:
            {syourei_content}

            回答形式：
            選択されたファイル: [01-09.mdまでのファイル名のみ、<>の中身のみを出力、<>は出力しない]
            選択理由:
            [具体的な理由の説明]
            """

            result = self.llm.invoke([HumanMessage(content=prompt)])
            print("\n=== 関連省令の特定結果 ===")
            print(result.content)
            print("=" * 50)
            return result.content
        except Exception as e:
            print(f"Error in identify_relevant_regulations: {str(e)}")
            return None

    def analyze_accident(self, accident_report, regulation_content):
        """事故の詳細分析と改善提案の生成"""
        analysis_prompt = f"""
        以下の事故報告書と関連省令を分析し、詳細な分析と改善提案を行ってください。

        事故報告書:
        {accident_report}

        関連省令:
        {regulation_content}

        以下の形式で分析結果を出力してください：

        ===事故分析===
        1. 直接的な原因:
        - [主要な原因の詳細な分析]
        - [関連する要因の特定]

        2. 背景要因:
        - [組織的な要因]
        - [環境的な要因]
        - [人的要因]

        3. 省令との関連:
        - [関連する省令条項の具体的な指摘]
        - [規定と実態のギャップ分析]

        ===改善提案===
        1. 即時対応が必要な項目:
        - [具体的な改善アクション]
        - [期待される効果]
        - [実施における注意点]

        2. 中長期的な改善案:
        - [システム面での改善]
        - [運用面での改善]
        - [教育・訓練面での改善]

        3. 省令改善の提案:
        - [現行規定の問題点]
        - [改善が必要な条項]
        - [具体的な改正案（省令の該当部分を引用し、省令の文章を考えてください）]

        ===予防措置===
        1. 再発防止策:
        - [具体的な防止策]
        - [実施手順]
        - [効果測定方法]

        2. 水平展開:
        - [類似事故の予防策]
        - [他路線・他社への展開可能性]
        """

        try:
            result = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            print("\n=== 事故分析と改善提案 ===")
            print(result.content)
            print("=" * 50)
            return result.content
        except Exception as e:
            print(f"Error in analyze_accident: {str(e)}")
            return None


def analyze_random_accident():
    """ランダム事故報告の解析を開始"""

    with open("data/accident-report.md", "r", encoding="utf-8") as f:
        content = f.read()

    total_reports = extract_total_reports(content)
    if total_reports == 0:
        print("❌ 報告書の総件数を取得できませんでした。")
        return

    report = get_random_report(content, total_reports)
    display_report_summary(report)

    analyzer = AccidentAnalyzer()

    relevant_reg_result = analyzer.identify_relevant_regulations(report)
    if not relevant_reg_result:
        print("❌ 関連省令の特定に失敗しました")
        return

    reg_file_match = re.search(r"選択されたファイル:\s*(\d+\.md)", relevant_reg_result)
    if not reg_file_match:
        print("❌ 省令ファイル名の抽出に失敗しました")
        return

    regulation_file = reg_file_match.group(1)
    print(f"\n📁 使用する省令ファイル: {regulation_file}")

    try:
        with open(os.path.join("data", regulation_file), "r", encoding="utf-8") as f:
            regulation_content = f.read()
            print(f"✅ 省令ファイル読み込み完了 ({len(regulation_content)} 文字)")
    except Exception as e:
        print(f"❌ ファイル読み込み時にエラーが発生: {str(e)}")
        return

    print("\n🔍 事故分析を開始...")
    analysis_result = analyzer.analyze_accident(report, regulation_content)
    if not analysis_result:
        print("❌ 事故分析に失敗しました")
        return
