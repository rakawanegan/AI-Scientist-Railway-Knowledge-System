# src/railway_service.py
import os
import random
from src.accident_analysis_tool import (
    extract_total_reports,
    get_random_report,
    display_report_summary,
    AccidentAnalyzer,
    analyze_random_accident,
)
from src.agent_solver import (
    setup_environment,
    inference,
    evaluate_responses,
    batch_inference,
    visualize_tool_usage,
)
from src.eval_system import (
    initialize_llm,
    load_yaml_config,
    setup_agents,
    evaluate_by_llm,
    process_dataframe,
)
from src.visualize_txt import (
    get_embedding,
    preprocess_and_save_embeddings,
    plot_umap,
    read_markdown_as_plain_text,
    tokenize_japanese,
    extract_keywords_with_tfidf,
    extract_keywords_with_bm25,
    generate_wordcloud,
    analyze_technical_terms,
)


def load_config():
    """
    Load configuration values from environment variables or configuration files.
    """
    config = {
        "accident_reports_path": "templates/railway_command/docs/accident-report.md",
        "summary_path": "templates/railway_command/docs/summary.txt",
        "tool_configs": [
            "templates/railway_command/docs/tools/code.yaml",
            "templates/railway_command/docs/tools/incident_rag.yaml",
            "templates/railway_command/docs/tools/knowledge_rag.yaml",
            "templates/railway_command/docs/tools/raw_text.yaml",
            "templates/railway_command/docs/tools/search.yaml",
        ],
        "eval_configs": [
            "templates/railway_command/docs/evals/accuracy.yaml",
            "templates/railway_command/docs/evals/calculation.yaml",
            "templates/railway_command/docs/evals/evidence.yaml",
            "templates/railway_command/docs/evals/expertise.yaml",
            "templates/railway_command/docs/evals/expression.yaml",
            "templates/railway_command/docs/evals/relevance.yaml",
        ],
        "react_prompt": "templates/railway_command/docs/react_base_prompt.md",
        "railway_text_path": "data/document/鉄道に関する技術上の基準を定める省令の解釈基準.md",
        "another_domain_texts": [
            "data/document/another_document/建築基準法道路関係規定運用指針の解説.md",
            "data/document/another_document/道路構造令の各規定の解説.md",
            "data/document/another_document/電気設備の技術基準の解釈.md",
        ],
    }
    return config


def main_service():
    """
    Main function to manage the railway business support service workflow.
    """
    config = load_config()

    # Step 1: Accident Analysis
    print("Starting Accident Analysis...")

    # Read accident report data
    with open(config["accident_reports_path"], "r") as file:
        content = file.read()

    total_reports = extract_total_reports(content)
    if total_reports == 0:
        print("No accident reports found.")
        return

    random_report = get_random_report(content, total_reports)
    display_report_summary(random_report)

    analyzer = AccidentAnalyzer()
    relevant_regulation = analyzer.identify_relevant_regulations(random_report)
    print(f"Relevant Regulation: {relevant_regulation}")

    regulation_content = ""  # Placeholder: Load relevant regulation content
    analysis_result = analyzer.analyze_accident(random_report, regulation_content)
    print(f"Analysis Result:/n{analysis_result}")

    # Step 2: NLP Agent Setup and Response Evaluation
    print("Setting up NLP Agent...")
    agent, criteria_with_weights = setup_environment(
        3, config["tool_configs"], config["eval_configs"], config["react_prompt"]
    )

    sample_question = "What are the recent railway safety measures?"
    response = inference(agent, sample_question)
    print(f"Agent Response: {response}")

    correct_answer = "Recent railway safety measures include..."
    evaluation = evaluate_responses(
        agent, criteria_with_weights, sample_question, correct_answer
    )
    print(f"Evaluation Results: {evaluation}")

    # Step 3: Text Visualization and Keyword Analysis
    print("Performing Text Visualization...")
    analyze_technical_terms(
        config["railway_text_path"],
        config["another_domain_texts"],
        [extract_keywords_with_tfidf, extract_keywords_with_bm25],
    )


if __name__ == "__main__":
    main_service()
