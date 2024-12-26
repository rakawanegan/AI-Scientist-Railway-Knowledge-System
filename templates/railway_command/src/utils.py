import yaml
from langchain.agents import Tool


def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def load_yaml_config(file_path: str) -> dict[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"警告: ファイルが見つかりません: {file_path}")
        return dict()
    except yaml.YAMLError as e:
        print(f"警告: YAMLの読み込みエラー: {e}")
        return dict()


def load_tool(p_config: dict[str], func_dict) -> Tool:
    tool_config = load_yaml_config(p_config)
    return Tool(
        name=tool_config["name"],
        func=func_dict[tool_config["name"]],
        description=tool_config["description"],
    )


def load_criteria_with_weights(p_configs: list[str]) -> list[dict[str]]:
    return [load_yaml_config(p_config) for p_config in p_configs]
