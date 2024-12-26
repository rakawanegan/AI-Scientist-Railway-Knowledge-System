
---

## 仕様書: `accident_analysis_tool.py`

### 概要
`accident_analysis_tool.py` は、事故報告書を解析し、関連する省令を特定および詳細分析を行うためのツールです。ランダムに選択した報告書を解析し、省令との関連性を明らかにしたうえで、詳細な事故分析や改善提案を提供します。

---

### 使用方法
1. 必要なデータを以下の形式で準備します:
   - `accident-report.md`: 事故報告書一覧（Markdown形式）。
   - `summary.txt`: 省令の候補内容。
   - 個別の省令ファイル（例: `01.md`, `02.md` など）。

2. スクリプトを実行します:
   ```bash
   python accident_analysis_tool.py
   ```

3. 標準出力に選択された報告書とその分析結果が表示されます。

---

### 関数およびクラスの説明

#### extract_total_reports(content)
- 概要:  
  事故報告書の総件数を抽出します。

- 引数:  
  - `content` (str): 報告書一覧の内容。

- 戻り値:  
  - (int): 報告書の総件数（抽出失敗時は0）。

- 依存関係:  
  なし。

---

#### get_random_report(content, total_reports)
- 概要:  
  報告書一覧からランダムに1件の報告書を選択します。

- 引数:  
  - `content` (str): 報告書一覧の内容。
  - `total_reports` (int): 報告書の総件数。

- 戻り値:  
  - (str): 選択された報告書の内容。

- 依存関係:  
  - `random.choice`: ランダムな選択を行うために使用。
  - `extract_total_reports`: 報告書総件数の計算が必要。

---

#### display_report_summary(report)
- 概要:  
  選択された報告書の概要を解析し、わかりやすい形式で表示します。

- 引数:  
  - `report` (str): 報告書の内容。

- 戻り値:  
  - なし（標準出力に表示）。

- 依存関係:  
  - `re.search`: 報告書内の情報を抽出するために使用。
  - `get_random_report`: 表示対象の報告書を取得する前段階で必要。

---

#### class AccidentAnalyzer
- 概要:  
  ChatOpenAI を活用し、事故報告書の分析および関連省令の特定を行うクラス。

---

##### AccidentAnalyzer.__init__()
- 概要:  
  ChatOpenAI のインスタンスを初期化します。

- 引数:  
  - なし。

- 戻り値:  
  - なし。

- 依存関係:  
  - `ChatOpenAI`: GPT-4ベースの言語モデルを使用。

---

##### AccidentAnalyzer.identify_relevant_regulations(accident_report)
- 概要:  
  報告書の内容に基づいて関連性の高い省令ファイルを特定します。

- 引数:  
  - `accident_report` (str): 事故報告書の内容。

- 戻り値:  
  - (str): 選択された省令ファイル名と理由。

- 依存関係:  
  - `ChatOpenAI.invoke`: 言語モデルを用いて省令を分析。
  - 外部ファイル `summary.txt`: 省令候補リストを参照。

---

##### AccidentAnalyzer.analyze_accident(accident_report, regulation_content)
- 概要:  
  報告書と関連省令を解析し、原因分析や改善提案を生成します。

- 引数:  
  - `accident_report` (str): 事故報告書の内容。
  - `regulation_content` (str): 関連省令の内容。

- 戻り値:  
  - (str): 分析結果。

- 依存関係:  
  - `ChatOpenAI.invoke`: 言語モデルを用いて事故を詳細分析。

---

#### analyze_random_accident()
- 概要:  
  報告書データを解析し、ランダムに選択した事故の概要、関連省令、および分析結果を出力します。

- 引数:  
  - なし。

- 戻り値:  
  - なし（標準出力に表示）。

- 依存関係:  
  - `extract_total_reports`: 報告書の総件数を取得。
  - `get_random_report`: ランダムに報告書を選択。
  - `display_report_summary`: 報告書の概要を表示。
  - `AccidentAnalyzer`: 省令の特定および詳細分析を実施。
  - 外部ファイル `accident-report.md`, `summary.txt`, 各省令ファイル: 必要なデータを提供。

---

### 依存関係の全体図
1. `analyze_random_accident`  
   - 呼び出し元関数として全体の流れを管理。
   - 依存関係: `extract_total_reports` → `get_random_report` → `display_report_summary` → `AccidentAnalyzer`

2. `AccidentAnalyzer`  
   - `identify_relevant_regulations` と `analyze_accident` が他関数から利用され、ChatOpenAIに依存。

3. 外部ライブラリおよびファイル依存:  
   - `os`, `random`, `re`, `pandas`, `langchain_openai`, `ChatOpenAI`: 主にファイル操作、データ処理、AIモデル利用に必要。
   - 外部データ: `accident-report.md`, `summary.txt`, 各省令ファイル。

--- 

### データ構造
- `accident-report.md`: 事故報告書一覧（Markdown形式）。  
- `summary.txt`: 省令候補のリスト。  
- 個別の省令ファイル（例: `01.md`, `02.md`）。  

--- 

## 仕様書: `agent_solver.py`

### 概要
`agent_solver.py` は、自然言語処理エージェントを用いて、質問応答の生成およびその評価を行うためのツールです。設定されたツール群と評価基準に基づき、個別または一括での質問解析を実施し、その結果を定量的に評価します。また、エージェントの動作状況やツール使用状況を可視化する機能も備えています。

---


### setup_environment(k, tool_configs, eval_configs, react_prompt)
- 目的:  
  実行環境を設定し、エージェントおよび評価基準を初期化します。

- 引数:  
  - `k` (int): ツール呼び出しの深さ。
  - `tool_configs` (list of str): ツール設定ファイルのパスリスト。
  - `eval_configs` (list of str): 評価基準設定ファイルのパスリスト。
  - `react_prompt` (str): React Promptファイルのパス。

- 戻り値:  
  - `agent`: 初期化されたエージェント。
  - `criteria_with_weights`: 評価基準とその重み。

- 依存関係:  
  - `os.environ`: 環境変数の設定。
  - `ChatOpenAI`: 言語モデルの初期化。
  - `make_agent`: エージェントを作成。
  - `load_criteria_with_weights`: 評価基準を読み込み。

---

### timer(name)
- 目的:  
  処理時間を計測するコンテキストマネージャを提供します。

- 引数:  
  - `name` (str): 計測対象の名前。

- 戻り値:  
  - なし（コンテキスト内で処理時間を計測し、終了時に標準出力へ表示）。

- 依存関係:  
  - `time.time`: 現在の時刻を取得。

---

### inference(agent, question)
- 目的:  
  エージェントを使用して質問に応答を生成します。

- 引数:  
  - `agent`: 初期化されたエージェント。
  - `question` (str): 質問文。

- 戻り値:  
  - `output` (str): 応答またはエラー内容。

- 依存関係:  
  - `agent.run`: エージェントの応答生成。

---

### evaluate_responses(agent, criteria_with_weights, query, answer)
- 目的:  
  エージェントの応答を評価基準に基づいて評価します。

- 引数:  
  - `agent`: 初期化されたエージェント。
  - `criteria_with_weights`: 評価基準と重み。
  - `query` (str): 質問文。
  - `answer` (str): 正解文。

- 戻り値:  
  - `predict` (str): エージェントの応答。
  - `dict_eval` (dict): 評価結果の辞書。

- 依存関係:  
  - `timer`: 処理時間の計測。
  - `inference`: 応答生成。
  - `evaluate_by_llm_with_criteria`: 応答の評価。

---

### batch_inference(agent, criteria_with_weights, input_file, output_file, all_run=True)
- 目的:  
  一括で質問応答と評価を行い、結果をCSVファイルに出力します。

- 引数:  
  - `agent`: 初期化されたエージェント。
  - `criteria_with_weights`: 評価基準と重み。
  - `input_file` (str): 入力データCSVファイルのパス。
  - `output_file` (str): 出力データCSVファイルのパス。
  - `all_run` (bool, デフォルト=True): 全行を処理するかどうか。

- 戻り値:  
  - `df_answer` (pd.DataFrame): 出力データフレーム。

- 依存関係:  
  - `pd.read_csv`: 入力データの読み込み。
  - `inference`: 応答生成。
  - `evaluate_by_llm_with_criteria`: 応答の評価。
  - `tqdm`: プログレスバーの表示。
  - `pd.DataFrame.to_csv`: 出力データの保存。

---

### visualize_tool_usage()
- 目的:  
  ツールの使用状況を可視化します。

- 引数:  
  - なし。

- 戻り値:  
  - なし（ツール使用状況の可視化結果を表示）。

- 依存関係:  
  - `tool_usage_tracker.get_counts`: ツール使用回数の取得。
  - `plot_tools_count`: 使用状況をプロット。

---

### main()
- 目的:  
  全体の実行フローを管理します。

- 処理内容:
  1. 環境設定。
  2. 個別質問の応答と評価。
  3. 一括処理の実行。
  4. ツール使用状況の可視化。

- 引数:  
  - なし。

- 戻り値:  
  - なし（標準出力へ結果を表示し、一部ファイル出力を実施）。

- 依存関係:  
  - `setup_environment`: 実行環境の初期化。
  - `evaluate_responses`: 個別質問の応答評価。
  - `batch_inference`: 一括処理。
  - `visualize_tool_usage`: ツール使用状況の可視化。

---

## 仕様書: `eval_system.py`

### 概要
`eval_system.py` は、自然言語処理（LLM）を活用した応答評価システムを構築するツールです。LLMベースの評価基準を使用し、複数の基準に基づいて応答を採点し、スコアリングを行います。評価は単一のモデル、マルチエージェント、または基準に基づく直接採点の3種類の方式で実行可能です。本システムはデータフレーム処理やYAML設定の読み込みもサポートします。

---

### 関数の仕様

#### initialize_llm()
- 目的:  
  LLM（言語モデル）を初期化します。

- 引数:  
  - なし。

- 戻り値:  
  - `llm` (ChatOpenAI): 初期化されたLLMオブジェクト。

- 依存関係:  
  - `ChatOpenAI`: 言語モデルの生成。

---

#### load_yaml_config(file_path: str) -> dict[str, str]
- 目的:  
  指定したYAMLファイルを読み込み、辞書形式で返します。

- 引数:  
  - `file_path` (str): YAMLファイルのパス。

- 戻り値:  
  - `data` (dict[str, str]): YAMLファイルの内容を格納した辞書。

- 依存関係:  
  - `yaml.safe_load`: YAMLファイルの解析。
  - `os.open`: ファイル操作。

- エラー処理:  
  ファイルが見つからない場合やYAML解析エラー時に警告メッセージを表示。

---

#### setup_agents(p_eval_configs, llm_config)
- 目的:  
  マルチエージェント評価システムを構築します。

- 引数:  
  - `p_eval_configs` (list[str]): 評価基準設定ファイルのパスリスト。
  - `llm_config` (dict): LLMの設定。

- 戻り値:  
  - `group_chat_manager` (autogen.GroupChatManager): 初期化されたエージェント管理システム。

- 依存関係:  
  - `load_yaml_config`: YAML設定の読み込み。
  - `autogen.AssistantAgent`: 個々のエージェントを作成。
  - `autogen.GroupChat`: マルチエージェントのグループ管理。

---

#### inference(manager, question)
- 目的:  
  マルチエージェントシステムを使用して質問に対する応答を生成します。

- 引数:  
  - `manager` (autogen.GroupChatManager): エージェント管理システム。
  - `question` (str): 質問文。

- 戻り値:  
  - `output` (str): 応答またはエラーメッセージ。

- 依存関係:  
  - `manager.initiate_chat`: エージェントチャットを開始。
  - `manager.groupchat.messages`: 応答メッセージの取得。

---

#### evaluate_by_multiagent_llm(manager, pred, answer, question=None)
- 目的:  
  マルチエージェントを使用して応答を評価します。

- 引数:  
  - `manager` (autogen.GroupChatManager): エージェント管理システム。
  - `pred` (str): 応答文。
  - `answer` (str): 正解文。
  - `question` (str, オプション): 質問文。

- 戻り値:  
  - (str): 評価結果。

- 依存関係:  
  - `PromptTemplate`: プロンプトを生成。
  - `inference`: 評価プロセスを実行。

---

#### evaluate_by_llm(llm, criterions, pred, answer, question)
- 目的:  
  単一のLLMを使用して応答を評価します。

- 引数:  
  - `llm` (ChatOpenAI): 言語モデル。
  - `criterions` (list[dict]): 評価基準。
  - `pred` (str): 応答文。
  - `answer` (str): 正解文。
  - `question` (str): 質問文。

- 戻り値:  
  - (str): 評価結果。

- 依存関係:  
  - `PromptTemplate`: プロンプトを生成。
  - `llm.predict`: プロンプトに基づく評価結果を生成。

---

#### process_dataframe(df, evaluate_function, output_file, kwargs)
- 目的:  
  データフレーム内のすべての行を評価し、結果を新しいデータフレームとして保存します。

- 引数:  
  - `df` (pd.DataFrame): 入力データフレーム。
  - `evaluate_function` (function): 評価関数。
  - `output_file` (str): 出力CSVファイルのパス。
  - `kwargs`: 評価関数に渡す追加の引数。

- 戻り値:  
  - `df_answer` (pd.DataFrame): 評価結果を含む新しいデータフレーム。

- 依存関係:  
  - `tqdm`: プログレスバーの表示。
  - `re.search`: スコアの抽出。
  - `pd.DataFrame.to_csv`: 結果の保存。

---

### 依存関係の全体図
1. 主要関数:
   - `main`: 全体の制御。
   - `initialize_llm`: LLMの初期化。
   - `setup_agents`: マルチエージェントの構築。
   - `process_dataframe`: データフレームの評価。

2. 外部ライブラリ:
   - `os`, `re`, `yaml`, `pandas`, `tqdm`: ファイル操作、データ処理、進捗表示など。
   - `langchain.prompts.PromptTemplate`: プロンプト生成。
   - `langchain_openai.ChatOpenAI`: 言語モデル。

3. カスタムモジュール:
   - `autogen`: マルチエージェント管理。
   - `src.main`: 評価関数。
   - `src.utils`: 設定ファイルの読み込み。

--- 

## 仕様書: `visualize_txt.py`

### 概要
`visualize_txt.py` は、テキストデータを可視化および分析するためのツールです。日本語トークン化、TF-IDFやBM25を用いたキーワード抽出、埋め込みデータの生成と保存、UMAPを用いた次元削減可視化、WordCloudによるキーワードの視覚的表現など、幅広い分析機能を提供します。本スクリプトは特に鉄道技術用語の分析や、他分野との比較を想定した設計になっています。

---

### 関数の仕様

#### get_embedding(text, model="text-embedding-3-small")
- 目的:  
  指定されたテキストの埋め込みベクトルを生成します。

- 引数:  
  - `text` (str): 埋め込み対象のテキスト。
  - `model` (str, デフォルト="text-embedding-3-small"): 埋め込み生成に使用するモデル。

- 戻り値:  
  - `embedding` (list[float]): 埋め込みベクトル（長さ1536）。

- 依存関係:  
  - `OpenAI.embeddings.create`: 埋め込み生成。

---

#### preprocess_and_save_embeddings(data, filename="railway-knowledge-text-embedding-3-small.npy")
- 目的:  
  埋め込みデータを保存し、次元削減を実施します。

- 引数:  
  - `data` (np.ndarray): 埋め込みデータ。
  - `filename` (str, デフォルト="railway-knowledge-text-embedding-3-small.npy"): 保存先ファイル名。

- 戻り値:  
  - `embedding` (np.ndarray): 次元削減後の埋め込みデータ。

- 依存関係:  
  - `np.save`: データの保存。
  - `StandardScaler`: データの標準化。
  - `umap.UMAP`: 次元削減の実施。

---

#### plot_umap(embedding)
- 目的:  
  次元削減結果をUMAPプロットとして表示します。

- 引数:  
  - `embedding` (np.ndarray): 次元削減後の埋め込みデータ。

- 戻り値:  
  - なし（プロットを表示）。

- 依存関係:  
  - `matplotlib.pyplot.scatter`: プロット描画。

---

#### read_markdown_as_plain_text(filepath)
- 目的:  
  Markdown形式のファイルをプレーンテキストとして読み取ります。

- 引数:  
  - `filepath` (str): 読み込むファイルのパス。

- 戻り値:  
  - `content` (str): プレーンテキスト形式の内容。

- 依存関係:  
  - `open`: ファイルの読み込み。
  - `re.sub`: Markdown特有の形式を除去。

---

#### tokenize_japanese(text)
- 目的:  
  日本語テキストを名詞のみでトークン化します。

- 引数:  
  - `text` (str): トークン化対象の日本語テキスト。

- 戻り値:  
  - `tokens` (str): トークン化された名詞の集合。

- 依存関係:  
  - `janome.tokenizer.Tokenizer`: 日本語のトークン化。

---

#### extract_keywords_with_tfidf(text, top_n=5)
- 目的:  
  TF-IDFスコアに基づいてキーワードを抽出します。

- 引数:  
  - `text` (str): テキストデータ。
  - `top_n` (int, デフォルト=5): 上位N個のキーワードを抽出。

- 戻り値:  
  - `tfidf_scores` (pd.DataFrame): キーワードとスコアのデータフレーム。

- 依存関係:  
  - `tokenize_japanese`: テキストのトークン化。
  - `TfidfVectorizer`: TF-IDFスコアの計算。

---

#### extract_keywords_with_bm25(text, top_n=5)
- 目的:  
  BM25スコアに基づいてキーワードを抽出します。

- 引数:  
  - `text` (str): テキストデータ。
  - `top_n` (int, デフォルト=5): 上位N個のキーワードを抽出。

- 戻り値:  
  - `bm25_scores` (pd.DataFrame): キーワードとスコアのデータフレーム。

- 依存関係:  
  - `tokenize_japanese`: テキストのトークン化。
  - `BM25Okapi`: BM25スコアの計算。

---

#### generate_wordcloud(text, extract_keywords, top_n=50)
- 目的:  
  抽出したキーワードをWordCloudで可視化します。

- 引数:  
  - `text` (str): テキストデータ。
  - `extract_keywords` (function): キーワード抽出関数。
  - `top_n` (int, デフォルト=50): 可視化するキーワード数。

- 戻り値:  
  - `top_keywords` (pd.DataFrame): 抽出されたキーワードとスコア。

- 依存関係:  
  - `WordCloud.generate_from_frequencies`: WordCloudの生成。
  - `matplotlib.pyplot.imshow`: プロット描画。

---

#### analyze_technical_terms(p_railway_txt, p_another_domain_txts, extract_keywords_list)
- 目的:  
  鉄道技術用語と他分野用語を比較し、技術的な差異を分析します。

- 引数:  
  - `p_railway_txt` (str): 鉄道技術分野のテキストファイルパス。
  - `p_another_domain_txts` (list[str]): 他分野テキストファイルのパスリスト。
  - `extract_keywords_list` (list[function]): 使用するキーワード抽出関数のリスト。

- 戻り値:  
  - なし（分析結果とWordCloudを表示）。

- 依存関係:  
  - `read_markdown_as_plain_text`: テキストの読み取り。
  - `generate_wordcloud`: 用語の可視化。

---

### 依存関係の全体図
1. 主要関数:
   - テキスト処理: `read_markdown_as_plain_text`, `tokenize_japanese`
   - キーワード抽出: `extract_keywords_with_tfidf`, `extract_keywords_with_bm25`
   - 可視化: `plot_umap`, `generate_wordcloud`, `analyze_technical_terms`

2. 外部ライブラリ:
   - `matplotlib`, `numpy`, `pandas`: データ操作と可視化。
   - `umap`: 次元削減。
   - `janome`: 日本語トークン化。
   - `rank_bm25`: BM25スコア計算。
   - `WordCloud`: WordCloud生成。

--- 
