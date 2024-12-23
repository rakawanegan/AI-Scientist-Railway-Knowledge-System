# 鉄道業支援人工知能科学者 

このプロジェクトは、Sakana.ai の [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) をフォークして作成されました。本プロジェクトでは、数値実験を伴わない研究分野においても適用可能な形に改良を加え、鉄道業における科学的発見や業務効率化を支援するAIツールを目指しています。

---

## 概要

人工知能の主要な課題の1つは、科学研究を遂行し、新たな知識を発見できるエージェントを開発することです。本プロジェクトでは、既存の「AI-Scientist」の成果を基に、鉄道業界特有の課題に対応するためのカスタマイズを行いました。

例えば、鉄道規制文書の知識検索システムや運行指令業務支援におけるAIエージェントの応用可能性を検討しています。本システムは、鉄道技術基準や関連規制文書を基にしたインタラクティブな情報検索と、業務支援のための高度なAIアプローチを統合しています。

---

## 主な機能

- **鉄道規制文書の知識検索**: 技術基準や法規文書から関連情報を検索し、引用箇所を明確に提示します。
- **AIエージェントの応用**: 規制情報を基にした意思決定支援や運行管理業務の効率化。
- **カスタマイズ可能なテンプレート**: 鉄道業務に特化したテンプレートを活用し、プロジェクト固有の課題に対応します。

---

## 導入方法

このコードはLinux環境での利用を想定しており、NVIDIA GPUを用いたCUDAとPyTorchが必要です。以下の手順でセットアップを行ってください。

### 必要要件

1. Python 3.11
2. TexLive（PDF生成のため）
3. PyTorch（CUDA対応）

### インストール

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# 必要なパッケージのインストール
sudo apt-get install texlive-full
pip install -r requirements.txt
```

### モデルおよびAPIキー

OpenAIやAnthropicなどの各種モデルをサポートしています。APIキーを環境変数として設定してください。

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

---

## 鉄道業向けテンプレートの設定

以下は、鉄道業務に特化したテンプレートの設定手順です。

### テンプレートの概要

- **運行指令業務テンプレート**: 運行状況を基に適切な指令を生成。
- **規制文書検索テンプレート**: 特定の技術基準や規制文書に関連する情報を抽出。

### テンプレートの設定

1. 必要なデータを準備する。

```bash
python data/railway/prepare.py
```

2. テンプレートを実行し、結果を確認する。

```bash
cd templates/railway_command
python experiment.py --out_dir run_0
python plot.py
```

---

## 注意事項

このコードベースでは、AIモデルによって生成されたコードを実行するため、一定のリスクが伴います。Dockerによるコンテナ化を推奨します。

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE>
```

## 免責事項

このリポジトリは [Sakana.ai](https://github.com/SakanaAI/AI-Scientist) のフォークとして開発されました。本プロジェクトは、元のコードベースを改良し、鉄道業界での適用を目的としています。

