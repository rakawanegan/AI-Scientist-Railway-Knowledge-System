import os
import pandas as pd
import requests

# CSVファイルのパス
csv_path = "./index.csv"

# PDFを保存するディレクトリ
output_dir = "paper"
os.makedirs(output_dir, exist_ok=True)

# CSVファイルを読み込む
df = pd.read_csv(csv_path)

# PDFをダウンロードして保存
def download_pdf(url, output_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {output_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# PathカラムのURLをダウンロード
for index, row in df.iterrows():
    url = row['Path']
    file_name = f"paper_{index + 1}.pdf"
    output_path = os.path.join(output_dir, file_name)
    if url.endswith(".pdf"):
        download_pdf(url, output_path)
    else:
        print(f"Skipping non-PDF URL: {url}")
