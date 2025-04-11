import re
import csv

# 入力・出力ファイル名（必要に応じて変更可能）
input_file = 'input.md'
output_file = 'output.csv'

model_dict = {}

# 入力ファイルの読み込み
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if '|' not in line:
            continue  # スキップ（ヘッダや無関係な行）
        columns = [col.strip() for col in line.strip().strip('|').split('|')]
        if len(columns) < 2:
            continue  # データが不完全な行はスキップ

        model_name = columns[0]
        tps_match = re.search(r'([\d.]+)\s±', columns[-1])
        if tps_match:
            tps = float(tps_match.group(1))
            if model_name not in model_dict:
                model_dict[model_name] = []
            model_dict[model_name].append(tps)

# CSV形式で出力
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for model, tps_list in model_dict.items():
        row = [model] + [f'{tps:.2f}' for tps in tps_list]
        writer.writerow(row)

print(f'Done! Output saved to {output_file}')

