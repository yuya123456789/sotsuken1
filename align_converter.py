import pandas as pd

# 物語タイトルを入力
title = input("物語を入力: ")
fn = title + '_transcript.csv'

# CSVファイルを読み込み（元データにヘッダーがある場合は header=0、ない場合は header=None）
df = pd.read_csv("align.csv", sep=",", header=None)  # 元データの1行目もデータとして読み込む
# print(df.shape)
print(df.head())

# 列数を確認
num_cols = df.shape[1]

# 2,3,4列目のインデックスを存在する範囲で取得
col_indices = [i for i in range(1, min(4, num_cols))]  # 1,2,3列目（0始まり）

# 2列目に '<unk>' がある場合、1列目の内容で置き換え
if len(col_indices) >= 1:
    df.iloc[:, col_indices[0]] = df.iloc[:, col_indices[0]].where(
        df.iloc[:, col_indices[0]] != "<unk>",
        df.iloc[:, 0]
    )

# 抽出する列を選択
df_selected = df.iloc[:, col_indices]

# TSV（またはCSV）として保存、最初にタイトル行を追加
df_selected.to_csv(
    fn,
    sep=",",
    index=False,
    header=['word', 'onset', 'offset'][:len(df_selected.columns)]
)

print("保存完了: " + fn)