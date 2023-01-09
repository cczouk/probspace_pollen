
import numpy as np
import pandas as pd

def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < 0.5, np.floor(arr), result_array)
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where((result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)))&(result_array >= 0.5), np.ceil(arr), result_array)
    return result_array

# sample_submissionからindex, column名を取得
sub = pd.read_csv("./data/sample_submission.csv", index_col='datetime')
target_cols = sub.columns.values.tolist()
sub[target_cols] = 0
sub_s = sub.copy()

# 通常予測のweightリスト target, model別
weights_list = [
    [0.4, 0.1, 0.3, 0.2], #0
    [0.9, 0.0, 0.1, 0.0], #1
    [0.2, 0.3, 0.4, 0.1], #2 
    ]
# mse予測のweightリスト
weights_list_s = [
    [0.3, 0.0, 0.6, 0.1], #0
    [0.9, 0.0, 0.0, 0.1], #1
    [1.0, 0.0, 0.0, 0.0], #2
    ]

#　mseによる予測に切り替える閾値
threshold = [40, 20, 99999]
# mse予測値の大小関係をより強調/緩和するために累乗する
exs = [0.5, 1.0, 1.0]

dfs = [
        "./sub/lgb_10cv_seed_6_4_en.csv",
        "./sub/cat_10cv_seed_6_4_en.csv",
        "./sub/xgb_10cv_seed_6_4_en.csv",
        "./sub/dnn_10cv_seed_6_4_en.csv",
]

# catboostの読み込むファイル名を間違えているのに気づかず、catboostは逆効果だと思い込みweight0にしている
dfs_s = [
        "./sub/lgb_mse_10cv_seed_6.csv",
        "./sub/cat_10cv_seed_6_4_en.csv",
        "./sub/xgb_10cv_seed_6_4_en_mse.csv",
        "./sub/dnn_10cv_seed_6_4_en_mse.csv",
]

# weight ensemble mae
for i, target in enumerate(target_cols):
    weights=weights_list[i]
    for df,weight in zip(dfs,weights):
        df = pd.read_csv(df, index_col='datetime')
        sub[target] = sub[target]+df[target]*weight
    #　おそらくアンサンブル後に整数値に丸めた後、4倍したほうがいいはず（推測で決め打ち）
    sub[target] = sub[target].apply(lambda x: geometric_round(x))
    sub[target] = sub[target]*4


# weight ensemble mse
for i, target in enumerate(target_cols):
    weights=weights_list_s[i]
    for df,weight in zip(dfs_s,weights):
        df = pd.read_csv(df, index_col='datetime')
        sub_s[target] = sub_s[target]+df[target]*weight
    #　mseによる予測はより大きい値を予測値として得たいため切り上げを選択
    sub_s[target] = np.floor(sub_s[target])
    sub_s[target] = sub_s[target]*4    


df_mae = sub.copy()
df_mse = sub_s.copy()
# 通常予測で閾値以上の場合はmse予測の累乗に切り替え
for i, target in enumerate(target_cols):
    df_mae[target] = df_mae[target].mask(df_mae[target] >= threshold[i], df_mse[target]**exs[i])

# 提出用ファイル出力
df_mae.to_csv('./sub/lgb_hybrid_05.csv')


