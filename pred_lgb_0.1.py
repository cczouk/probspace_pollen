import copy
import os
import random
import pandas as pd
import numpy as np
import warnings
import shap
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
import lightgbm as lgb
from lightgbm import LGBMRegressor
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

import datetime as dt

from sklearn.impute import IterativeImputer

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < 0.5, np.floor(arr), result_array)
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where((result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)))&(result_array >= 0.5), np.ceil(arr), result_array)
    return result_array

class StratifiedKFoldReg(StratifiedKFold):
    def split(self, X, y, groups=None):
        n_samples = len(y)
        n_labels = int(np.round(n_samples/self.n_splits))
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits) \
            for ii in range(n_labels)])
        mod = np.mod(n_samples, self.n_splits)
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])
        return super().split(X, y_labels, groups)


seed = 6 #26 
seed_everything(seed)

Jun_enc_fillna = 0
submit_cv = 0

#データの読み込み
train = pd.read_csv("./data/train_v2.csv")
test = pd.read_csv("./data/test_v2.csv")
print(train.shape)
print(test.shape)
# 特徴量選択の結果読み込み　[[未使用]]
null_imp_cols = pd.read_pickle('data/selectcol_obj_0.1_cols.pkl')
select_col_index = [20,23,22,]

# 前処理
train = train.replace('欠測', np.nan)
lgb_imp = IterativeImputer(
                        estimator=LGBMRegressor(num_iterations=1000, random_state=seed),
                        max_iter=10, 
                        initial_strategy='mean',
                        imputation_order='ascending',
                        verbose=0,
                        random_state=seed)

train = pd.DataFrame(lgb_imp.fit_transform(train), columns=train.columns)

# 訓練データのカラムを格納
columns = train.columns.values.tolist()
l_train = len(train)


# #　訓練データに使うカラムと、予測するカラムを設定
target_columns = columns[13:]
drop_cols = ['datetime', 'time', 'ymdh'] #'year', 
cday = 21


df_all = pd.concat([train, test])

# #週番号 曜日 日にち
df_all['time'] = pd.to_datetime(df_all.datetime.astype(int).astype(str).str[:-2])
df_all['year'] = df_all['time'].dt.year
df_all['month'] = df_all['time'].dt.month.astype('int8')
df_all['day'] = df_all['time'].dt.day.astype('int8')
df_all['hour'] = df_all.datetime.astype(int).astype(str).str[-2:].astype('int8')
df_all['weekday'] = df_all['time'].dt.weekday.astype('int8')
df_all["week_num"] = df_all['time'].dt.isocalendar().week.astype('int8')
df_all['day_of_year'] = df_all['time'].dt.dayofyear.astype('int8')
df_all['day_of_year'] = df_all.apply(lambda x: x['day_of_year']-1 if (x['time'] > pd.Timestamp('2020-02-29')) else x['day_of_year'], axis=1)
df_all['day_sin'] = np.sin(df_all['day_of_year'] * (2 * np.pi / 365))
df_all['day_cos'] = np.cos(df_all['day_of_year'] * (2 * np.pi / 365))
df_all['ymdh'] = pd.to_datetime(df_all.datetime.astype(int).astype(str).str[:-2]+'T'+(df_all['hour']-1).astype(str).str.zfill(2))
df_all['ymdh'] = df_all['ymdh'] + dt.timedelta(hours=1)


#風　ベクトル化
wd_col = ['winddirection_utsunomiya', 'winddirection_chiba', 'winddirection_tokyo']
ws_col = ['windspeed_utsunomiya', 'windspeed_chiba', 'windspeed_tokyo']
l_name = ['_u', '_c', '_t']
for d, s, l in zip(wd_col, ws_col, l_name):
    df_all[f'wd{l}_x'] = np.cos((90-df_all[d]*22.5)/180*np.pi)
    df_all[f'wd{l}_x'] = df_all[f'wd{l}_x'].mask((df_all[f'wd{l}_x'] < 1e-4) & (df_all[f'wd{l}_x'] > -1e-4), 0)
    df_all[f'wd{l}_x'] = df_all[f'wd{l}_x'].mask((df_all[d] == 0), 0)
    df_all[f'wv{l}_x'] = df_all[f'wd{l}_x']*df_all[s]
    df_all[f'wd{l}_y'] = np.sin((90-df_all[d]*22.5)/180*np.pi)
    df_all[f'wd{l}_y'] = df_all[f'wd{l}_y'].mask((df_all[f'wd{l}_y'] < 1e-4) & (df_all[f'wd{l}_y'] > -1e-4), 0)
    df_all[f'wd{l}_y'] = df_all[f'wd{l}_y'].mask((df_all[d] == 0), 0)
    df_all[f'wv{l}_y'] = df_all[f'wd{l}_y']*df_all[s]
    drop_cols.append(f'wd{l}_x')
    drop_cols.append(f'wv{l}_x')
    drop_cols.append(f'wd{l}_y')
    drop_cols.append(f'wv{l}_y')
    # df_all[d] = df_all[d].astype('category')

# ラグ特徴
def add_lag_feat(df, feat:list, group:str):
    outputs = [df]
    grp_df = df.groupby(group)
    for lag in [1, 2, 3, 4, 5]:
      # shift
      outputs.append(grp_df[feat].shift(lag).add_prefix(f'shift{lag}_'))
      # diff
      outputs.append(grp_df[feat].diff(lag).add_prefix(f'diff{lag}_'))
    # rolling
    for window in [3,24]:
        tmp_df = grp_df[feat].rolling(window, min_periods=1)
        tmp_df = tmp_df.mean().add_prefix(f'rolling{window}_mean_')
        outputs.append(tmp_df.reset_index(drop=True))

    return pd.concat(outputs, axis=1)

# ラグ特徴の追加
lag_feat = [
'precipitation_utsunomiya', 'precipitation_chiba', 'precipitation_tokyo', 
'temperature_utsunomiya', 'temperature_chiba', 'temperature_tokyo', 
'windspeed_utsunomiya', 'windspeed_chiba', 'windspeed_tokyo',
'winddirection_utsunomiya', 'winddirection_chiba', 'winddirection_tokyo',
# 'wv_u_x', 'wv_c_x', 'wv_t_x', 'wv_u_y', 'wv_c_y', 'wv_t_y',
]
df_all = add_lag_feat(df_all.reset_index(drop=True), lag_feat, 'year')

def add_mlag_feat(df, feat:list, group:str):
    outputs = [df]
    grp_df = df.groupby(group)
    outputs.append(grp_df[feat].shift(-1).add_prefix('shift-1_'))
    return pd.concat(outputs, axis=1)
mlag_feat = [
'precipitation_utsunomiya', 'precipitation_chiba', 'precipitation_tokyo', 
# 'temperature_utsunomiya', 'temperature_chiba', 'temperature_tokyo', 
# 'windspeed_utsunomiya', 'windspeed_chiba', 'windspeed_tokyo',
# 'winddirection_utsunomiya', 'winddirection_chiba', 'winddirection_tokyo',
# 'wv_u_x', 'wv_c_x', 'wv_t_x', 'wv_u_y', 'wv_c_y', 'wv_t_y',
]
df_all = add_mlag_feat(df_all.reset_index(drop=True), mlag_feat, 'year')

# 積算特徴
def add_cumsum_feat(df, feat:list, group:str):
    outputs = [df]
    grp_df = df.groupby(group)
    outputs.append(grp_df[feat].cumsum().add_prefix('cumsum_'))
    return pd.concat(outputs, axis=1)

cs_feat = [
'precipitation_utsunomiya', 'precipitation_chiba', 'precipitation_tokyo', 
'temperature_utsunomiya', 'temperature_chiba', 'temperature_tokyo', 
]
df_all = add_cumsum_feat(df_all.reset_index(drop=True), cs_feat, 'year')

# 雨量0連続カウント
def zero_count(df, feat:list, group:str, alpha=0.005):
    def zero_count_i(df, alpha=0.005):
        df_count = []
        n_count = 0
        for i in range(len(df)):
            if df.iloc[i] < 0.5:
                n_count += 1
            else:
                n_count = 0
            df_count.append(n_count)
        df_count = np.tanh(np.array(df_count)*alpha)
        return pd.DataFrame(df_count, index=df.index)
    outputs = [df]
    grp_df = df.groupby(group) 
    for col in feat:
        outputs.append(grp_df[col].apply(zero_count_i).add_prefix(f'zero_count_{col}'))
    return pd.concat(outputs, axis=1)

rain_feat = [
'precipitation_utsunomiya', 'precipitation_chiba', 'precipitation_tokyo', 
]
df_all = zero_count(df_all.reset_index(drop=True), rain_feat, 'year')

# 気温agg
tmp_feat = [
'temperature_utsunomiya', 'temperature_chiba', 'temperature_tokyo', 
]
df_all['ymd'] = df_all['time'].dt.strftime('%Y-%m-%d').astype('category')
tmp_max = df_all.groupby('ymd')[tmp_feat].agg(np.max)
tmp_min = df_all.groupby('ymd')[tmp_feat].agg(np.min)
tmp_mean = df_all.groupby('ymd')[tmp_feat].agg(np.mean)
tmp_max_min = df_all.groupby('ymd')[tmp_feat].agg(lambda x: max(x) - min(x))
for i, col in enumerate(tmp_feat):
    df_all[f'{col}_day_max'] = df_all['ymd'].map(lambda x: tmp_max[col][x]).values
    df_all[f'{col}_day_min'] = df_all['ymd'].map(lambda x: tmp_min[col][x]).values
    df_all[f'{col}_day_mean'] = df_all['ymd'].map(lambda x: tmp_mean[col][x]).values
    df_all[f'{col}_day_max_min'] = df_all['ymd'].map(lambda x: tmp_max_min[col][x]).values
# df_all = df_all.drop(columns='ymd')


# 6月の気温から次の年の総花粉量の相対目安を算出
def Jun_tmp(df, feat:list, group:str):
    def Jun_tmp_i(df):
        df_late = df[(df['day']>20)&(df['month']==6)]
        df_sum = (df_late[feat] > 25 ).sum()
        df_sum = df_late[feat].sum()
        return df_sum
    grp_df = df.groupby(group) 
    target_dict = grp_df.apply(Jun_tmp_i)
    return target_dict

target_dict = Jun_tmp(df_all.reset_index(drop=True), tmp_feat, 'year')

s_nan = pd.DataFrame([[np.nan,np.nan,np.nan]], columns=target_dict.columns, index=[2016])
target_dict = pd.concat([s_nan,target_dict])
target_dict = target_dict.drop(index=2020)
target_dict.index = list(target_dict.index.to_numpy()+1)
if Jun_enc_fillna == 1:
    target_dict = pd.DataFrame(
        IterativeImputer().fit_transform(target_dict),
        index=target_dict.index,
        columns=target_dict.columns
        )
for i, col in enumerate(tmp_feat):
    df_all[f'{col}_Jun_tmp_Enc'] = df_all['year'].map(lambda x: target_dict[col][x]).values
    
# 積算気温による相対積算花粉飛散量予測
coeff=[(1.1,-1.0e-4),(1.1,-1.0e-4),(1.1,-1.0e-4)]
for target, ce, tmp in zip(target_columns, coeff, tmp_feat):
    df_all[f'{target}_expected_pollen'] = ce[0]/(np.exp(ce[1]*df_all[f'cumsum_{tmp}'])*0.1 +1) -1
    df_all[f'{target}_remaining_pollen'] = df_all[f'{tmp}_Jun_tmp_Enc']**2 - df_all[f'{target}_expected_pollen']**2*df_all[f'{tmp}_Jun_tmp_Enc']*10
    
    df_all[f'{target}_expected_pollen_ilogit'] = np.exp(df_all[f'cumsum_{tmp}'])/(np.exp(df_all[f'cumsum_{tmp}']) +1)

#花粉好条件
location = ['chiba'] #'utsunomiya', 'tokyo'
for l in location:
    df_all[f'{l}_good_condition'] = 0
    df_all[f'{l}_good_condition'] = df_all[f'{l}_good_condition'].mask(
        (df_all[f'temperature_{l}_day_max'] > 15)&
        # (df_all[f'windspeed_{l}'] > 5)&
        (df_all[f'zero_count_precipitation_{l}0'] > 0.12)&
        (df_all['hour'] > 10)&
        (df_all['hour'] < 18)&
        (df_all['month'] < 6)
        , 1)
    if l =='chiba':
        df_all[f'{l}_good_condition'] = df_all[f'{l}_good_condition'].mask(
            (df_all[f'windspeed_{l}'] <= 5)
            # (df_all['windspeed_utsunomiya'] < 4.5)&
            # (df_all['hour'] < 7)&
            # (df_all['hour'] > 17)
            #day_std_precipitation_tokyo
            , 0)

# Savitzky-Golay filter
# lag_feat, mlag_feat, cs_feat, rain_feat, tmp_feat, wd_col, ws_col
for col in ws_col:
    for w in [12]:
        for p in [2,3,4]:
            for d in [0,1,2]:
                df_all[f'{col}_SG_w{w}_p{p}_der{d}'] = savgol_filter(df_all[col], w, p, deriv=d)


ind_feat = []
ind_feat_null = [null_imp_cols[c][select_col_index[c]] for c in range(len(target_columns))]    


# # 不要な列削除
df_all = df_all.drop(columns=drop_cols) 


# train test 再分割分
train = df_all.iloc[0:l_train,:].reset_index(drop=True)
test = df_all.iloc[l_train:,:].drop(columns=target_columns).reset_index(drop=True)

# add_agg
def additional_encoding(train, test, cat_col:list, num_col:list): 
    trdf = train.copy()
    tedf = test.copy()  

    # Count Encoding
    for ccol in cat_col:
        encoder = trdf[(trdf['month']==4)&(trdf['day']<15)][ccol].value_counts()
        trdf[f'ce_{ccol}'] = trdf[ccol].map(encoder)
        tedf[f'ce_{ccol}'] = tedf[ccol].map(encoder)

    # Add Aggregate Features
    agg_cols = ['mean', 'std', 'min', 'max']
    for ccol in cat_col:
        for ncol in num_col:
            agg_df = trdf.groupby(ccol)[ncol].agg(agg_cols)
            agg_df['abs_mean'] = np.abs(agg_df['mean'])
            agg_df['min_max'] = agg_df['min']*agg_df['max']
            agg_df.columns = [f'{ccol}_{c}_{ncol}' for c in agg_df.columns]
            trdf = trdf.merge(agg_df, on=ccol, how='left')
            tedf = tedf.merge(agg_df, on=ccol, how='left')

    return trdf, tedf

cat_columns = ['year', 'month', 'day', 'hour', 'winddirection_utsunomiya', 'winddirection_chiba', 'winddirection_tokyo']
num_columns = ['precipitation_utsunomiya', 'precipitation_chiba', 'precipitation_tokyo', 
           'temperature_utsunomiya', 'temperature_chiba', 'temperature_tokyo', 
           'windspeed_utsunomiya', 'windspeed_chiba', 'windspeed_tokyo']
train, test = additional_encoding(train, test, cat_columns, num_columns)


#異常値除外
train = train[(train['pollen_utsunomiya'] >= 0)&(train['pollen_chiba'] >= 0)&(train['pollen_tokyo'] >= 0)]


# 訓練データを、説明変数Xと目的変数yに分割
X = train.drop(columns=target_columns)
y = train[target_columns] / 4

# テストデータの説明変数test_Xを作成
test_X = test.copy()

# 予測結果を保存する辞書型データを作成
results = dict({})
ind_feat = [X.columns.tolist() for c in range(len(target_columns))]    


lgb_params = { 
    # 'learning_rate': 0.07,
    'objective': 'tweedie', # huber, fair, regression_l1, tweedie, poisson
    'tweedie_variance_power': 1.9,
    'num_leaves': 255,
    'max_bin': 127,
    'min_data_in_leaf': 80,
    'verbosity': -1,
    'random_state' : seed,
    } 

ind_params = []

# 特徴量選択の場合分け（未使用）
ind_feat_mode = 1
if ind_feat_mode == 1:
    if_cols = copy.copy(ind_feat)
else:
    if_cols = copy.copy(ind_feat_null)

# CV設定
n_fold = 10
sfkfold = StratifiedKFoldReg(n_splits=n_fold, random_state=seed, shuffle=True)    

lt_idx = []
lv_idx = []
for i, target in enumerate(target_columns):
    t_idx, v_idx = [[]], [[]]
    for tr_idx, va_idx in sfkfold.split(train.drop(columns=target_columns), train[target]):
        t_idx.append(tr_idx)
        v_idx.append(va_idx)
    del t_idx[0]
    del v_idx[0]
    lt_idx.append(t_idx)
    lv_idx.append(v_idx)
    
    
# train vaild index
valid_score_t = []
valid_score = []

best_itr_num_t = []
best_itr_num = []

best_itr_num_df = pd.DataFrame(index=range(len(target_columns)),columns=list(range(n_fold)))
best_itr_num_df = best_itr_num_df.add_prefix('cv')

cv_predict = []

df_train_pred = pd.DataFrame(index=train.index, columns=target_columns)

# best_iterationを得るためのCV
for n in tqdm(range(n_fold)):
    valid_score_t = []
    best_itr_num_t = []
    cv_predict_t = pd.DataFrame(columns=target_columns)
    for i, target in enumerate(target_columns):
        X_train, X_test = X.iloc[lt_idx[i][n]], X.iloc[lv_idx[i][n]]
        y_train, y_test = y.iloc[lt_idx[i][n]], y.iloc[lv_idx[i][n]]
        train_ds = lgb.Dataset(X_train[if_cols[i]], y_train[target])
        valid_ds = lgb.Dataset(X_test[if_cols[i]], y_test[target], reference=train_ds) 
        # lgb_params = ind_params[i].copy()
        lgb_model = lgb.train(lgb_params,train_ds,valid_sets=valid_ds,
                  num_boost_round=100_000,
                  callbacks=[lgb.early_stopping(400,verbose=False)],
                              )  
        y_pred_lgb = lgb_model.predict(X_test[if_cols[i]])
        y_pred_lgb = np.where(y_pred_lgb < 0, 0, y_pred_lgb)
        results[target] = y_pred_lgb
                
        valid_score_t.append(mean_absolute_error(y_test[target]*4, y_pred_lgb*4))
        best_itr_num_t.append(lgb_model.best_iteration)
        
        df_train_pred[target].iloc[lv_idx[i][n]] = y_pred_lgb*4
        
        cv_predict_t[target] = lgb_model.predict(test_X[if_cols[i]]) *4

    valid_score.append(np.mean(valid_score_t))
    best_itr_num.append(np.median(best_itr_num_t))
    best_itr_num_df.iloc[:,n] = pd.Series(best_itr_num_t)
    
    cv_predict.append(cv_predict_t)

print(f'{n_fold}-fold CV score: ', np.mean(valid_score))

df_4_ana = pd.concat([train,df_train_pred.add_prefix('pred_')],axis=1)
df_4_ana.to_csv("./data/ana/train_pred.csv")
test_X.to_csv("./data/ana/test_pred.csv")


test_result = pd.concat(cv_predict).groupby(level=0).median()


ind_params = []
for i, c in enumerate(target_columns):
    params_base = lgb_params.copy()
    if c in [target_columns[n] for n in [0,]]:
        params_base['learning_rate'] = 0.2 #0.4
        params_base['tweedie_variance_power'] = 1.99
        params_base['num_leaves']=511
        # params_base['max_bin']=127
        params_base['min_data_in_leaf']=100
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [1]]:
        params_base['learning_rate'] = 0.02
        params_base['tweedie_variance_power'] = 1.8
        params_base['num_leaves']=127
        # params_base['max_bin']=256
        params_base['min_data_in_leaf']=20
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [2]]:
        # params_base['learning_rate'] = 0.1
        params_base['tweedie_variance_power'] = 1.8  
        params_base['num_leaves']=255
        params_base['max_bin']=256
        params_base['min_data_in_leaf']=40
        ind_params.append(params_base)


results = pd.read_csv("./data/sample_submission.csv", index_col='datetime')
results4c = results.copy()

n_itr_list = [3, 5, 0.9] 

# 場所毎の予測を行う
for i, target in enumerate(tqdm(target_columns)):
    X_tmp = X[if_cols[i]]
    y_temp = y.copy()
    train_ds = lgb.Dataset(X_tmp, y_temp[target])
    #学習
    model = lgb.train(
            ind_params[i],
            train_ds,
            num_boost_round=int(best_itr_num_df.iloc[i].max()*n_itr_list[i]),
                          ) 
    #予測
    pred_y = model.predict(test_X[if_cols[i]])
    shap_values = model.predict(data=test_X, pred_contrib=True)
    pred_y = np.where(pred_y < 0, 0, pred_y)

    #予測結果を格納
    results[target] = pred_y
    results[target] = results[target].apply(lambda x: geometric_round(x))
    results[target] = results[target]*4
    
    results4c[target] = pred_y
    
#テスト結果の出力
submit_df = results.copy()

if submit_cv==1:
    submit_df = pd.DataFrame(test_result.to_numpy(), index=submit_df.index, columns=submit_df.columns)

submit_df = results.copy()

if submit_cv!=1:
    submit_df.to_csv(f'./sub/lgb_{n_fold}cv_{np.mean(valid_score)}_seed_{seed}.csv')
    results4c.to_csv(f'./sub/lgb_{n_fold}cv_seed_{seed}_4_en.csv')

