import copy
import os
import gc
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
from lightgbm import LGBMRegressor
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import datetime as dt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
from torch import nn
from skorch import NeuralNet
from skorch.callbacks import Callback, LRScheduler
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import skew


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class GRUClassifier(nn.Module):
    def __init__(self, params):
        super(GRUClassifier, self).__init__()
        self.layers = params['layer']
        self.transformer_encoder = nn.ModuleList()        
        self.emb = nn.Sequential(nn.Dropout(params['dropout_0']), nn.Linear(params['input_size'], params['hidden_0']))
        for l in range(self.layers):
            self.transformer_encoder.append(nn.TransformerEncoderLayer(params['hidden_0'],\
                                                       nhead=params['head'],\
                                                       dim_feedforward=params['hidden_1'],\
                                                       dropout=params['dropout_1'],\
                                                       activation = params['transformer_activation']))
        self.dense2 = nn.Sequential(nn.Linear(params['hidden_0'], params['hidden_3']),params['Activation'],nn.Dropout(params['dropout_3']),nn.Linear(params['hidden_3'], params['target_dim']))
        # self.dense2 = nn.Sequential(nn.Linear(params['hidden_0'], params['hidden_3']),
        #                             nn.LayerNorm([params['hidden_3']]),
        #                             params['Activation'],
        #                             nn.Dropout(params['dropout_3']),
        #                             nn.Linear(params['hidden_3'], params['target_dim']))

    def forward(self, X_input):
        x = self.emb(X_input)
        x = x.unsqueeze(dim=1)
        x = x.permute(1,0,2)
        for l in range(self.layers):
            x = self.transformer_encoder[l](x)
        x = x.permute(1,0,2).squeeze()        
        x = self.dense2(x)
        return x

def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < 0.5, np.floor(arr), result_array)
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where((result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)))&(result_array >= 0.5), np.ceil(arr), result_array)
    return result_array

class StratifiedKFoldReg(StratifiedKFold):
    def split(self, X, y, groups=None):
        
        n_samples = len(y)
        
        # Number of labels to discretize our target variable,
        # into bins of quasi equal size
        n_labels = int(np.round(n_samples/self.n_splits))
        
        # Assign a label to each bin of n_splits points
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits) \
            for ii in range(n_labels)])
        
        # Get number of points that would fall
        # out of the equally-sized bins
        mod = np.mod(n_samples, self.n_splits)
        
        # Find unique idxs of first unique label's ocurrence
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)
        
        # sample randomly the label idxs to which assign the 
        # the mod points
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)

        # insert these at the beginning of the corresponding bin
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])
        
        # find each element of y to which label corresponds in the sorted 
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label
    
        # put labels according to the given y order then
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
# train[['winddirection_chiba', 'winddirection_tokyo']] = train[['winddirection_chiba', 'winddirection_tokyo']].round().astype(int)

# 訓練データのカラムを格納
columns = train.columns.values.tolist()
l_train = len(train)



# #　訓練データに使うカラムと、予測するカラムを設定
target_columns = columns[13:]
drop_cols = ['datetime', 'time', 'ymdh'] #'year', 
# cat_feats = ['weekday']
# # config
# # CVのvaildデータ数
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
    #splitにのみ効く
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
# tmp_sum = df_all.groupby('ymd')[tmp_feat].agg(np.sum)
tmp_max_min = df_all.groupby('ymd')[tmp_feat].agg(lambda x: max(x) - min(x))
# tmp_max_min_d = df_all.groupby('ymd')[tmp_feat].agg(lambda x: min(x) / max(x))
for i, col in enumerate(tmp_feat):
    df_all[f'{col}_day_max'] = df_all['ymd'].map(lambda x: tmp_max[col][x]).values
    df_all[f'{col}_day_min'] = df_all['ymd'].map(lambda x: tmp_min[col][x]).values
    df_all[f'{col}_day_mean'] = df_all['ymd'].map(lambda x: tmp_mean[col][x]).values
    # df_all[f'{col}_day_sum'] = df_all['ymd'].map(lambda x: tmp_sum[col][x]).values
    df_all[f'{col}_day_max_min'] = df_all['ymd'].map(lambda x: tmp_max_min[col][x]).values
    # df_all[f'{col}_day_max_min_d'] = df_all['ymd'].map(lambda x: tmp_max_min_d[col][x]).values
df_all = df_all.drop(columns='ymd')

# 6月の気温から次の年の総花粉量の相対目安を算出
def Jun_tmp(df, feat:list, group:str, alpha=0.005):
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
                # df_all[f'{col}_SG_53_der1'] = savgol_filter(df_all[col], sg_window, sg_polyorder, deriv=1)
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



df_all = pd.concat([train, test])


df_all = df_all.fillna(0)


#cliping
num_cols =  df_all.drop(columns=target_columns).columns.values.tolist()
p00 = df_all[num_cols].quantile(0.01).fillna(-(np.inf))
p99 = df_all[num_cols].quantile(0.99).fillna(np.inf)
df_all[num_cols] = df_all[num_cols].clip(p00, p99, axis=1)
p99inf = p99[p99 == np.inf]
p00inf = p00[p00 == -np.inf]
for infindex in p99inf.index.values:
    infu = df_all[infindex].unique()
    infu = np.sort(infu)[::-1]
    p99[infindex] = infu[1]
    df_all[infindex] = df_all[infindex].clip(p00.at[infindex], infu[1])
for infindex in p00inf.index.values:
    infu = df_all[infindex].unique()
    infu = np.sort(infu)
    p00[infindex] = infu[1]
    df_all[infindex] = df_all[infindex].clip(infu[1], p99.at[infindex])

df_all = df_all.reset_index(drop=True)
   
#カテゴリカルでない特徴量
X_nc = df_all.drop(columns=target_columns)
# X_nc = X
X_nc = X_nc.loc[:,(X_nc.nunique(dropna=False)>3)]
skewed_feats = X_nc.apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats.abs() > 0.7].index
#歪度0.7より大きい特徴量を対数変換
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df_all[skewed_feats] = pt.fit_transform(df_all[skewed_feats])


#標準化
num_cols =  df_all.drop(columns=target_columns)
n_col_name =  num_cols.columns.values.tolist()
scaler = StandardScaler()
scaler.fit(num_cols)
df_all[n_col_name] = pd.DataFrame(scaler.transform(num_cols), columns=n_col_name)
df_all = df_all.reset_index(drop=True)


# train test 再分割分
train = df_all.iloc[0:l_train,:].reset_index(drop=True)
test = df_all.iloc[l_train:,:].drop(columns=target_columns).reset_index(drop=True)



#異常値除外
train = train[(train['pollen_utsunomiya'] >= 0)&(train['pollen_chiba'] >= 0)&(train['pollen_tokyo'] >= 0)]


# 訓練データを、説明変数Xと目的変数Yに分割
X = train.drop(columns=target_columns)
# X = sm.add_constant(x)
y = train[target_columns] / 4

# テストデータの説明変数test_Xを作成
test_X = test.copy()
# test_X = sm.add_constant(tx)

# 予測結果を保存する辞書型データを作成
results = dict({})
ind_feat = [X.columns.tolist() for c in range(len(target_columns))]    


params = {    'Activation': nn.SiLU(),
              'dropout_0': 0.3, #0.3, 0.1
              'hidden_0': 512,
              # 'dropout_4': 0.1,
              'head': 8,
              'hidden_1': 2048,
              "dropout_1": 0.0,
              'transformer_activation': 'gelu',
              # "embedding_drop": 0.1,
              # 'hidden_2': 512,
              # "dropout_2": 0.2,
              'hidden_3': 512,
              'dropout_3': 0.5,
              "layer": 1,

              'input_size': X.shape[-1],
              'target_dim': 1,
             } 

ind_params = []

ind_feat_mode = 1
if ind_feat_mode == 1:
    if_cols = copy.copy(ind_feat)
else:
    if_cols = copy.copy(ind_feat_null)

# CVは時間がかかるので使わない
n_fold = 10
sfkfold = StratifiedKFoldReg(n_splits=n_fold, random_state=seed, shuffle=True)    
# sfgkfold = StratifiedGroupKFold(n_splits=num_fold, random_state=seed, shuffle=True)
# kfold = KFold(n_splits=num_fold, random_state=seed, shuffle=True)

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


ind_params = []
optimizer_lr_l = []
for i, c in enumerate(target_columns):
    params_base = params.copy()
    if c in [target_columns[n] for n in [0,]]:
        params_base['dropout_0'] = 0.0
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(1e-7)
    elif c in [target_columns[n] for n in [1]]:
        params_base['dropout_0'] = 0.0
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(1e-7)
    elif c in [target_columns[n] for n in [2]]:
        params_base['dropout_0'] = 0.0
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(1e-7)



results = pd.read_csv("./data/sample_submission.csv", index_col='datetime')
results4c = results.copy()

# epoch数リスト　batch_sizeを小さくしているので低めに設定
n_itr_list = [24, 31, 15] #20, 30, 10


# 場所毎の予測を作成する
for i, target in enumerate(tqdm(target_columns)):
    X_tmp = X[if_cols[i]].to_numpy(dtype='float32')
    y_temp = y[target].to_numpy(dtype='float32')
    
    lrscheduler = LRScheduler(policy=ExponentialLR, gamma=0.99)

    #学習を実施
    dnn_model = NeuralNet(
                        module=GRUClassifier,
                        max_epochs=n_itr_list[i],
                        device = device,
                        batch_size=32,
                        module__params = ind_params[i],
                        # warm_start=True,
                        optimizer=torch.optim.AdamW,
                        # optimizer__batas = (0.9, 0.999),
                        # optimizer__eps = 1e-8,
                        optimizer__lr = optimizer_lr_l[i],
                        # optimizer__weight_decay = 0.0,
                        criterion=nn.HuberLoss(),
                        criterion__delta=5,
                        # iterator_train__shuffle=True,
                        # iterator_train__num_workers=4,
                        # iterator_train__pin_memory=True,
                        # iterator_valid__num_workers=4,
                        callbacks = [lrscheduler],
                        train_split=None,
                    ) 
    dnn_model.fit(X_tmp, y_temp)

    #予測を実施
    pred_y = dnn_model.predict(test_X[if_cols[i]].to_numpy(dtype='float32'))
    pred_y = np.where(pred_y < 0, 0, pred_y)

    #予測結果を格納
    
    results[target] = pred_y
    results[target] = results[target].apply(lambda x: geometric_round(x))
    results[target] = results[target]*4
    
    results4c[target] = pred_y
    
    del dnn_model
    gc.collect()
    

#テスト結果の出力
submit_df = results.copy()

# if submit_cv==1:
#     submit_df = pd.DataFrame(test_result.to_numpy(), index=submit_df.index, columns=submit_df.columns)

submit_df = results.copy()

if submit_cv!=1:
    submit_df.to_csv(f'./sub/dnn_{n_fold}cv_seed_{seed}.csv')
    results4c.to_csv(f'./sub/dnn_{n_fold}cv_seed_{seed}_4_en.csv')
