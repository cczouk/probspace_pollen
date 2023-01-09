# ProbSpace 花粉飛散量予測 1st Place Solution

ProbSpaceで開催された花粉飛散量予測のsolutionです。

## 解法の要約

- LightGBM, XGBoost, Catboost, NNのweight ensemble
- 損失関数はtweedie, huber, rmseで使い分け

## 解法の各要素

### 特徴量エンジニアリング

#### 効いていそうな特徴量
- ラグ特徴 - 降水量・気温・風速・風向
- 積算特徴量 - 降水量・気温
- 雨量0連続カウント
- aggregation特徴量（平均・最大・最小） - 気温


#### 効果が不明な特徴量
- 風のベクトル化
- 6月の気温から次の年の総花粉量の相対目安を算出（初年は空欄）
- 積算気温による相対積算花粉飛散量予測
- 花粉好条件判定（効果が薄そうなため、千葉のみ実装して中止）
- Savitzky-Golay filterによる特徴量 - 風速


### 学習・モデル

- モデル
  - LightGBM, XGBoost, Catboost, NN
- 損失関数
  - 各`pollen`を4で割った値を目的変数としてtweedie, huber, rmseを使用。  
        商品の販売数を予測するタスクでLightGBMとtweedieを使っているのを見たことがあったので参考にしました。  
        個人的な経験則ですが、LightGBMとtweedieでうまくいく場合でも、XGBoost, Catboostではtweedieを使っても高い精度になるとは限らないので、モデルの多様性も加味してhuberも使っています。  
                rmseについては後述します。
- ハイパーパラメータ
  - LightGBMのモデルをOptunaとLBスコアを参考に設定。  
        他のモデルをLightGBMを参考に設定。  
        ※時間がなかったので調整したパラメータは絞ってあります。
  - CVのbest_iterationを基にboost_roundを設定。  
        学習するデータが増えることや、アンサンブルを念頭に置いて、数をcvより増やして過学習気味にしています。

#### NNのモデル
NNのモデルは自前で以前に実装したTransformerをベースにしたモデルです。  
テーブルデータに使えそうな手持ちのモデルをパラメータごと使いまわしています。  
アンサンブル用として精度は期待せずに割り切って使っています。

### アンサンブル・後処理

- 基本はtweedie, huberを使って得られた各モデルの予測値のweight ensembleです。
- アンサンブル後に整数値に丸め、4をかけています。
- さらに、予測値が一定以上になった場合、rmseで得られた予測値のアンサンブル結果に置き換えています。  
   外れ値のような大きな値へのフィッティングはtweedie, huberよりもrmseが優れていると考えました。

- アンサンブルの重みはLBスコアを参考に調整しましたが、  
    コードにミスがあるのに気づかずに調整していたこともあり、かなり雑になってしまっているので  
    調整の余地はまだありそうです。


### 所感

どの要素がどの程度スコアに貢献しているか検証できていないのですが、
アンサンブルと損失関数の使い分けがよかったのではないかと思っています。  
トレーニングデータとテストデータでは花粉の数の分布が違っていたのではないかと推測されますが、  
上記の方法でうまく対応できたことがスコアの向上につながっているのではないかと思っています。

### 試していないアイディア

- 調整していないハイパラの調整
- 次元削減系（PCA, umapなど）の特徴量
- 特徴量同士の四則演算
- 特徴量選択
- 予測値の補正  
  花粉の数は単純な4の倍数ではなく、20～40おきに+1された数？になっている
- NNモデルで分類タスクとして予測
- 最適なNNモデルの作成
- アンサンブルするモデル追加（sklearnのHistGradientBoostingなど）

## コード

- 'pred_***_0.1.py': 各基本モデル
- 'pred_ensemble_0.2.py': アンサンブル・後処理
- 'pred_***_0.1_mse.py': 各基本モデルとrmseを使用した際の差分
