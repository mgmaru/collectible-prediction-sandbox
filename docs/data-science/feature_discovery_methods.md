# 特徴量を探す方法

作成日: 2026-07-05 JST

## このドキュメントの目的

今回の議論では、次の疑問を扱った。

```text
グラフを目視して特徴量を考える以外に、
機械的・統計的に特徴量を探す方法はあるのか
```

結論としては、方法は複数ある。

ただし、今回の学習課題では、最初から自動化しすぎず、次の順番で進めるのがよい。

```text
目視で仮説を作る
↓
意味を説明できる特徴量を作る
↓
ラベルを作る
↓
モデルで学習する
↓
機械的な方法で特徴量の有効性を確認する
↓
特徴量を改善する
```

## 特徴量探しの方法一覧

| 方法 | 内容 | 今回の使いどころ |
|---|---|---|
| 目視観察 | グラフを見てパターンを探す | 最初にやる |
| ルールベース特徴量作成 | 変化率、移動合計、イベント前後などを作る | 次にやる中心 |
| 統計的な特徴量選択 | ラベルと関係が強い特徴量を数値で見る | ラベル作成後に使う |
| モデルベース重要度 | モデルが重視した特徴量を見る | 学習後に使う |
| Permutation Importance | 特徴量を壊したとき性能が落ちるか見る | モデル評価後に使う |
| RFE | 重要度の低い特徴量を順番に削る | 特徴量が増えた後に使う |
| SHAP | 予測に効いた特徴量を説明する | 発展課題向き |
| 自動特徴量抽出 | 時系列から大量の特徴量を自動生成する | 発展課題向き |

## 1. 目視観察

### 何をするか

折れ線グラフを見て、人間がパターンを探す。

今回見るグラフ:

- `price`
- `sns_mentions`
- `trends_score`
- `listing_count`
- イベント縦線

### 見るポイント

| 観察 | 特徴量候補 |
|---|---|
| 最近価格が上がっている | `price_change_7d`, `price_change_30d` |
| SNSが価格より先に増えている | `sns_change_7d` |
| 検索関心が価格より先に増えている | `trends_change_7d` |
| 出品数が増えたあと価格が下がっている | `listing_change_7d` |
| 配布終了後に価格が上がっている | `days_to_distribution_end`, `is_after_distribution_end` |

### 位置づけ

目視観察は、厳密な検証ではない。

役割は、次のような仮説を作ること。

```text
SNSが増えると、数日後に価格が上がるかもしれない
出品数が増えると、価格が下がるかもしれない
配布終了後に価格が上がるかもしれない
```

この仮説を、あとで特徴量とモデルで検証する。

## 2. ルールベース特徴量作成

### 何をするか

人間が意味を説明できる特徴量を作る。

今回の課題では、まずこの方法が一番重要。

### 例

| 特徴量 | 元データ | 意味 |
|---|---|---|
| `current_price` | `price` | 現在価格 |
| `current_listing_count` | `listing_count` | 現在の出品数 |
| `price_change_7d` | `price` | 7日前からの価格変化率 |
| `price_change_30d` | `price` | 30日前からの価格変化率 |
| `sns_change_7d` | `sns_mentions` | 7日前からのSNS言及数変化率 |
| `trends_change_7d` | `trends_score` | 7日前からの検索関心変化率 |
| `listing_change_7d` | `listing_count` | 7日前からの出品数変化率 |
| `sold_count_7d_sum` | `sold_count` | 直近7日間の売却件数合計 |
| `days_to_distribution_end` | `days_to_distribution_end` | 配布終了までの日数 |

### よく使う考え方

| 種類 | 意味 | 例 |
|---|---|---|
| ラグ特徴量 | 過去の値を使う | 7日前の価格 |
| 変化率 | 過去と現在の差を見る | 7日価格変化率 |
| ローリング集計 | 直近N日の合計や平均を見る | 直近7日の売却件数 |
| イベント特徴量 | イベント前後かを見る | 配布終了後かどうか |

### 参考

- pandas `DataFrame.shift`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
- pandas `Series.rolling`: https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html
- Feature-engine LagFeatures: https://feature-engine.trainindata.com/en/1.8.x/api_doc/timeseries/forecasting/LagFeatures.html

## 3. 統計的な特徴量選択

### 何をするか

ラベルを作った後に、

```text
どの特徴量が target_up_30d_30pct と関係していそうか
```

を数値で見る。

### 代表的な方法

| 方法 | 内容 |
|---|---|
| `SelectKBest` | スコアが高い特徴量を上位K個選ぶ |
| `f_classif` | 分類ラベルとの線形的な関係を見る |
| `mutual_info_classif` | 分類ラベルとの依存関係を見る |

### 注意点

この方法は、ラベルが必要。

つまり、先に次を作る必要がある。

```text
target_up_30d_30pct
```

また、未来情報が混ざった特徴量に対して統計的選択をすると、意味のない高スコアが出る。

そのため、必ず次を守る。

```text
特徴量 = 予測時点までに見えている情報
ラベル = 予測時点より未来の結果
```

### 参考

- scikit-learn Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
- scikit-learn SelectKBest: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- scikit-learn mutual_info_classif: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

## 4. モデルベース重要度

### 何をするか

モデルを学習した後に、

```text
モデルがどの特徴量を重視したか
```

を見る。

### 例

| モデル | 見るもの |
|---|---|
| ロジスティック回帰 | 係数 |
| Random Forest | feature importance |
| 勾配ブースティング系 | feature importance |

### 今回の使い方

最初のモデルはロジスティック回帰を想定している。

そのため、モデル作成後に、

```text
どの特徴量の係数が大きいか
プラス方向かマイナス方向か
```

を見るとよい。

ただし、係数を見るときは特徴量のスケールに注意する。
値の大きさが違う特徴量をそのまま比べると、解釈を誤ることがある。

## 5. Permutation Importance

### 何をするか

学習済みモデルに対して、特徴量を1つずつランダムにシャッフルする。

その結果、モデルの性能がどれくらい落ちるかを見る。

```text
ある特徴量を壊す
↓
モデル性能が大きく落ちる
↓
その特徴量は重要そう
```

### なぜ便利か

Permutation Importanceは、モデル種類に依存しにくい。

ロジスティック回帰でも、Random Forestでも使いやすい。

### 注意点

モデルの性能がそもそも低い場合、重要度の解釈も弱くなる。

また、似た特徴量が複数あると、重要度が分散することがある。

### 参考

- scikit-learn Permutation Importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- scikit-learn `permutation_importance`: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

## 6. RFE

### 何をするか

RFEは、特徴量を全部入れて学習し、重要度が低い特徴量を順番に削る方法。

```text
全特徴量で学習
↓
弱い特徴量を削る
↓
また学習
↓
残す特徴量を選ぶ
```

### 今回の位置づけ

最初から使う必要はない。

特徴量が増えてきて、

```text
どれを残すべきか分からない
```

となった段階で使う。

### 参考

- scikit-learn RFE: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

## 7. SHAP

### 何をするか

SHAPは、予測に対して、

```text
どの特徴量が、どれくらい予測に影響したか
```

を説明する方法。

モデル全体の重要度だけでなく、個別商品の予測理由も見やすい。

### 今回の位置づけ

発展課題向き。

最初のロジスティック回帰、評価、Permutation Importanceまでできてからでよい。

### 参考

- SHAP documentation: https://shap.readthedocs.io/

## 8. 自動特徴量抽出

### 何をするか

時系列データから大量の特徴量を自動生成する。

代表例:

- 平均
- 最大値
- 分散
- ピーク数
- 変化量
- 周期性に関する特徴

### 代表的なライブラリ

`tsfresh`

### 今回の位置づけ

発展課題向き。

最初から使うと、

```text
なぜその特徴量が必要なのか
どういう意味の特徴量なのか
```

が分かりにくくなる。

今回の目的は学習なので、まずは自分で意味を説明できる特徴量を作る方がよい。

### 参考

- tsfresh documentation: https://tsfresh.readthedocs.io/
- tsfresh extracted features: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
- tsfresh rolling forecasting: https://tsfresh.readthedocs.io/en/latest/text/forecasting.html

## 今回のおすすめ順

このプロジェクトでは、以下の順番で進める。

```text
1. グラフを目視で観察する
2. 気づいたパターンをメモする
3. 意味を説明できる特徴量を作る
4. ラベルを作る
5. 時系列分割でtrain/testに分ける
6. ロジスティック回帰で学習する
7. 評価指標を見る
8. 係数やPermutation Importanceで特徴量の効き方を見る
9. 必要なら特徴量を追加・削除する
10. 発展としてRFE、SHAP、tsfreshを試す
```

### なぜこの順番か

| 順番 | 理由 |
|---|---|
| 目視観察を先にする | データの動きを理解するため |
| ルールベース特徴量を先に作る | 意味を説明しやすいため |
| ラベル作成後に統計的選択を使う | ラベルとの関係を見るには正解データが必要なため |
| モデル学習後に重要度を見る | 重要度はモデルがあって初めて評価できるため |
| 自動特徴量抽出は後にする | 最初から使うと学習目的から外れやすいため |

## 今回の最初の実践ステップ

まずは、10商品程度を観察する。

観察メモには、次を書く。

```text
商品名:
価格の動き:
SNS言及数の動き:
検索関心の動き:
出品数の動き:
イベントとの関係:
特徴量に使えそうなもの:
```

その後、観察結果をもとに、最初の特徴量セットを作る。

最初の候補:

| 特徴量 | 理由 |
|---|---|
| `current_price` | 現在の価格水準を見るため |
| `current_listing_count` | 現在の供給量を見るため |
| `price_change_7d` | 短期の価格変化を見るため |
| `price_change_30d` | 中期の価格変化を見るため |
| `sns_change_7d` | 話題化の短期変化を見るため |
| `trends_change_7d` | 検索関心の短期変化を見るため |
| `listing_change_7d` | 供給量の短期変化を見るため |
| `sold_count_7d_sum` | 直近の売れ行きを見るため |
| `days_to_distribution_end` | 配布終了との距離を見るため |

## 注意点

### 未来情報を入れない

特徴量は、予測時点までに見えている情報だけで作る。

例:

```text
2025-02-01に予測する行
```

であれば、使ってよいのは `2025-02-01` までの情報。

使ってはいけないのは、`2025-02-02` 以降の情報。

### ラベルだけは未来を見る

ラベルは正解なので、未来を見て作る。

例:

```text
target_up_30d_30pct
= 30日後に現在価格より+30%以上上がったか
```

整理すると、次の関係になる。

| 種類 | 使う情報 |
|---|---|
| 特徴量 | 予測時点までの過去・現在 |
| ラベル | 予測時点より未来の結果 |

## まとめ

特徴量探しには、目視以外にも機械的な方法がある。

ただし、今回の課題では次の考え方が重要。

```text
目視で仮説を作る
機械的な方法で後から検証する
```

最初から自動化しすぎるより、まずは自分で意味を説明できる特徴量を作る。

その後、モデルの係数、Permutation Importance、統計的選択などで、実際に効いているかを確認する。
