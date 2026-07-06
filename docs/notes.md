# notes

## ドキュメント概要
`ml-model-tuning-sandbox`の開発における反省(やったこと、わかったこと、疑問点など)を記載する。

## 反省
### 2026-07-01
#### やったこと
- `uv`で`Python`の環境を構築
- `csv`ファイルを読み込む簡単なスクリプトを作成
- `Python`の基本知識の勉強
#### わかったこと
- `Python`におけるリスト
  - `list`：可変長、型が混在
  - `array`：可変長、型固定
  - `ndarray`：配列/Vecの連続メモリ（数値計算用の配列）
  - `tuple`：固定長、変更不可（イミュータブル）
---
### 2026-07-02 / 2026-07-03
#### やったこと
- 機能を実装
    - `CSV`読み込み機能（データソース読み込み）
    - 選択した商品の現在価格、SNS言及数、出品数、検索関心を折れ線グラフで表示する機能
    - スナップショット、イベントデータを表で表示
#### わかったこと
- `basedpyright`は、Pythonの静的型チェッカー。
  - microsoftの`pyright`をフォークしたもので、`pyright`よりも厳格。
  - 主に、`Any`を検出して、制限してくれる。
- `Dataframe`の特定の行に対する重複削除は、`drop_duplicates()`で簡単にできる。
- `Dataframe`の特定の行に対するソートは、`sort_values()`で簡単にできる。
#### わからなかったところ
- 実装の際に、一時変数でおくべきか、おかなくても良いかが迷った。
1. 一時変数で置く例：
```python
def extract_selected_item_data(
  selected_item: str, items_data: pd.DataFrame
) -> pd.DataFrame:
    selected_item_data = items_data[items_data["item_name"] == selected_item]
  return selected_item_data
```
2. 一時変数で置かない例：
```python
def extract_selected_item_data(
  selected_item: str, items_data: pd.DataFrame
) -> pd.DataFrame:
  return items_data[items_data["item_name"] == selected_item]
```
---
### 2026-07-05
#### やったこと
- サンプルデータを架空のデータから実データに置き換え（bike_sharingへ）
- csvを読み込んで、
#### わかったこと
- `hour.csv`などの時間ごとのデータを時系列で表すときは、横軸は時間まで直したものにする。
  - 修正前：`dteday`（`2011-01-01`）
  - 修正後：`dteday` + `hr` （`2011-01-01 05:00:00`）
実装例：
```python
data["datetime"] = (
    pd.to_datetime(data["dteday"])
    + pd.to_timedelta(data["hr"], unit="h")
)
```
- `pandas`のデータフレームにおいて条件で行抽出を行うとき、`and`は使えない。
  - エラー：`data[data["dteday"] >= start_date and data["dteday"] <= end_date]`
  - 正しい：`data[(data["dteday"] >= start_date) & (data["dteday"] <= end_date)]`
- データフレームの時刻については、**日付型に変換（`pd.to_datetime()`を使用）して、条件比較などを行う。**
  - 良くない：`data[data["dteday"] >= start_date and data["dteday"] <= end_date]`
  - 正しい：`data[(data["dteday"] >= pd.to_datetime(start_date)) & (data["dteday"] <= pd.to_datetime(end_date))]`
---
