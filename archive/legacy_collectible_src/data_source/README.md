# サンプルデータ (Step 1: 欠損なし)

`src/generate_sample_data.py` が生成する架空のコレクタブル履歴データです。
再生成するには次を実行します。

```bash
python3 src/generate_sample_data.py
```

- 商品数: 30 / 期間: 180日 (2025-01-01 〜 2025-06-29) / 粒度: 商品 × 日付
- 欠損: なし（Step 1）
- 乱数シード固定のため、何度生成しても同じ内容になります

> このデータの目的は本番の予測精度確認ではなく、特徴量→ラベル→学習→評価→可視化の
> 流れを練習することです。高精度が出ても本番で当たることは意味しません。

## ファイル

### `daily_snapshots.csv` — メインの日次データ（5,400行）

| カラム | 型 | 説明 |
|---|---|---|
| date | date | 観測日 (YYYY-MM-DD) |
| item_id | str | 商品ID |
| item_name | str | 商品名 |
| ip_name | str | 作品名 |
| category | str | bookstore_bonus / movie_bonus / promo_card / sealed_box |
| release_type | str | limited |
| price | int | 現在価格（円） |
| listing_count | int | 出品数 |
| sold_count | int | 売却件数（当日） |
| sns_mentions | int | SNS言及数 |
| trends_score | int | Google Trends 風の検索関心（0–100） |
| event_flag | int | その日にイベントがあれば 1、なければ 0 |
| event_type | str | イベント種類。なければ `none` |
| days_to_distribution_end | int | 配布終了までの日数（負の値=配布終了後） |

### `items.csv` — 商品マスタ（30行）

| カラム | 説明 |
|---|---|
| item_id, item_name, ip_name, category, release_type | 商品の基本属性 |
| price_pattern | 価格の正解パターン（下表）。**メタ情報であり、特徴量には使わないこと** |

### `events.csv` — イベント一覧（実際に起きたイベントのみ）

| カラム | 説明 |
|---|---|
| item_id | 商品ID |
| date | イベント発生日 |
| event_type | announcement / distribution_start / distribution_end / restock / youtube_feature / overseas_signal |

## 価格パターン

商品ごとに以下いずれかのパターンで動きます。モデルがどのパターンを拾えるかを観察できます。

| price_pattern | 動き |
|---|---|
| rising（上昇型） | SNS・検索量が先行し、価格が徐々に上がる |
| initial_spike（初動天井型） | 配布直後に高騰し、出品増加で下落する |
| flat（横ばい型） | 注目が弱く価格も動かない |
| restock_decline（再販下落型） | 再販イベント後に出品が増えて下落する |
| overseas_follow（海外需要後追い型） | 海外シグナル後、少し遅れて価格が上がる |
| post_distribution_rise（配布終了後上昇型） | 配布終了後に供給が細り価格が上がる |

## 想定ラベル（この段階では未生成）

課題セクション8のラベルはまだ作成していません（特徴量とあわせて次のステップで作ります）。
参考として、`target_up_30d_30pct`（30日後に+30%以上上昇=1）を作った場合の正例率は約 **13%**、
`flat` は 0%、`rising` は約 21% と、パターンごとに差が出るよう設計しています。
