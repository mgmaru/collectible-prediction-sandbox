# collectible-prediction-sandbox

機械学習の予測モデル構築を学ぶための小さな練習用リポジトリです。

当初は、架空のコレクタブル商品データを使って「30日後に価格が +30% 以上上がるか」を予測する流れを学ぶ構成でした。
現在は、短期間でモデル評価と改善プロセスを学ぶため、正解データ付きの実データを使う課題を主課題にしています。

> 本番アプリ `Collectible Value Radar` の縮小版という位置づけです。高精度なモデルを作ることが目的ではなく、モデル構築の流れを理解することが目的です。

## 学習方針の更新

短期間で機械学習の理解を深めるため、主課題は正解データ付きの実データを使う **Bike Sharing Demand Model Tuning Sandbox** に移行します。

- 新しい主課題: [docs/assignment/bike_sharing_model_tuning_assignment.md](docs/assignment/bike_sharing_model_tuning_assignment.md)
- 既存のコレクタブル課題: ML全体の流れを理解する Phase 0 として残す
- 使用データ: UCI Bike Sharing Dataset (`data/raw/bike_sharing/hour.csv`)

## 目的

- 実データを使って、予測モデルの性能を評価する
- ベースライン、特徴量追加、モデル変更によるスコア変化を確認する
- MAE, RMSE, R2 などの回帰評価指標を理解する
- 誤差分析により、モデルが苦手な条件を探す
- Streamlitで、モデル改善の試行錯誤を可視化する

既存のコレクタブル商品データ課題では、以下を Phase 0 として学びます。

- サンプルデータを作り、履歴データ（価格・SNS・検索量・出品数・イベント）を扱う
- 特徴量（7日/30日変化率、イベントまでの日数など）を作る
- ラベル（30日後に +30% 以上上昇したか）を作る
- ロジスティック回帰などで予測モデルを学習・評価する
- 予測結果や価格推移をグラフ・簡易UIで確認する

## 技術スタック（予定）

| 用途 | ツール |
|---|---|
| 言語 | Python |
| データ処理 | pandas |
| モデル | scikit-learn |
| グラフ | matplotlib / plotly |
| 簡易アプリ | Streamlit |
| データ保存 | CSV |

## 全体の流れ

```text
実データ読み込み → EDA → ベースライン作成 → 特徴量作成
→ train/test分割 → モデル学習 → 評価 → 誤差分析 → Streamlitで表示
```

## 進め方（4週間）

| 週 | 目標 |
|---:|---|
| 1週目 | Bike Sharingデータを読み、EDAとベースラインを作る |
| 2週目 | 特徴量を追加し、評価指標の変化を見る |
| 3週目 | 複数モデルを比較し、誤差分析をする |
| 4週目 | Streamlitで実験条件・評価・改善履歴を表示する |

## ディレクトリ構成

```text
.
├── docs/assignment/          課題仕様
└── src/
    ├── generate_sample_data.py   サンプルデータ生成スクリプト (Step 1: 欠損なし)
    └── data/                     生成済みCSVとデータ辞書 (README参照)
```

## サンプルデータ

Step 1（欠損なし・30商品・180日）のサンプルデータを生成できます。

```bash
python3 src/generate_sample_data.py
```

生成物は `src/data/`（`daily_snapshots.csv` / `items.csv` / `events.csv`）に出力されます。
カラム定義や価格パターンは [src/data/README.md](src/data/README.md) を参照してください。

## ステータス

開発中。

- 新しい主課題仕様: [docs/assignment/bike_sharing_model_tuning_assignment.md](docs/assignment/bike_sharing_model_tuning_assignment.md)
- 既存の導入課題仕様: [docs/assignment/collectible_prediction_sandbox_assignment.md](docs/assignment/collectible_prediction_sandbox_assignment.md)
- 取得済み実データ: `data/raw/bike_sharing/hour.csv`
