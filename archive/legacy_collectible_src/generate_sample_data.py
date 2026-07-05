"""
Collectible Prediction Sandbox - サンプルデータ生成スクリプト (Step 1: 欠損なし)

課題ドキュメント docs/assignment/collectible_prediction_sandbox_assignment.md の
セクション 5・6 に沿って、架空のコレクタブル商品の日次履歴データを生成します。

  - 商品数 : 30
  - 期間   : 180 日
  - 粒度   : 商品 × 日付 の日次データ
  - 欠損   : なし (Step 1)
  - 価格パターン: 上昇型 / 初動天井型 / 横ばい型 / 再販下落型 /
                  海外需要後追い型 / 配布終了後上昇型

出力 (src/data/):
  - items.csv            商品マスタ
  - events.csv           イベント一覧 (実際に起きたイベントのみ)
  - daily_snapshots.csv  日次スナップショット (メインの学習用データ)

このデータの目的は「本番の予測精度を確認すること」ではなく、
特徴量作成・ラベル作成・モデル学習・評価・可視化の流れを練習することです。
価格・SNS・検索量・出品数はある程度連動しますが、単純すぎない程度にノイズを入れています。

標準ライブラリのみで動作します (pandas 等は不要):
    python3 src/generate_sample_data.py
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SEED = 42
N_ITEMS = 30
N_DAYS = 180
START_DATE = date(2025, 1, 1)
OUT_DIR = Path(__file__).resolve().parent / "data"

# カテゴリ (release_type と 想定価格帯 / 出品数上限)
CATEGORIES = {
    "bookstore_bonus": {"jp": "書店特典カード", "price": (300, 1500), "listing": (20, 150)},
    "movie_bonus": {"jp": "映画特典カード", "price": (500, 2500), "listing": (20, 120)},
    "promo_card": {"jp": "プロモカード", "price": (400, 3000), "listing": (15, 130)},
    "sealed_box": {"jp": "未開封BOX", "price": (3000, 15000), "listing": (5, 40)},
}

IP_POOL = [
    "ONE PIECE", "呪術廻戦", "鬼滅の刃", "推しの子", "SPY×FAMILY",
    "架空IP-A", "架空IP-B", "架空TCG", "架空アニメC", "架空ゲームD",
]

# 価格パターンごとの商品数 (合計 30)
PATTERN_PLAN = {
    "rising": 6,                 # 上昇型
    "initial_spike": 5,          # 初動天井型
    "flat": 6,                   # 横ばい型
    "restock_decline": 5,        # 再販下落型
    "overseas_follow": 4,        # 海外需要後追い型
    "post_distribution_rise": 4, # 配布終了後上昇型
}

PATTERN_JP = {
    "rising": "上昇型",
    "initial_spike": "初動天井型",
    "flat": "横ばい型",
    "restock_decline": "再販下落型",
    "overseas_follow": "海外需要後追い型",
    "post_distribution_rise": "配布終了後上昇型",
}


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def smoothstep(x):
    """0->1 の滑らかな S 字。x<=0 で 0、x>=1 で 1。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return x * x * (3 - 2 * x)


def ramp(t, start, span):
    """start から span 日かけて 0->1 に立ち上がる。"""
    if span <= 0:
        return 1.0 if t >= start else 0.0
    return smoothstep((t - start) / span)


def bump(t, center, width):
    """center を頂点とする釣鐘状のインパルス (0..1)。"""
    return math.exp(-((t - center) ** 2) / (2 * width * width))


# ---------------------------------------------------------------------------
# 商品マスタの生成
# ---------------------------------------------------------------------------
def build_items(rng):
    patterns = []
    for name, count in PATTERN_PLAN.items():
        patterns.extend([name] * count)
    rng.shuffle(patterns)

    cat_names = list(CATEGORIES.keys())
    items = []
    for i, pattern in enumerate(patterns):
        category = rng.choice(cat_names)
        cat = CATEGORIES[category]
        ip = rng.choice(IP_POOL)
        letter = chr(ord("A") + (i % 26))
        item = {
            "item_id": f"item_{i + 1:03d}",
            "item_name": f"{ip} {cat['jp']}{letter}",
            "ip_name": ip,
            "category": category,
            "release_type": "limited",
            "price_pattern": pattern,       # 正解パターン (メタ情報 / 特徴量には使わない)
            "base_price": rng.randint(*cat["price"]),
            "listing_max": rng.randint(*cat["listing"]),
        }
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# 1 商品分の時系列 (価格倍率・ヒート・出品調整・イベント) を作る
# ---------------------------------------------------------------------------
def build_timeline(rng, pattern):
    announce = rng.randint(8, 30)
    dist_start = announce + rng.randint(10, 18)
    if pattern == "post_distribution_rise":
        # 配布終了後の上昇を観測するため、終了日は期間内で後ろに余白を残す
        dist_end = min(dist_start + rng.randint(30, 55), N_DAYS - 45)
    else:
        dist_end = dist_start + rng.randint(35, 80)
    return {"announce": announce, "dist_start": dist_start, "dist_end": dist_end}


def build_series(rng, pattern, tl):
    """price_mult, heat, listing_extra, events を返す。"""
    N = N_DAYS
    ann, ds, de = tl["announce"], tl["dist_start"], tl["dist_end"]

    price_mult = [1.0] * N
    heat = [0.06] * N          # 全体の注目度 (SNS / 検索量を駆動)
    listing_extra = [0.0] * N  # 出品数への追加分 (正=増加, 負=減少)
    events = {}

    events[ann] = "announcement"
    if ds < N:
        events[ds] = "distribution_start"
    if de < N:
        events[de] = "distribution_end"

    # どのパターンでも告知・配布開始で少し注目が上がる
    for t in range(N):
        heat[t] += 0.10 * bump(t, ann, 4) + 0.12 * bump(t, ds, 5)

    if pattern == "rising":
        rise_start = ds + rng.randint(4, 12)
        span = rng.randint(30, 55)
        target = rng.uniform(1.45, 2.0)
        yt_day = clamp(rise_start + rng.randint(-3, 6), 0, N - 1)
        events[yt_day] = "youtube_feature"
        for t in range(N):
            price_mult[t] = 1 + (target - 1) * ramp(t, rise_start, span)
            heat[t] += 0.9 * ramp(t, rise_start - 10, span * 0.8)  # ヒートが価格に先行
            heat[t] += 0.25 * bump(t, yt_day, 5)

    elif pattern == "initial_spike":
        peak = ds + rng.randint(4, 9)
        peak_mult = rng.uniform(1.9, 2.7)
        floor_mult = rng.uniform(0.8, 1.0)
        decay = rng.uniform(18, 30)
        for t in range(N):
            if t < ds:
                price_mult[t] = 1.0
            elif t <= peak:
                price_mult[t] = 1 + (peak_mult - 1) * smoothstep((t - ds) / (peak - ds))
            else:
                price_mult[t] = floor_mult + (peak_mult - floor_mult) * math.exp(-(t - peak) / decay)
            heat[t] += 1.1 * bump(t, peak - 2, 6)         # ヒートは価格の少し手前で急騰
            listing_extra[t] += 1.0 * ramp(t, peak + 2, 12)  # 出品が増えて下落を招く

    elif pattern == "flat":
        m = 1.0
        for t in range(N):
            m = clamp(m + rng.uniform(-0.012, 0.012), 0.85, 1.15)
            price_mult[t] = m
            heat[t] += 0.04 * bump(t, ann, 8)  # 注目は弱いまま

    elif pattern == "restock_decline":
        peak_mult = rng.uniform(1.25, 1.55)
        rise_span = rng.randint(20, 35)
        restock_day = min(ds + rng.randint(30, 60), N - 25)
        events[restock_day] = "restock"
        drop_span = rng.randint(18, 30)
        drop_target = rng.uniform(0.72, 0.9)
        for t in range(N):
            up = (peak_mult - 1) * ramp(t, ds + 5, rise_span)
            down = (peak_mult - drop_target) * ramp(t, restock_day, drop_span)
            price_mult[t] = 1 + up - down
            heat[t] += 0.5 * ramp(t, ds + 2, rise_span)
            heat[t] += 0.20 * bump(t, restock_day, 5)
            listing_extra[t] += 1.4 * ramp(t, restock_day, 8)  # 再販で出品急増

    elif pattern == "overseas_follow":
        os_day = rng.randint(max(ds + 5, 35), 120)
        events[os_day] = "overseas_signal"
        lag = rng.randint(8, 16)
        span = rng.randint(25, 45)
        target = rng.uniform(1.4, 1.9)
        for t in range(N):
            price_mult[t] = 1 + (target - 1) * ramp(t, os_day + lag, span)  # 遅れて価格上昇
            heat[t] += 0.9 * ramp(t, os_day, span * 0.7)                    # 海外シグナルが先行
            heat[t] += 0.3 * bump(t, os_day, 5)

    elif pattern == "post_distribution_rise":
        span = rng.randint(30, 50)
        target = rng.uniform(1.35, 1.7)
        for t in range(N):
            price_mult[t] = 1 + (target - 1) * ramp(t, de + 2, span)  # 終了後に上昇
            heat[t] += 0.5 * ramp(t, de - 3, span * 0.8)
            listing_extra[t] += -0.6 * ramp(t, de + 1, 20)           # 供給が細る

    return price_mult, heat, listing_extra, events


# ---------------------------------------------------------------------------
# 観測値 (price, listing_count, sold_count, sns_mentions, trends_score...) に変換
# ---------------------------------------------------------------------------
def build_records(rng, item, tl, price_mult, heat, listing_extra, events):
    N = N_DAYS
    ds, de = tl["dist_start"], tl["dist_end"]
    base_price = item["base_price"]
    l_max = item["listing_max"]

    records = []
    for t in range(N):
        d = START_DATE + timedelta(days=t)
        h = max(0.02, heat[t])

        # 価格: パターン倍率 + 日次の細かいノイズ
        price = round(base_price * price_mult[t] * (1 + rng.uniform(-0.02, 0.02)))

        # 検索量 (0-100) と SNS 言及数はヒートから生成
        trends = int(clamp(round(6 + 62 * h + rng.uniform(-6, 6)), 0, 100))
        sns = max(0, round((2 + 45 * h) * (1 + rng.uniform(-0.25, 0.25))))

        # 出品数: 配布開始後に立ち上がり + パターン調整 + 価格高騰時のわずかな増加
        base_listing = l_max * (0.15 + 0.85 * ramp(t, ds, 12))
        base_listing *= (1 + 0.3 * (price_mult[t] - 1))
        base_listing += l_max * 0.5 * listing_extra[t]
        listing = max(0, round(base_listing * (1 + rng.uniform(-0.15, 0.15))))

        # 売却件数: 需要(ヒート)に比例しつつ、出回っている量で頭打ち
        demand = (0.6 + 3.0 * h) * (0.6 + 0.8 * rng.random())
        sold = max(0, round(min(demand, listing * 0.5 + 1)))

        etype = events.get(t, "none")
        records.append({
            "date": d.isoformat(),
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "ip_name": item["ip_name"],
            "category": item["category"],
            "release_type": item["release_type"],
            "price": price,
            "listing_count": listing,
            "sold_count": sold,
            "sns_mentions": sns,
            "trends_score": trends,
            "event_flag": 0 if etype == "none" else 1,
            "event_type": etype,
            "days_to_distribution_end": de - t,  # 負の値 = 配布終了後
        })
    return records


# ---------------------------------------------------------------------------
# 診断: 30日後 +30% ラベルのバランスを確認 (CSV には保存しない)
# ---------------------------------------------------------------------------
def summarize_labels(all_records_by_item, items):
    print("\n--- 30日後 +30% ラベル (target_up_30d_30pct) の内訳 ---")
    pattern_pos = {p: 0 for p in PATTERN_PLAN}
    pattern_tot = {p: 0 for p in PATTERN_PLAN}
    total_pos = total = 0
    for item in items:
        recs = all_records_by_item[item["item_id"]]
        pat = item["price_pattern"]
        for t in range(N_DAYS - 30):
            future = recs[t + 30]["price"]
            now = recs[t]["price"]
            label = 1 if future >= now * 1.30 else 0
            pattern_pos[pat] += label
            pattern_tot[pat] += 1
            total_pos += label
            total += 1
    for p in PATTERN_PLAN:
        tot = pattern_tot[p]
        rate = pattern_pos[p] / tot if tot else 0
        print(f"  {PATTERN_JP[p]:<12} 正例 {pattern_pos[p]:5d} / {tot:5d}  ({rate:5.1%})")
    print(f"  {'全体':<12} 正例 {total_pos:5d} / {total:5d}  ({total_pos / total:5.1%})")


# ---------------------------------------------------------------------------
# CSV 書き出し
# ---------------------------------------------------------------------------
def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = build_items(rng)

    daily_rows = []
    event_rows = []
    records_by_item = {}

    for i, item in enumerate(items):
        item_rng = random.Random(SEED * 1000 + i)  # 商品ごとに独立・再現可能
        tl = build_timeline(item_rng, item["price_pattern"])
        price_mult, heat, listing_extra, events = build_series(item_rng, item["price_pattern"], tl)
        recs = build_records(item_rng, item, tl, price_mult, heat, listing_extra, events)

        records_by_item[item["item_id"]] = recs
        daily_rows.extend(recs)
        for t, etype in sorted(events.items()):
            event_rows.append({
                "item_id": item["item_id"],
                "date": (START_DATE + timedelta(days=t)).isoformat(),
                "event_type": etype,
            })

    # items.csv (listing_max は内部用なので出力しない)
    item_rows = [{
        "item_id": it["item_id"],
        "item_name": it["item_name"],
        "ip_name": it["ip_name"],
        "category": it["category"],
        "release_type": it["release_type"],
        "price_pattern": it["price_pattern"],
    } for it in items]

    write_csv(OUT_DIR / "items.csv", item_rows,
              ["item_id", "item_name", "ip_name", "category", "release_type", "price_pattern"])
    write_csv(OUT_DIR / "events.csv", event_rows,
              ["item_id", "date", "event_type"])
    write_csv(OUT_DIR / "daily_snapshots.csv", daily_rows,
              ["date", "item_id", "item_name", "ip_name", "category", "release_type",
               "price", "listing_count", "sold_count", "sns_mentions", "trends_score",
               "event_flag", "event_type", "days_to_distribution_end"])

    print(f"生成完了 -> {OUT_DIR}")
    print(f"  items.csv          : {len(item_rows):5d} 行 (商品マスタ)")
    print(f"  events.csv         : {len(event_rows):5d} 行 (イベント)")
    print(f"  daily_snapshots.csv: {len(daily_rows):5d} 行 ({N_ITEMS} 商品 × {N_DAYS} 日)")
    summarize_labels(records_by_item, items)


if __name__ == "__main__":
    main()
