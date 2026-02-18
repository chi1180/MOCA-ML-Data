"""
オンデマンドバス 需要数予測用ダミーデータ生成スクリプト
=========================================================
【出力形式】
  1行 = 1日 × 1時間帯（morning / daytime / evening）の集計レコード

【目的変数】
  demand_count : その日・時間帯の予約件数（0も含む）

【特徴量グループ】
  カレンダー系  : 曜日, 祝日, 学期, 月初月末, 農繁期
  気象系        : 気温, 降水量, 積雪, 風速
  時間帯系      : morning / daytime / evening
  ラグ系        : 前日・1週前・直近7日平均 の需要数
"""

import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

SEED = 42
np.random.seed(SEED)

START_DATE = date(2022, 4, 1)
END_DATE   = date(2025, 3, 31)
TIME_SLOTS = ["morning", "daytime", "evening"]

# ─────────────────────────────────────────
# 0. ファイルパス
# ─────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
CSV_FILE   = os.path.join(DATA_DIR, "expanded_points.csv")
DUMMY_FILE = os.path.join(DATA_DIR, "dummy.csv")

df_stops = pd.read_csv(CSV_FILE)

# ─────────────────────────────────────────
# 1. 祝日定義（2022〜2025）
# ─────────────────────────────────────────
HOLIDAYS = set([
    # 2022
    "2022-01-01","2022-01-10","2022-02-11","2022-02-23","2022-03-21",
    "2022-04-29","2022-05-03","2022-05-04","2022-05-05","2022-07-18",
    "2022-08-11","2022-09-19","2022-09-23","2022-10-10","2022-11-03",
    "2022-11-23","2022-12-31",
    # 2023
    "2023-01-01","2023-01-02","2023-01-09","2023-02-11","2023-02-23",
    "2023-03-21","2023-04-29","2023-05-03","2023-05-04","2023-05-05",
    "2023-07-17","2023-08-11","2023-09-18","2023-09-23","2023-10-09",
    "2023-11-03","2023-11-23",
    # 2024
    "2024-01-01","2024-01-08","2024-02-11","2024-02-12","2024-02-23",
    "2024-03-20","2024-04-29","2024-05-03","2024-05-04","2024-05-05",
    "2024-05-06","2024-07-15","2024-08-11","2024-08-12","2024-09-16",
    "2024-09-22","2024-09-23","2024-10-14","2024-11-03","2024-11-04",
    "2024-11-23",
    # 2025
    "2025-01-01","2025-01-13","2025-02-11","2025-02-23","2025-02-24",
    "2025-03-20",
])

def is_holiday(d: date) -> bool:
    return d.isoformat() in HOLIDAYS

# ─────────────────────────────────────────
# 2. カレンダー系ヘルパー
# ─────────────────────────────────────────

def get_season(month: int) -> str:
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"

def is_school_term(d: date) -> bool:
    """広島県の学校カレンダーに準拠した学期中フラグ"""
    m, day = d.month, d.day
    # 夏休み: 7/21〜8/31
    if m == 7 and day >= 21: return False
    if m == 8:               return False
    # 冬休み: 12/25〜1/7
    if m == 12 and day >= 25: return False
    if m == 1 and day <= 7:   return False
    # 春休み: 3/25〜4/7
    if m == 3 and day >= 25:  return False
    if m == 4 and day <= 7:   return False
    # 土日は授業なし
    if d.weekday() >= 5:      return False
    return True

def is_farming_season(month: int) -> bool:
    """農繁期（田植え5〜6月・収穫9〜10月）フラグ"""
    return month in (5, 6, 9, 10)

def is_month_boundary(day: int) -> bool:
    """月初（1〜5日）・月末（25日〜）: 通院予約が集中"""
    return day <= 5 or day >= 25

# ─────────────────────────────────────────
# 3. 気象シミュレーター
# ─────────────────────────────────────────

# 月別 平均気温・気温振れ幅（℃）東広島市付近の実測値を参考
MONTHLY_TEMP = {
    1:( 3.5,4.0), 2:( 4.5,4.0), 3:( 8.5,3.5), 4:(14.5,3.0),
    5:(19.5,3.0), 6:(23.5,2.5), 7:(28.0,2.5), 8:(29.5,2.0),
    9:(24.5,2.5),10:(18.0,3.0),11:(12.0,3.5),12:( 6.0,4.0),
}

# 月別 降水確率・雪の確率（1〜2月のみ）
MONTHLY_RAIN_PROB = {
    1:0.18, 2:0.18, 3:0.22, 4:0.20,
    5:0.22, 6:0.32, 7:0.35, 8:0.28,
    9:0.32,10:0.22,11:0.22,12:0.20,
}

def simulate_weather(d: date) -> dict:
    m = d.month
    mean_t, std_t = MONTHLY_TEMP[m]
    temperature = round(np.random.normal(mean_t, std_t), 1)

    # 降水確率
    is_rainy = np.random.random() < MONTHLY_RAIN_PROB[m]
    precipitation_mm = 0.0
    if is_rainy:
        # 指数分布で雨量をシミュレーション（平均8mm）
        precipitation_mm = round(np.random.exponential(8.0), 1)

    # 積雪（1〜2月、かつ気温0℃以下の場合）
    snowfall_cm = 0.0
    if m in (1, 2) and temperature <= 0 and is_rainy:
        snowfall_cm = round(np.random.exponential(3.0), 1)

    # 風速（m/s）
    wind_speed = round(np.random.exponential(2.5), 1)

    # 天候ラベル
    if snowfall_cm > 0:
        weather_label = "snowy"
    elif is_rainy:
        weather_label = "rainy"
    elif np.random.random() < 0.4:
        weather_label = "cloudy"
    else:
        weather_label = "sunny"

    return {
        "temperature"     : temperature,
        "precipitation_mm": precipitation_mm,
        "snowfall_cm"     : snowfall_cm,
        "wind_speed"      : wind_speed,
        "weather_label"   : weather_label,
    }

# ─────────────────────────────────────────
# 4. 需要数の生成モデル
# ─────────────────────────────────────────

def base_lambda(d: date, slot: str, weather: dict) -> float:
    """
    ポアソン分布の λ（期待需要数）を状況に応じて計算する。
    福富町の規模感（平均 1〜2件/日）を維持しつつ、
    各要因がどのように需要を増減させるかを明示的にモデル化。
    """
    # ── 時間帯の基礎需要 ──
    slot_base = {"morning": 0.90, "daytime": 0.75, "evening": 0.45}
    lam = slot_base[slot]

    # ── カレンダー補正 ──
    is_we      = d.weekday() >= 5
    is_hol     = is_holiday(d)
    school     = is_school_term(d)
    farming    = is_farming_season(d.month)
    boundary   = is_month_boundary(d.day)

    if is_we or is_hol:
        # 週末・祝日: 通勤通学ゼロ → 全体需要減
        lam *= 0.55
        # ただし daytime の観光・買い物需要はやや戻る
        if slot == "daytime":
            lam *= 1.30
    else:
        # 平日
        if school and slot == "morning":
            # 学期中の朝 → 通学・送迎需要UP
            lam *= 1.50
        if not school and slot == "morning":
            # 長期休暇中の朝 → 通学ゼロ
            lam *= 0.40

    # 農繁期（朝・夕の農家移動が増える）
    if farming and slot in ("morning", "evening"):
        lam *= 1.20

    # 月初月末（通院需要 → daytime に効く）
    if boundary and slot == "daytime":
        lam *= 1.35

    # ── 気象補正 ──
    t   = weather["temperature"]
    prp = weather["precipitation_mm"]
    snw = weather["snowfall_cm"]
    wnd = weather["wind_speed"]

    # 気温: 極端な暑さ寒さは外出抑制
    if t >= 35 or t <= -2:
        lam *= 0.50
    elif t >= 30 or t <= 2:
        lam *= 0.75
    elif 15 <= t <= 25:
        lam *= 1.10   # 快適な気温は外出促進

    # 降水量
    if prp >= 20:
        lam *= 0.40   # 大雨
    elif prp >= 5:
        lam *= 0.65   # 中雨
    elif prp > 0:
        lam *= 0.80   # 小雨

    # 積雪（山間部は特に影響大）
    if snw >= 10:
        lam *= 0.20   # 大雪: ほぼ外出不可
    elif snw > 0:
        lam *= 0.50

    # 強風
    if wnd >= 10:
        lam *= 0.70
    elif wnd >= 7:
        lam *= 0.85

    return max(lam, 0.0)

# ─────────────────────────────────────────
# 5. 全レコード生成（ゼロ需要も含む）
# ─────────────────────────────────────────
records = []
current = START_DATE

while current <= END_DATE:
    weather = simulate_weather(current)
    season  = get_season(current.month)

    for slot in TIME_SLOTS:
        lam   = base_lambda(current, slot, weather)
        count = int(np.random.poisson(lam))   # ← ゼロも自然に発生

        records.append({
            # ── 識別子 ──
            "date"            : current.isoformat(),
            "time_slot"       : slot,

            # ── 目的変数 ──
            "demand_count"    : count,

            # ── カレンダー特徴量 ──
            "day_of_week"     : current.weekday(),          # 0=月〜6=日
            "month"           : current.month,
            "is_weekend"      : int(current.weekday() >= 5),
            "is_holiday"      : int(is_holiday(current)),
            "is_school_term"  : int(is_school_term(current)),
            "is_farming_season": int(is_farming_season(current.month)),
            "is_month_boundary": int(is_month_boundary(current.day)),
            "season"          : season,

            # ── 気象特徴量 ──
            "temperature"     : weather["temperature"],
            "precipitation_mm": weather["precipitation_mm"],
            "snowfall_cm"     : weather["snowfall_cm"],
            "wind_speed"      : weather["wind_speed"],
            "weather_label"   : weather["weather_label"],
        })

    current += timedelta(days=1)

df = pd.DataFrame(records)

# ─────────────────────────────────────────
# 6. ラグ特徴量の付与（時間帯ごとに独立して計算）
# ─────────────────────────────────────────
df = df.sort_values(["time_slot", "date"]).reset_index(drop=True)

for slot in TIME_SLOTS:
    mask = df["time_slot"] == slot
    s    = df.loc[mask, "demand_count"]

    df.loc[mask, "lag_1_demand"]      = s.shift(1)
    df.loc[mask, "lag_7_demand"]      = s.shift(7)
    df.loc[mask, "lag_14_demand"]     = s.shift(14)
    df.loc[mask, "rolling_7day_avg"]  = s.shift(1).rolling(7, min_periods=1).mean().round(2)
    df.loc[mask, "rolling_14day_avg"] = s.shift(1).rolling(14, min_periods=1).mean().round(2)

# ラグが計算できない先頭行は除去
df = df.dropna(subset=["lag_7_demand"]).reset_index(drop=True)

# 日付順に並べ直す
df = df.sort_values(["date", "time_slot"]).reset_index(drop=True)

# ─────────────────────────────────────────
# 7. 保存 & サマリー
# ─────────────────────────────────────────
df_out = df
df_out.to_csv(DUMMY_FILE, index=False, encoding="utf-8-sig")

print(f"✅ 生成完了: {len(df_out)} レコード（ゼロ需要含む）")
print(f"   期間: {df_out['date'].min()} 〜 {df_out['date'].max()}")
print(f"   保存先: {DUMMY_FILE}")

print("\n── 需要数の分布 ──")
print(df_out["demand_count"].value_counts().sort_index().to_string())

print("\n── 時間帯別 平均需要数 ──")
print(df_out.groupby("time_slot")["demand_count"].mean().round(3).to_string())

print("\n── 曜日別 平均需要数（0=月〜6=日）──")
print(df_out.groupby("day_of_week")["demand_count"].mean().round(3).to_string())

print("\n── 天候ラベル別 平均需要数 ──")
print(df_out.groupby("weather_label")["demand_count"].mean().round(3).to_string())

print("\n── 学期中 vs 休暇中（morning） ──")
morning = df_out[df_out["time_slot"] == "morning"]
print(morning.groupby("is_school_term")["demand_count"].mean().round(3).to_string())

print("\n── ゼロ需要の割合 ──")
zero_ratio = (df_out["demand_count"] == 0).mean()
print(f"   {zero_ratio:.1%} の時間帯で需要ゼロ")

print("\n── 列一覧 ──")
for col in df_out.columns:
    print(f"   {col}")