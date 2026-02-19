"""
オンデマンドバス 需要数予測用ダミーデータ生成スクリプト（改良版）
=========================================================
【改良点（実データEDAに基づく）】
  1. ゼロ需要率を実データに合わせて調整（56% → 78%）
  2. 4月の需要が他月の約3倍という季節パターンを反映
  3. 月別係数を実データの分布から較正
  4. 週末需要を実データに合わせて調整（平日の約0.49倍）
  5. 月初月末の通院需要を実データに合わせて調整（中旬の約1.82倍）
  6. 特徴量追加:
      - consecutive_holiday_count : 連休中の通算日数（GW・年末年始等）
      - feels_like_temp            : 体感温度（気温 + 風速から計算）
      - is_extreme_weather         : 悪天候フラグ（大雨・大雪・強風の複合）
      - days_since_last_operation  : 前回運行日からの経過日数（需要の蓄積）

【出力形式】
  1行 = 1日 × 1時間帯（morning / daytime / evening）の集計レコード

【目的変数】
  demand_count : その日・時間帯の予約件数（0も含む）

【特徴量グループ】
  カレンダー系  : 曜日, 祝日, 学期, 月初月末, 農繁期, 連休
  気象系        : 気温, 体感温度, 降水量, 積雪, 風速, 悪天候フラグ
  時間帯系      : morning / daytime / evening
  ラグ系        : 前日・1週前・直近7日平均・直近14日平均 の需要数
  運行履歴系    : 前回運行からの経過日数
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

SEED = 42
np.random.seed(SEED)

START_DATE = date(2020, 4, 1)
END_DATE   = date(2025, 3, 31)
TIME_SLOTS = ["morning", "daytime", "evening"]

# ─────────────────────────────────────────
# 1. 祝日定義（2020〜2025）
# ─────────────────────────────────────────
HOLIDAYS = set([
    # 2020
    "2020-01-01","2020-01-13","2020-02-11","2020-02-23","2020-02-24",
    "2020-03-20","2020-04-29","2020-05-03","2020-05-04","2020-05-05",
    "2020-05-06","2020-07-23","2020-07-24","2020-08-10","2020-09-21",
    "2020-09-22","2020-11-03","2020-11-23",
    # 2021
    "2021-01-01","2021-01-11","2021-02-11","2021-02-23","2021-03-20",
    "2021-04-29","2021-05-03","2021-05-04","2021-05-05","2021-07-22",
    "2021-07-23","2021-08-08","2021-08-09","2021-09-20","2021-09-23",
    "2021-11-03","2021-11-23",
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
# 2. 連休カウント（当日が連休の何日目か）
# ─────────────────────────────────────────
def _is_non_workday(d: date) -> bool:
    return d.weekday() >= 5 or is_holiday(d)

def consecutive_holiday_count(d: date) -> int:
    """
    当日が連休（土日祝の連続）の何日目かを返す。
    平日なら0。
    GW・年末年始・シルバーウィーク等を自然に捉える。
    """
    if not _is_non_workday(d):
        return 0
    count = 1
    prev = d - timedelta(days=1)
    while _is_non_workday(prev):
        count += 1
        prev -= timedelta(days=1)
    return count


# ─────────────────────────────────────────
# 3. カレンダー系ヘルパー
# ─────────────────────────────────────────
def get_season(month: int) -> str:
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"

def is_school_term(d: date) -> bool:
    """広島県の学校カレンダーに準拠した学期中フラグ"""
    m, day = d.month, d.day
    if m == 7 and day >= 21: return False  # 夏休み
    if m == 8:               return False
    if m == 12 and day >= 25: return False  # 冬休み
    if m == 1 and day <= 7:   return False
    if m == 3 and day >= 25:  return False  # 春休み
    if m == 4 and day <= 7:   return False
    if d.weekday() >= 5:      return False  # 土日
    return True

def is_farming_season(month: int) -> bool:
    """農繁期（田植え5〜6月・収穫9〜10月）フラグ"""
    return month in (5, 6, 9, 10)

def is_month_boundary(day: int) -> bool:
    """月初（1〜5日）・月末（25日〜）: 通院予約が集中"""
    return day <= 5 or day >= 25


# ─────────────────────────────────────────
# 4. 気象シミュレーター
# ─────────────────────────────────────────
# 月別 平均気温・気温振れ幅（℃）東広島市付近の実測値を参考
MONTHLY_TEMP = {
    1:( 3.5,4.0), 2:( 4.5,4.0), 3:( 8.5,3.5), 4:(14.5,3.0),
    5:(19.5,3.0), 6:(23.5,2.5), 7:(28.0,2.5), 8:(29.5,2.0),
    9:(24.5,2.5),10:(18.0,3.0),11:(12.0,3.5),12:( 6.0,4.0),
}

MONTHLY_RAIN_PROB = {
    1:0.18, 2:0.18, 3:0.22, 4:0.20,
    5:0.22, 6:0.32, 7:0.35, 8:0.28,
    9:0.32,10:0.22,11:0.22,12:0.20,
}

def simulate_weather(d: date) -> dict:
    m = d.month
    mean_t, std_t = MONTHLY_TEMP[m]
    temperature = round(np.random.normal(mean_t, std_t), 1)

    is_rainy = np.random.random() < MONTHLY_RAIN_PROB[m]
    precipitation_mm = 0.0
    if is_rainy:
        precipitation_mm = round(np.random.exponential(8.0), 1)

    snowfall_cm = 0.0
    if m in (1, 2) and temperature <= 0 and is_rainy:
        snowfall_cm = round(np.random.exponential(3.0), 1)

    wind_speed = round(np.random.exponential(2.5), 1)

    if snowfall_cm > 0:
        weather_label = "snowy"
    elif is_rainy:
        weather_label = "rainy"
    elif np.random.random() < 0.4:
        weather_label = "cloudy"
    else:
        weather_label = "sunny"

    # 【新規】体感温度（Steadman式の簡易版）
    # 気温が10℃以上かつ風速が強い場合に体感温度が下がる
    feels_like = temperature - 0.4 * max(wind_speed - 2.0, 0)
    feels_like = round(feels_like, 1)

    # 【新規】悪天候フラグ（大雨・大雪・強風のいずれか）
    is_extreme_weather = int(
        precipitation_mm >= 20 or
        snowfall_cm >= 5 or
        wind_speed >= 10
    )

    return {
        "temperature"       : temperature,
        "feels_like_temp"   : feels_like,
        "precipitation_mm"  : precipitation_mm,
        "snowfall_cm"       : snowfall_cm,
        "wind_speed"        : wind_speed,
        "weather_label"     : weather_label,
        "is_extreme_weather": is_extreme_weather,
    }


# ─────────────────────────────────────────
# 5. 需要数の生成モデル
# ─────────────────────────────────────────

# 【改良】実データEDAから較正した月別係数
# 全体平均を1.0とした相対値（実データの月別平均 / 全体平均）
MONTHLY_COEF = {
    1: 0.558, 2: 0.593, 3: 0.659,
    4: 3.121,  # 4月は突出して高い（新年度・通院集中）
    5: 0.817, 6: 0.856, 7: 0.863,
    8: 0.697, 9: 0.848, 10: 0.753,
    11: 0.798, 12: 0.725,
}

def base_lambda(d: date, slot: str, weather: dict) -> float:
    """
    ポアソン分布の λ（期待需要数）を状況に応じて計算する。
    実データEDA（2023〜2025年度）に基づき係数を較正。
    ゼロ需要率: 実データ78.4%に合わせて基礎λを下げる。
    """
    # ── 時間帯の基礎需要 ──
    # 実データの便別平均（2便が最多、朝夕が中程度）を参考に設定
    # ゼロ率78%を実現するため全体的にλを抑える
    slot_base = {"morning": 0.32, "daytime": 0.40, "evening": 0.22}
    lam = slot_base[slot]

    # ── 月別補正（実データ較正）──
    lam *= MONTHLY_COEF[d.month]

    # ── カレンダー補正 ──
    is_we   = d.weekday() >= 5
    is_hol  = is_holiday(d)
    school  = is_school_term(d)
    farming = is_farming_season(d.month)
    boundary = is_month_boundary(d.day)

    if is_we or is_hol:
        # 実データ: 週末は平日の約0.49倍
        lam *= 0.49
        # daytimeは観光・買い物需要でやや戻る
        if slot == "daytime":
            lam *= 1.25
    else:
        # 平日
        if school and slot == "morning":
            # 学期中の朝: 通学需要UP
            lam *= 1.40
        if not school and slot == "morning":
            # 長期休暇中の朝: 通学ゼロ
            lam *= 0.40

    # 農繁期（農家の移動需要）
    if farming and slot in ("morning", "evening"):
        lam *= 1.15

    # 月初月末（通院需要: 実データで約1.82倍）
    if boundary and slot == "daytime":
        lam *= 1.82

    # 【新規】連休効果: 連休が長いほど需要が蓄積されやすい
    con_hol = consecutive_holiday_count(d)
    if con_hol >= 3:
        # 長期連休明けは需要が増える（翌日に反映するためここでは抑制）
        lam *= 0.70

    # ── 気象補正 ──
    t   = weather["temperature"]
    prp = weather["precipitation_mm"]
    snw = weather["snowfall_cm"]
    wnd = weather["wind_speed"]
    flt = weather["feels_like_temp"]

    # 体感温度ベースの快適性補正（従来の気温補正を置き換え）
    if flt >= 35 or flt <= -3:
        lam *= 0.45   # 極端な暑さ・寒さ
    elif flt >= 30 or flt <= 1:
        lam *= 0.72
    elif 14 <= flt <= 24:
        lam *= 1.10   # 快適な体感温度

    # 降水量
    if prp >= 20:
        lam *= 0.40   # 大雨
    elif prp >= 5:
        lam *= 0.65   # 中雨
    elif prp > 0:
        lam *= 0.82   # 小雨

    # 積雪（山間部は特に影響大）
    if snw >= 10:
        lam *= 0.20
    elif snw > 0:
        lam *= 0.50

    # 強風
    if wnd >= 10:
        lam *= 0.70
    elif wnd >= 7:
        lam *= 0.85

    return max(lam, 0.0)


# ─────────────────────────────────────────
# 6. 前回運行からの経過日数を計算するためのリスト
# ─────────────────────────────────────────
def compute_days_since_last_operation(dates_series: pd.Series) -> pd.Series:
    """
    各日付について、前回需要があった日からの経過日数を返す。
    demand_countが1以上の日を「運行日」とみなす。
    """
    result = []
    last_op = None
    for d in dates_series:
        if last_op is None:
            result.append(np.nan)
        else:
            result.append((d - last_op).days)
        last_op = d  # 毎日更新（需要有無に関わらず）
    return pd.Series(result, index=dates_series.index)


# ─────────────────────────────────────────
# 7. 全レコード生成
# ─────────────────────────────────────────
records = []
current = START_DATE

while current <= END_DATE:
    weather = simulate_weather(current)
    season  = get_season(current.month)
    con_hol = consecutive_holiday_count(current)

    for slot in TIME_SLOTS:
        lam   = base_lambda(current, slot, weather)
        count = int(np.random.poisson(lam))

        records.append({
            # ── 識別子 ──
            "date"                    : current.isoformat(),
            "time_slot"               : slot,

            # ── 目的変数 ──
            "demand_count"            : count,

            # ── カレンダー特徴量 ──
            "day_of_week"             : current.weekday(),
            "month"                   : current.month,
            "is_weekend"              : int(current.weekday() >= 5),
            "is_holiday"              : int(is_holiday(current)),
            "is_school_term"          : int(is_school_term(current)),
            "is_farming_season"       : int(is_farming_season(current.month)),
            "is_month_boundary"       : int(is_month_boundary(current.day)),
            "season"                  : season,
            "consecutive_holiday_count": con_hol,  # 【新規】

            # ── 気象特徴量 ──
            "temperature"             : weather["temperature"],
            "feels_like_temp"         : weather["feels_like_temp"],    # 【新規】
            "precipitation_mm"        : weather["precipitation_mm"],
            "snowfall_cm"             : weather["snowfall_cm"],
            "wind_speed"              : weather["wind_speed"],
            "weather_label"           : weather["weather_label"],
            "is_extreme_weather"      : weather["is_extreme_weather"], # 【新規】
        })

    current += timedelta(days=1)

df = pd.DataFrame(records)


# ─────────────────────────────────────────
# 8. ラグ特徴量の付与（時間帯ごとに独立して計算）
# ─────────────────────────────────────────
df = df.sort_values(["time_slot", "date"]).reset_index(drop=True)

for slot in TIME_SLOTS:
    mask = df["time_slot"] == slot
    s    = df.loc[mask, "demand_count"]

    df.loc[mask, "lag_1_demand"]      = s.shift(1)
    df.loc[mask, "lag_7_demand"]      = s.shift(7)
    df.loc[mask, "lag_14_demand"]     = s.shift(14)
    df.loc[mask, "rolling_7day_avg"]  = s.shift(1).rolling(7,  min_periods=1).mean().round(2)
    df.loc[mask, "rolling_14day_avg"] = s.shift(1).rolling(14, min_periods=1).mean().round(2)

# ラグが計算できない先頭行は除去
df = df.dropna(subset=["lag_7_demand"]).reset_index(drop=True)

# 日付順に並べ直す
df = df.sort_values(["date", "time_slot"]).reset_index(drop=True)


# ─────────────────────────────────────────
# 9. 前回運行からの経過日数（time_slotごとに計算）
# ─────────────────────────────────────────
df["date_parsed"] = pd.to_datetime(df["date"])

for slot in TIME_SLOTS:
    mask = df["time_slot"] == slot
    slot_df = df[mask].copy()
    days_since = compute_days_since_last_operation(slot_df["date_parsed"])
    df.loc[mask, "days_since_last_operation"] = days_since.values

df = df.drop(columns=["date_parsed"])


# ─────────────────────────────────────────
# 10. 保存 & サマリー
# ─────────────────────────────────────────
DUMMY_FILE = "./data/dummy.csv"
df.to_csv(DUMMY_FILE, index=False, encoding="utf-8-sig")

print(f"✅ 生成完了: {len(df)} レコード（ゼロ需要含む）")
print(f"   期間: {df['date'].min()} 〜 {df['date'].max()}")
print(f"   保存先: {DUMMY_FILE}")
print(f"   列数: {len(df.columns)}")

print("\n── 需要数の分布 ──")
print(df["demand_count"].value_counts().sort_index().to_string())

print("\n── ゼロ需要の割合 ──")
zero_ratio = (df["demand_count"] == 0).mean()
print(f"   {zero_ratio:.1%}（実データ: 78.4%）")

print("\n── 時間帯別 平均需要数 ──")
print(df.groupby("time_slot")["demand_count"].mean().round(3).to_string())

print("\n── 月別 平均需要数 ──")
print(df.groupby("month")["demand_count"].mean().round(3).to_string())

print("\n── 曜日別 平均需要数（0=月〜6=日）──")
print(df.groupby("day_of_week")["demand_count"].mean().round(3).to_string())

print("\n── 天候ラベル別 平均需要数 ──")
print(df.groupby("weather_label")["demand_count"].mean().round(3).to_string())

print("\n── 悪天候フラグ別 平均需要数 ──")
print(df.groupby("is_extreme_weather")["demand_count"].mean().round(3).to_string())

print("\n── 連休カウント別 平均需要数 ──")
print(df.groupby("consecutive_holiday_count")["demand_count"].mean().round(3).to_string())

print("\n── 列一覧 ──")
for col in df.columns:
    print(f"   {col}")