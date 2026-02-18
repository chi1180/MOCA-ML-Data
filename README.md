# MOCA-ML-Data

バス停留所のタグデータから機械学習用の特徴量を生成するプロジェクト

## 概要

このプロジェクトは、バス停留所に付与されたタグ情報を基に、機械学習モデルで使用できる特徴量スコアを自動生成します。タグとベースタグの類似度計算により、教育・観光・福祉・地形・交通などの各カテゴリスコアを算出します。

## 主な機能

- **タグベースの特徴量生成**: 停留所のタグから複数のカテゴリスコアを自動計算
- **ベクトル類似度計算**: 未知のタグに対しても、類似するベースタグから適切なスコアを推定
- **多言語対応の埋め込みモデル**: `intfloat/multilingual-e5-small` を使用した高精度なベクトル表現
- **柔軟なスコアリング**: 教育、観光、福祉、地形、交通、住宅、商業など多角的な評価指標

## インストール

### 前提条件

- Python 3.12 以上
- uv (推奨) または pip

### セットアップ

1. リポジトリのクローン:
```bash
git clone https://github.com/chi1180/MOCA-ML-Data.git
cd MOCA-ML-Data
```

2. 依存関係のインストール:

**uvを使用する場合 (推奨):**
```bash
uv sync
```

**pipを使用する場合:**
```bash
pip install -r requirements.txt
# または
pip install pandas requests scikit-learn sentence-transformers
```

## 使い方

### 基本的な実行

```bash
python main.py
```

このコマンドにより:
1. バス停留所データの取得（APIまたはキャッシュから）
2. ベースタグの読み込み
3. 各停留所のタグを分析してスコアを計算
4. 結果を `data/expanded_points.csv` に出力

### データフロー

```
停留所データ (API/Cache)
    ↓
タグの抽出
    ↓
┌─────────────────────┐
│ タグがベースタグに  │
│ 存在するか？        │
└─────────────────────┘
    │              │
   Yes             No
    │              │
    ↓              ↓
直接スコア取得   ベクトル類似度計算
                 (top 5 類似タグ)
    │              │
    └──────┬───────┘
           ↓
    スコアの集計・平均化
           ↓
    expanded_points.csv
```

## プロジェクト構成

```
MOCA-ML-Data/
├── README.md              # このファイル
├── pyproject.toml         # プロジェクト設定と依存関係
├── main.py                # メインスクリプト
├── src/                   # ソースコード
│   ├── base_tags.py       # ベースタグの読み込み
│   ├── data_fetch.py      # データ取得（API/キャッシュ）
│   ├── vector.py          # テキストのベクトル化
│   └── dummy.py           # ダミーデータ生成（開発用）
├── data/                  # データファイル
│   ├── base_tags.csv      # ベースタグとスコア定義
│   ├── points_cache.csv   # 停留所データキャッシュ
│   └── expanded_points.csv # 出力: 拡張された停留所データ
└── doc/                   # ドキュメント
    └── src.dummy.readme.md # ダミーデータ生成の詳細

```

## 生成されるスコア

各停留所について、以下のスコアが生成されます:

### カテゴリスコア
- `education_score`: 教育関連施設の近さ・重要度
- `tourism_score`: 観光地としての魅力度
- `welfare_score`: 福祉施設の充実度
- `terrain_score`: 地形的特徴（平坦性など）
- `transport_score`: 交通アクセスの利便性
- `residential_score`: 住宅地としての性格
- `commercial_score`: 商業施設の充実度

### 需要パターン関連
- `base_demand_score`: 基礎需要レベル
- `morning_peak_factor`: 朝のピーク需要係数
- `evening_peak_factor`: 夕方のピーク需要係数
- `daytime_factor`: 日中の需要係数
- `weekend_factor`: 週末の需要係数

### 環境要因
- `weather_sensitivity`: 天候の影響度
- `seasonal_variation`: 季節変動の大きさ

### 停留所分類
- `stop_type`: 停留所のタイプ分類

## データソース

- **停留所データAPI**: `https://moca-jet.vercel.app/api/stops`
- **ベースタグ**: `data/base_tags.csv` に定義された参照タグとスコア

## 技術仕様

### 使用ライブラリ

- **pandas**: データ操作と処理
- **scikit-learn**: コサイン類似度計算
- **sentence-transformers**: 多言語テキスト埋め込み
- **requests**: API通信

### ベクトル埋め込みモデル

- モデル: `intfloat/multilingual-e5-small`
- 目的: タグのセマンティック類似度計算
- 利点: 日本語を含む多言語対応、軽量で高速

## 開発

### 開発環境のセットアップ

```bash
uv sync --group dev
```

これにより、開発用の依存関係（JupyterKernelなど）もインストールされます。

### データキャッシュの更新

APIから最新データを取得するには、キャッシュファイルを削除してから実行:

```bash
rm data/points_cache.csv
python main.py
```

## ライセンス

このプロジェクトのライセンス情報については、リポジトリの所有者にお問い合わせください。

## 貢献

バグ報告や機能要望は、GitHubのIssuesでお願いします。

---

# MOCA-ML-Data (English)

A project for generating machine learning features from bus stop tag data.

## Overview

This project automatically generates feature scores for machine learning models based on tag information assigned to bus stops. It calculates category scores (education, tourism, welfare, terrain, transportation, etc.) using similarity calculations between tags and base tags.

## Key Features

- **Tag-based Feature Generation**: Automatically calculates multiple category scores from stop tags
- **Vector Similarity Computation**: Estimates appropriate scores for unknown tags using similar base tags
- **Multilingual Embedding Model**: High-precision vector representation using `intfloat/multilingual-e5-small`
- **Flexible Scoring**: Multi-faceted evaluation metrics including education, tourism, welfare, terrain, transportation, residential, and commercial aspects

## Installation

### Prerequisites

- Python 3.12 or higher
- uv (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chi1180/MOCA-ML-Data.git
cd MOCA-ML-Data
```

2. Install dependencies:

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install pandas requests scikit-learn sentence-transformers
```

## Usage

### Basic Execution

```bash
python main.py
```

This command will:
1. Fetch bus stop data (from API or cache)
2. Load base tags
3. Analyze tags for each stop and calculate scores
4. Output results to `data/expanded_points.csv`

## Technical Specifications

### Libraries Used

- **pandas**: Data manipulation and processing
- **scikit-learn**: Cosine similarity calculation
- **sentence-transformers**: Multilingual text embedding
- **requests**: API communication

### Vector Embedding Model

- Model: `intfloat/multilingual-e5-small`
- Purpose: Semantic similarity calculation for tags
- Advantages: Multilingual support including Japanese, lightweight and fast

## License

Please contact the repository owner for license information.

## Contributing

Bug reports and feature requests are welcome via GitHub Issues.
