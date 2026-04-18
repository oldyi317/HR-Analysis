# HR 離職風險分析 — People Analytics 作品集

以 IBM HR Analytics Attrition 公開資料集為基礎，建立可解釋、可落地的員工離職風險預測分析。
本專案不是 Kaggle 教學式的「跑一個模型看 AUC」，而是同時服務 **Data Science reviewer** 與 **People Analytics / HR reviewer** 的雙讀者作品集。

## 專案特色

相對於一般離職預測範例，本專案強化四大能力：

| # | 改良重點 | 實作位置 |
|---|---|---|
| 1 | **SHAP 可解釋性** — TreeExplainer + LinearExplainer + summary / waterfall / dependence plots | Section 9 |
| 2 | **豐富的 Feature Engineering** — 新增 4 個 HR 複合指標（無 leakage 設計） | Section 7 |
| 3 | **嚴謹的模型評估** — PR-AUC 為主要指標、ECE 量化校準、Lift Table | Section 8 |
| 4 | **Monitoring Layer** — Intervention Queue 含 `PrimarySignal` / `SuggestedAction` / `ReviewCadence` | Section 10 |

## 結果速覽（IBM HR Attrition 1,470 筆資料）

- **最佳模型**：Logistic-L1 + Sigmoid Calibration
- **PR-AUC**：0.58（baseline = 0.16）
- **ROC-AUC**：0.80
- **ECE**：0.04（經校準，機率可直接作為預算排序依據）
- **Lift @ Top 10%**：顯著高於隨機
- **Intervention Queue 產出**：147 位高風險員工，每位均附首要風險訊號與具體介入建議

### 三個結構性發現（HR 語言）

1. 加班文化是組織的結構性痛點（OverTime 員工離職率 30.5% vs 10.4%）
2. 首年與前三年是組織的留任關鍵窗口（LOWESS 曲線明顯下降拐點）
3. Sales Representative 與 Laboratory Technician 是兩個結構性高風險 role

## 資料集

- **來源**：[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
- **路徑**：`./Data/附件_資料分析資料集.csv`
- **性質**：IBM 資料科學家刻意合成的公開教學資料（非真實公司紀錄）
- **形狀**：1,470 筆員工 × 35 欄位，離職率 16.12%

## 專案結構

```
HR-Analysis/
├── CLAUDE.md                                # 專案規範（Claude Code 工作指引）
├── README.md                                # 本檔案
├── pyproject.toml                           # uv 套件定義
├── uv.lock                                  # 鎖定的依賴版本
├── .python-version                          # 3.12
├── .gitignore
├── Data/
│   └── 附件_資料分析資料集.csv               # IBM HR Attrition 原始資料
├── employee_attrition_analysis.ipynb        # 主要分析 notebook（11 個 section）
└── output/
    └── hr_attrition_export.csv              # Intervention Queue 匯出
```

## 環境建置與執行

### 前置需求

- Python 3.12
- [uv](https://github.com/astral-sh/uv)（極速 Python 套件管理器）

### 建立環境

```bash
# 從 repo 根目錄
uv sync
```

### 執行 notebook

```bash
# 方式 1：用 jupyter 打開（互動式）
uv run jupyter lab employee_attrition_analysis.ipynb

# 方式 2：全部執行並輸出結果（CI 適用）
uv run jupyter nbconvert --to notebook --execute \
    employee_attrition_analysis.ipynb \
    --output employee_attrition_analysis.ipynb
```

## Notebook 章節導覽

| Section | 主題 | 關鍵產出 |
|---|---|---|
| 0 | Setup & Imports | 繁體中文字型、THEME 色彩字典、helper functions |
| 1 | Business Context | 4 個管理問題、為什麼這份分析不同、分析限制 |
| 2 | Dataset Intake | Snapshot、schema、合成資料性質說明 |
| 3 | Data Quality Gate | 缺失／重複／常數欄位檢查 → Analytic Readiness Verdict |
| 4 | Executive KPIs | 6 cards 儀表板 |
| 5 | Diagnostic EDA | 數值分布、類別 × 離職率、LOWESS 曲線、Likert bar、相關矩陣（排除序數）、5 Key Patterns |
| 6 | Workforce Segmentation | Department / JobRole / CareerStage scorecard、OverTime × WorkLifeBalance 熱圖 |
| 7 | Feature Engineering | 新增 4 個 HR 複合指標、Feature Catalogue |
| 8 | Modelling | 3 候選模型、PR-AUC / ROC / Calibration 三合一圖、閾值搜尋、Lift Table |
| 9 | SHAP Explainability | LinearExplainer + TreeExplainer、summary / waterfall / dependence、Top 10 Drivers（含 HR_Interpretation）、SHAP vs Permutation Importance |
| 10 | Monitoring Layer | Segment Monitor、Fairness Monitor（含警語）、Tenure Monitor、Intervention Queue |
| 11 | Conclusions & Export | CSV 匯出、Executive Summary（3 structural + 2 modeling + 2 limitations + 1 next step） |

## 新增的 4 個 HR 複合特徵

| 特徵名稱 | 定義 | HR 意涵 |
|---|---|---|
| `ManagerTurnoverRatio` | `(YearsAtCompany − YearsWithCurrManager) / (YearsAtCompany + 1)` | 主管更換頻率、心理安全感代理 |
| `CompensationToLevelRatio` | `MonthlyIncome / (JobLevel × 1000 + 1)` | 薪資 vs 職級比、被提拔但未被獎勵訊號 |
| `PromotionStagnation` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | 升遷停滯比、職涯卡關訊號 |
| `JobHoppingRate` | `NumCompaniesWorked / (TotalWorkingYears + 1)` | 換工作頻率、外部流動傾向 |

所有指標均為**列內計算**（row-wise transformation），不依賴全體統計量，因此 train / test 切分後不存在 data leakage 風險。

## 主要套件

- `pandas`、`numpy`、`matplotlib`、`seaborn`、`statsmodels`（資料處理與視覺化）
- `scikit-learn`（LogisticRegression、CalibratedClassifierCV、Pipeline、ColumnTransformer）
- `xgboost`（樹模型對照組）
- `shap`（TreeExplainer / LinearExplainer）
- `jupyter`、`ipykernel`（notebook 執行環境）

## 技術取捨記錄

- **Calibration 方法**：選 Sigmoid 而非 Isotonic — 因為 n 小（訓練集僅 1,176 筆），Isotonic 容易過擬合
- **相關係數矩陣**：**排除序數變數**（Likert 1-5），因為 Pearson r 對 Likert 的「距離」解釋失真；序數變數與離職的關聯以獨立 bar chart 呈現
- **閾值選擇**：採用 F1 最大化作為 intervention cutoff；實務上還應加入 intervention cost × recall goal 的成本函數
- **Fairness check**：附警語提醒——風險分數差異可能來自歷史偏差／採樣偏誤／模型捷徑，不可自動調整閾值解決

## 授權

程式碼採用 MIT License；資料集屬 Kaggle 公開資料，遵循其原始授權。

---

作者：Tommy｜建置日期：2026-04-18
