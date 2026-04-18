# HR Attrition Analytics — 專案規範

## 專案目標
以 IBM HR Analytics Attrition Dataset 為基礎，建立改良版的
HR 離職風險分析專案，作為 People Analytics / Data Science 求職作品集。

## 資料集
來源：https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
路徑：./Data/附件_資料分析資料集.csv

## 輸出格式
Jupyter Notebook（.ipynb），每個 section 為獨立 cell group。
圖表一律使用 matplotlib / seaborn，禁止使用 plotly。

## 雙讀者設計原則
每個 section 同時服務兩種讀者：
- Data Science reviewer：需要技術嚴謹性、正當的建模選擇、完整評估框架
- People Analytics / HR reviewer：需要業務框架、HR 可讀的指標、可行動的輸出

## 四大改良重點
1. SHAP 可解釋性（TreeExplainer + summary / waterfall / dependence plots）
2. 更豐富的 feature engineering（至少 2 個新複合 HR 指標）
3. 更嚴謹的模型評估（PR-AUC 為主要指標，附明確理由說明）
4. Monitoring layer 提供具體 HR 介入建議文字，不只是風險標籤

## 分析流程與各 Section 規範

### Section 0 — Setup & Imports
- 建立 THEME 色彩字典、helper functions、matplotlib rcParams
- 確認環境後印出 "Environment ready"

### Section 1 — Business Context
- 純 markdown，定義 4 個管理問題
- 加入「為什麼這份分析不同」段落（說明 PR-AUC 選擇理由、SHAP 用途）

### Section 2 — Dataset Intake
- 顯示 snapshot、schema、前 8 筆資料
- 加入 markdown 說明資料集的合成性質與因果推論限制

### Section 3 — Data Quality Gate
- 檢查缺失值、重複列、常數欄位
- 輸出「Analytic readiness verdict」段落

### Section 4 — Executive KPIs
- 6 個 KPI cards，兩排各 3 個
- 每個 card 附一行業務詮釋文字

### Section 5 — Diagnostic EDA
DS layer: 統計分布、交叉分析、LOWESS 趨勢線
HR layer: 每張圖上方有 markdown（說明圖表目的 + HR 意涵），
          section 末尾有「5 key patterns」總結（用 HR 語言撰寫）

### Section 6 — Workforce Segmentation
- Department / JobRole / CareerStage 三個 scorecard
- OverTime × LowWorkLifeBalance 熱圖

### Section 7 — Feature Engineering（改良版）
- 完整重現 baseline 所有 feature
- 新增 ManagerTenureRisk 和 CompensationGrowthGap（或自訂 2 個）
- 每個新 feature 附 markdown 說明 HR 設計邏輯
- 輸出 Feature Catalogue：Feature | Type | Is_New | Purpose

### Section 8 — Modelling（改良版）
DS layer: PR-AUC 為主要指標（附說明）、threshold 選擇邏輯、
          ECE 校準品質、Lift Table
HR layer: 模型選擇理由以 HR director 語言撰寫

候選模型：Sparse Logistic、Sparse Logistic + Calibration、XGBoost + Calibration
套件查詢：使用 context7 確認 shap、xgboost、sklearn 最新 API

### Section 9 — SHAP Explainability（新增）
DS layer: TreeExplainer / LinearExplainer、summary plot、waterfall plot、
          dependence plot
HR layer: Top 10 drivers 表格附 HR_Interpretation 欄位，
          SHAP vs permutation importance 比較說明

### Section 10 — Monitoring Layer（改良版）
- Segment Monitor、Fairness Monitor（附公平性警語）、Tenure Monitor
- Intervention Queue 新增欄位：PrimarySignal、SuggestedAction、ReviewCadence
- Watchlist 新增 PrimaryRiskFactor（個別員工的 top SHAP feature）

### Section 11 — Conclusions & Export
- 輸出至 ./output/hr_attrition_export.csv
- Executive Summary：3 structural findings + 2 modeling findings +
  2 analytic limitations + 1 recommended next step

## 套件管理
使用 uv 建立虛擬環境。主要套件：
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
（不需要 imbalanced-learn，本專案不使用重採樣策略）

## 版本控制
完成後 push 至 GitHub public repo，repo name 以專案資料夾名稱為準。
需撰寫標準格式 README.md。