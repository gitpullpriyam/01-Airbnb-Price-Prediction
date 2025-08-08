# 🏡 Airbnb Price Prediction (Seattle) — End-to-End Machine Learning with XGBoost

**Keywords:** Airbnb Price Prediction, XGBoost Regression, Predictive Modeling, Machine Learning, Data Science Portfolio, Kaggle Dataset, Feature Engineering, Hyperparameter Tuning, Regression, Python, pandas, scikit-learn

This project predicts Airbnb listing prices in Seattle using a complete **machine learning pipeline**:
**EDA → Data Cleaning → Feature Engineering → Baseline Models → XGBoost → Advanced Hyperparameter Tuning → Log Transformation**.

My goal here is to build a **real-world, reproducible model** that is:
- Accurate 📈
- Interpretable 🧠
- Well-documented 📚
- portfolio-friendly 💼

![Stars](https://img.shields.io/github/stars/gitpullpriyam/01-Airbnb-Price-Prediction?style=social)
![Forks](https://img.shields.io/github/forks/gitpullpriyam/01-Airbnb-Price-Prediction?style=social)

---

## 🔍 Problem Statement
Airbnb prices vary widely depending on location, room type, capacity, and reviews.  
My aim to **predict nightly prices** so that:
- Hosts can **price competitively**
- Guests can **understand price drivers**
- Platforms can **recommend optimal pricing**

---

## 🛠 Tech Stack
**Python**, **pandas**, **NumPy**, **scikit-learn**, **XGBoost**, **matplotlib**, **seaborn**, **Jupyter / VSCode**.

---

## 📁 Project Structure
```text
01-Airbnb-Price-Prediction/
├── data/                     # Raw CSV files (not tracked in git)
├── notebooks/                # 01 EDA → 05 Final log-XGB
├── outputs/                  # Cleaned datasets & evaluation results
├── images/                   # Saved plots for README & reports
├── requirements.txt          # Python package dependencies
├── README.md

---

## 📓 Notebook Workflow
1. **`01_eda_airbnb.ipynb`** — Data cleaning, missing value handling, price fixes, initial exploration  
2. **`02_model_baseline.ipynb`** — Linear Regression & Decision Tree baselines  
3. **`03_model_xgboost.ipynb`** — Baseline XGBoost  
4. **`04_advanced_xgboost_tuning.ipynb`** — RandomizedSearchCV (expanded hyperparameter grid)  
5. **`05_log_transformed_xgboost.ipynb`** — Final model (log-transform + tuning + charts) ✅

> Prefer a quick skim? Open **`05_log_transformed_xgboost.ipynb`**.

---

## 📊 Model Performance (Test Set)

| Model                      | RMSE   | MAE    | R²    |
|---------------------------|--------|--------|-------|
| Linear Regression         | 61.48  | 38.28  | 0.580 |
| Decision Tree             | 88.83  | 41.43  | 0.122 |
| Tuned Decision Tree       | 61.19  | 37.22  | 0.584 |
| XGBoost (baseline)        | 72.24  | 37.33  | 0.420 |
| Tuned XGBoost             | 60.27  | 34.45  | 0.596 |
| **Log-Transformed XGBoost** ✅ | **54.38** | **30.95** | **0.6711** |

> **Why performance model improved:** Log‑transforming `price` target variable reduces right‑skew and stabilizes variance, helping tree ensembles generalize better.

## 🧠 Key Insights
- ** Drivers of price: room type, capacity (bedrooms/accommodates), and neighborhood are most influential.

- ** Error behavior: residuals center near zero; the model slightly underestimates high‑end luxury listings.

- ** What moved the needle: log‑transforming the target improved RMSE from ~60 → 54 and R² from ~0.60 → 0.67.

- ** Adaptability: pipeline can be reused for other cities with minimal changes.

---

## 📈 Visualizations

**1️⃣ Actual vs Predicted Price**  
![Actual vs Predicted](images/actual_vs_predicted.png)

**2️⃣ Error Distribution**  
![Error Distribution](images/error_distribution.png)

**3️⃣ Top 20 Feature Importances (XGBoost)**  
![Feature Importance](images/feature_importance.png)

---
## GPT Generated for Code Reproducibility:
# Clone repo
git clone https://github.com/gitpullpriyam/01-Airbnb-Price-Prediction.git
cd 01-Airbnb-Price-Prediction

# Option A: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook

# Option B: conda
conda create -n airbnb-xgb python=3.10
conda activate airbnb-xgb
pip install -r requirements.txt
jupyter notebook

