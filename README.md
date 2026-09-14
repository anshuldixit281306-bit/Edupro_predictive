# EduPro Online Platform: Course Demand & Revenue Forecasting

An end-to-end Machine Learning and Business Intelligence solution for course demand forecasting, revenue prediction, and interactive decision support.

---

## 🚀 Quick Start (One-Click Launch)

### Option 1: Double Click
Simply double-click **`run_app.bat`** in this folder!

### Option 2: Terminal Command
Open PowerShell or Command Prompt in this folder and run:
```bash
python -m streamlit run app.py
```
Then open **`http://localhost:8501`** in your browser.

---

## 📁 Folder Structure

```
EduPro_Predictive_Platform/
├── app.py                          # Streamlit web application & interactive simulator
├── run_app.bat                     # Windows one-click launch batch script
├── requirements.txt                # Python package dependencies
├── README.md                       # Project documentation
│
├── data/
│   └── EduPro Online Platform.xlsx # Enterprise database (Users, Teachers, Courses, Txns)
│
├── models/                         # Serialized ML models and prediction outputs
│   ├── category_forecast.csv       # Target 3: Category revenue aggregates & variances
│   ├── course_predictions.csv      # Target 1 & 2: Course demand & revenue predictions
│   ├── feature_importance.json     # Feature importances & linear coefficients
│   ├── feature_pipeline.joblib     # Preprocessing pipeline (OneHotEncoder + Scaler)
│   ├── metrics_summary.json        # Evaluation scores (MAE, RMSE, R2, 5-fold CV)
│   ├── models_enrollment.joblib    # 5 fitted enrollment regression models
│   ├── models_revenue.joblib       # 5 fitted course revenue regression models
│   └── monthly_models.joblib       # Time series panel models
│
├── reports/                        # Deliverables and submission papers
│   ├── research_paper.md           # Formal Data Science & ML Research Paper
│   └── executive_summary_government.md # Executive briefing for public & government stakeholders
│
└── src/                            # Core Machine Learning source code
    ├── data_loader.py              # Ingestion, validation, and relational joining
    ├── feature_engineering.py      # Feature buckets, match scores, and transforms
    └── train_models.py             # Model training, cross-validation, and artifact export
```

---

## 🎯 Predictive Targets & Performance

| Target | Primary Algorithm | Performance Metric | Practical Interpretation |
| :--- | :--- | :--- | :--- |
| **1. Course Enrollment Count** | Random Forest / Linear | MAE: 12.4 students | **< 7.4% MAPE** on courses averaging 166.7 enrollments. |
| **2. Course Revenue** | Ridge Regression | **R²: 0.982**, RMSE: $2,990 | Explains **98.2% of variance** across $0 - $85K course revenues. |
| **3. Category Revenue** | Aggregated Ensemble | Variance: ±0.04% in top domains | High macroeconomic fidelity across all 12 disciplines. |

---

## 🖥️ Streamlit Web App Features

1. **Executive Overview Dashboard**: High-level platform KPIs, 2025 monthly revenue timeline, and disciplinary category breakdown.
2. **Course Demand Predictor**: Actual vs predicted enrollments with price-weighted bubbles and error residual distributions.
3. **Revenue Forecast Visualizations**: Course and Category comparisons, top revenue generators, and price elasticity curves.
4. **Feature Importance Explorer**: Top 15 influential drivers across all models (course price, instructor rating, experience, domain match).
5. **Interactive What-If Demand & Pricing Simulator**: Live simulation tool allowing stakeholders to input course price, duration, level, instructor experience/rating, and category to immediately view forecasted demand, revenue, and sensitivity curves.
6. **Reports & Dataset Viewer**: Direct in-app viewer for the formal research paper and government policy briefing, with CSV export.
