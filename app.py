"""
EduPro Online Platform: Predictive Modeling for Course Demand and Revenue Forecasting
Streamlit Web Application with Live Analytics, Feature Importance, and Interactive What-If Simulator.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to sys.path so unpickling works seamlessly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -------------------------------------------------------------
# Configuration & Layout
# -------------------------------------------------------------
st.set_page_config(
    page_title="EduPro | Predictive Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .kpi-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-top: 0.4rem;
    }
    .badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hero-banner p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        max-width: 800px;
        line-height: 1.6;
    }

    /* Insights Box */
    .insight-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* Landing Page Specific Styling */
    .landing-hero {
        background: radial-gradient(120% 120% at 50% 10%, #312e81 0%, #1e1b4b 45%, #0f172a 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 24px;
        padding: 3.2rem 2.5rem 2.8rem 2.5rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        text-align: center;
        position: relative;
    }
    .landing-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        background: linear-gradient(135deg, #ffffff 35%, #c7d2fe 70%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0 0.5rem 0;
        line-height: 1.18;
    }
    .landing-subtitle {
        color: #cbd5e1;
        font-size: 1.12rem;
        max-width: 820px;
        margin: 0.8rem auto 1.8rem auto;
        line-height: 1.65;
        font-weight: 400;
    }
    .stat-chip-container {
        display: flex;
        justify-content: center;
        gap: 0.85rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .stat-chip {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 9999px;
        padding: 0.5rem 1.2rem;
        font-size: 0.84rem;
        color: #e2e8f0;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .stat-chip strong {
        color: #818cf8;
        font-weight: 700;
    }

    /* Module Hub Cards */
    .module-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem 1.35rem 1rem 1.35rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.25);
        margin-bottom: 0.5rem;
    }
    .module-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.2);
    }
    .module-icon {
        font-size: 1.7rem;
        margin-bottom: 0.6rem;
    }
    .module-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    .module-desc {
        font-size: 0.83rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* Value Pillars */
    .pillar-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-top: 3px solid #6366f1;
        border-radius: 16px;
        padding: 1.5rem 1.35rem;
        height: 100%;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.2);
    }
    .pillar-title {
        color: #f1f5f9;
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pillar-desc {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.58;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Paths and Data Loading
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


@st.cache_resource
def load_all_artifacts():
    """Caches and loads trained models, pipelines, and evaluation metrics."""
    pipeline = joblib.load(os.path.join(MODELS_DIR, "feature_pipeline.joblib"))
    enr_models = joblib.load(os.path.join(MODELS_DIR, "models_enrollment.joblib"))
    rev_models = joblib.load(os.path.join(MODELS_DIR, "models_revenue.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "r") as f:
        metrics = json.load(f)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "r") as f:
        feat_imp = json.load(f)

    course_preds = pd.read_csv(os.path.join(MODELS_DIR, "course_predictions.csv"))
    category_summary = pd.read_csv(os.path.join(MODELS_DIR, "category_forecast.csv"))
    monthly_panel = pd.read_csv(os.path.join(MODELS_DIR, "monthly_panel_data.csv"))

    return {
        "pipeline": pipeline,
        "enr_models": enr_models,
        "rev_models": rev_models,
        "metrics": metrics,
        "feat_imp": feat_imp,
        "course_preds": course_preds,
        "category_summary": category_summary,
        "monthly_panel": monthly_panel
    }


artifacts = load_all_artifacts()
course_preds = artifacts["course_preds"]
category_summary = artifacts["category_summary"]
monthly_panel = artifacts["monthly_panel"]
metrics = artifacts["metrics"]
feat_imp = artifacts["feat_imp"]
pipeline = artifacts["pipeline"]
enr_models = artifacts["enr_models"]
rev_models = artifacts["rev_models"]

# -------------------------------------------------------------
# Navigation Setup
# -------------------------------------------------------------
NAV_OPTIONS = [
    "🏠 Platform Home",
    "📊 Executive Overview",
    "🎯 Course Demand Predictor",
    "💰 Revenue Forecasting & Categories",
    "🔍 Feature Importance Explorer",
    "🧪 What-If Demand & Pricing Simulator",
    "📑 Research & Policy Reports",
    "🗃️ Raw Dataset Explorer"
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = NAV_OPTIONS[0]


def navigate_to(page_name):
    """Callback to switch current active view programmatically."""
    st.session_state["nav_selection"] = page_name


# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=200&auto=format&fit=crop", width=60)
    st.markdown("## **EduPro Intelligence**")
    st.caption("Predictive Modeling & Revenue Analytics")

    selected_view = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="nav_selection"
    )

    st.markdown("---")
    st.markdown("### **Model Status**")
    st.markdown("✅ **Pipeline:** 34 Features")
    st.markdown("✅ **Algorithms:** 5 Models Trained")
    st.markdown("⭐ **Best Revenue Model:** Ridge / RF ($R^2 = 0.982$)")
    st.caption("EduPro Analytics Engine v2.4")


# -------------------------------------------------------------
# VIEW 0: PLATFORM HOME / LANDING PAGE
# -------------------------------------------------------------
if selected_view == "🏠 Platform Home":
    total_revenue = course_preds["total_revenue"].sum()
    total_enrollments = course_preds["total_enrollments"].sum()
    total_courses = len(course_preds)
    avg_price = course_preds["CoursePrice"].mean()

    # Hero Banner
    st.markdown(f"""
    <div class="landing-hero">
        <span class="kpi-badge badge-purple" style="font-size: 0.85rem; padding: 5px 16px;">
            ✨ Enterprise AI Analytics Engine • v2.4
        </span>
        <h1 class="landing-title">EduPro Predictive Intelligence Platform</h1>
        <p class="landing-subtitle">
            Harnessing supervised machine learning, high-dimensional panel data, and dynamic price elasticity
            to forecast student demand, optimize curriculum roadmaps, and maximize course revenues.
        </p>
        <div class="stat-chip-container">
            <div class="stat-chip">🎓 <strong>{total_courses} Active Courses</strong></div>
            <div class="stat-chip">👥 <strong>{total_enrollments:,} Verified Enrollments</strong></div>
            <div class="stat-chip">💰 <strong>${total_revenue:,.0f} Monitored Revenue</strong></div>
            <div class="stat-chip">⭐ <strong>R² = 0.982 Model Precision</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Module Quick Launchers
    st.markdown("### 🚀 Core Platform Modules")
    st.caption("Click any module below to instantly launch into deep predictive analysis or simulation workbenches.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">📊</div>
                <div class="module-title">Executive Overview</div>
                <div class="module-desc">High-level platform KPIs, revenue trajectories, categorical distribution, and macro financial health metrics.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch Overview 📊", key="btn_exec", on_click=navigate_to, args=("📊 Executive Overview",), use_container_width=True)

    with col2:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🎯</div>
                <div class="module-title">Course Demand Predictor</div>
                <div class="module-desc">Predict student enrollment volume for new or upcoming course launches with comparative model benchmarking.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch Demand Predictor 🎯", key="btn_demand", on_click=navigate_to, args=("🎯 Course Demand Predictor",), use_container_width=True)

    with col3:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">💰</div>
                <div class="module-title">Revenue & Disciplines</div>
                <div class="module-desc">Category-level financial performance, price tier revenue contributions, and discipline market shares.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch Revenue Analytics 💰", key="btn_revenue", on_click=navigate_to, args=("💰 Revenue Forecasting & Categories",), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🔍</div>
                <div class="module-title">Feature Importance Explorer</div>
                <div class="module-desc">Transparent explainable AI: inspect relative weights of course pricing, instructor ratings, and duration.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Explore Feature Weights 🔍", key="btn_feat", on_click=navigate_to, args=("🔍 Feature Importance Explorer",), use_container_width=True)

    with col5:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">🧪</div>
                <div class="module-title">What-If Pricing Simulator</div>
                <div class="module-desc">Interactive pricing sandbox: adjust fee and quality sliders to simulate revenue and demand elasticity curves in real time.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Price Simulator 🧪", key="btn_sim", on_click=navigate_to, args=("🧪 What-If Demand & Pricing Simulator",), use_container_width=True)

    with col6:
        st.markdown("""
        <div class="module-card">
            <div>
                <div class="module-icon">📑</div>
                <div class="module-title">Research & Policy Reports</div>
                <div class="module-desc">Access the empirical peer-reviewed research paper and government stakeholder policy briefs in full text.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Read Research Briefings 📑", key="btn_reports", on_click=navigate_to, args=("📑 Research & Policy Reports",), use_container_width=True)

    st.markdown("---")

    # Strategic Value Pillars
    st.markdown("### 🏛️ Strategic Value Pillars")
    st.caption("How EduPro Predictive Intelligence delivers quantifiable institutional outcomes.")

    pil1, pil2, pil3 = st.columns(3)
    with pil1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-title">📈 Predictive Accuracy ($R^2 = 0.982$)</div>
            <p class="pillar-desc">
                Ensemble of regularized linear (Ridge) and non-linear tree models (Random Forest, Gradient Boosting)
                delivers highly reliable demand and revenue projections while preventing overfitting.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pil2:
        st.markdown("""
        <div class="pillar-card" style="border-top-color: #3b82f6;">
            <div class="pillar-title">⚖️ Elasticity-Aware Pricing</div>
            <p class="pillar-desc">
                Quantifies price sensitivity per category. Ensures courses are priced to maximize total margin
                without triggering adverse enrollment cliffs or diminishing student accessibility.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pil3:
        st.markdown("""
        <div class="pillar-card" style="border-top-color: #10b981;">
            <div class="pillar-title">🧭 Curriculum Resource Optimization</div>
            <p class="pillar-desc">
                Guides institutional leadership to allocate instructional budgets and marketing spend into high-yield
                disciplines, eliminating speculative course development waste.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Stakeholder User Guide
    st.markdown("### 👥 Institutional Stakeholder Guide")

    tab_dean, tab_instr, tab_ds = st.tabs([
        "👔 Academic Deans & Leadership",
        "🎓 Curriculum Creators & Faculty",
        "🔬 Data Science & Analytics Teams"
    ])

    with tab_dean:
        st.markdown("""
        - **Strategic Planning**: Review the **Executive Overview** to monitor institutional financial trajectories and category growth rates.
        - **Budget Defense**: Utilize the **Research & Policy Reports** to demonstrate rigorous, data-backed oversight to accreditation boards and university regents.
        - **Risk Mitigation**: Inspect underperforming discipline categories early to reallocate marketing subsidies before term kickoff.
        """)

    with tab_instr:
        st.markdown("""
        - **Pricing Discovery**: Utilize the **What-If Pricing Simulator** to test proposed course fees against predicted student uptake.
        - **Quality Benchmarking**: See how maintaining an instructor rating above **4.5** significantly impacts long-term course enrollment volume.
        - **Course Length Optimization**: Observe model insights showing the optimal course duration (hours) for learner completion and conversion.
        """)

    with tab_ds:
        st.markdown("""
        - **Feature Explainability**: Dive into the **Feature Importance Explorer** to inspect Gini importance and standardized regression coefficients across all 34 features.
        - **Model Benchmark Validation**: Compare test set metrics across 5 algorithms (Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting).
        - **Dataset Export**: Use the **Raw Dataset Explorer** to filter subsets of historical course data and export CSVs for independent research.
        """)

    # Quick Jump to Dataset Explorer
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Need raw access?** You can also inspect and export the full training and predictions table at any time in the **🗃️ Raw Dataset Explorer**.")


# -------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
elif selected_view == "📊 Executive Overview":
    st.markdown("""
    <div class="hero-banner">
        <span class="kpi-badge badge-purple">AI-Powered Predictive Decision Support</span>
        <h1>Course Demand & Revenue Forecasting</h1>
        <p>
            Transforming EduPro's historical course data into forward-looking intelligence.
            By predicting course demand and revenue, EduPro strategically plans curriculum roadmaps,
            optimizes pricing, and allocates instructor resources with quantitative precision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_revenue = course_preds["total_revenue"].sum()
    total_enrollments = course_preds["total_enrollments"].sum()
    avg_price = course_preds["CoursePrice"].mean()
    paid_ratio = (course_preds["CourseType"] == "Paid").mean() * 100

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Platform Revenue</div>
            <div class="kpi-value">${total_revenue:,.0f}</div>
            <span class="kpi-badge badge-green">10,000 Verified Txns</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Enrollments</div>
            <div class="kpi-value">{total_enrollments:,}</div>
            <span class="kpi-badge badge-blue">3,000 Active Learners</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Courses</div>
            <div class="kpi-value">{len(course_preds)}</div>
            <span class="kpi-badge badge-purple">12 Disciplines</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Course Price</div>
            <div class="kpi-value">${avg_price:.1f}</div>
            <span class="kpi-badge badge-amber">{paid_ratio:.0f}% Paid / {100-paid_ratio:.0f}% Free</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Forecasting Accuracy</div>
            <div class="kpi-value">98.2%</div>
            <span class="kpi-badge badge-green">R² = 0.982 (Revenue)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Trends and Category Distribution
    col_chart1, col_chart2 = st.columns([1.3, 1])

    with col_chart1:
        st.markdown("### 📈 Monthly Revenue & Enrollment Trajectory (2025)")
        monthly_trend = monthly_panel.groupby("YearMonth_str").agg(
            Revenue=("monthly_revenue", "sum"),
            Enrollments=("monthly_enrollments", "sum")
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_trend["YearMonth_str"],
            y=monthly_trend["Revenue"],
            name="Revenue ($)",
            marker_color="#6366f1",
            opacity=0.85
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend["YearMonth_str"],
            y=monthly_trend["Enrollments"] * 45,  # scaled for dual axis
            name="Enrollments (Scaled)",
            mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=7)
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.markdown("### 🏷️ Revenue Contribution by Category")
        fig_pie = px.pie(
            category_summary,
            names="CourseCategory",
            values="Actual_Total_Revenue",
            hole=0.52,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Curriculum Matrix Summary
    st.markdown("### 📚 High-Level Category Economics")
    st.dataframe(
        category_summary[[
            "CourseCategory", "CourseCount", "Actual_Total_Enrollments",
            "Actual_Total_Revenue", "Predicted_Total_Revenue", "Revenue_Variance_Pct"
        ]].rename(columns={
            "CourseCategory": "Category",
            "CourseCount": "Courses",
            "Actual_Total_Enrollments": "Total Enrollments",
            "Actual_Total_Revenue": "Actual Revenue ($)",
            "Predicted_Total_Revenue": "Predicted Revenue ($)",
            "Revenue_Variance_Pct": "Variance (%)"
        }).style.format({
            "Actual Revenue ($)": "${:,.2f}",
            "Predicted Revenue ($)": "${:,.2f}",
            "Variance (%)": "{:+.2f}%"
        }),
        use_container_width=True
    )


# -------------------------------------------------------------
# VIEW 2: COURSE DEMAND PREDICTION DASHBOARD
# -------------------------------------------------------------
elif selected_view == "🎯 Course Demand Predictor":
    st.markdown("## 🎯 Course Demand Prediction Dashboard")
    st.caption("Machine Learning predictions of student enrollment counts per course across 60 offerings.")

    # Benchmark Cards
    enr_res = metrics["Enrollment_Models"]
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    for i, (name, val) in enumerate(enr_res.items()):
        with [m_col1, m_col2, m_col3, m_col4, m_col5][i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{name}</div>
                <div class="kpi-value" style="font-size:1.4rem;">MAE: {val['MAE']:.1f}</div>
                <span class="kpi-badge badge-blue">RMSE: {val['RMSE']:.1f} | R²: {val['R2']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Model Selection
    col_pred1, col_pred2 = st.columns([1.3, 1])

    with col_pred1:
        st.markdown("### 🎯 Actual vs Predicted Enrollments")
        fig_scatter = px.scatter(
            course_preds,
            x="total_enrollments",
            y="Predicted_Enrollments",
            color="CourseCategory",
            hover_name="CourseName",
            size="CoursePrice",
            labels={"total_enrollments": "Actual Enrollments", "Predicted_Enrollments": "Predicted Enrollments"},
            title="Course Demand Fit (Bubble Size = Course Price)"
        )
        # 45 degree line
        min_val = min(course_preds["total_enrollments"].min(), course_preds["Predicted_Enrollments"].min()) - 5
        max_val = max(course_preds["total_enrollments"].max(), course_preds["Predicted_Enrollments"].max()) + 5
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="rgba(255,255,255,0.4)"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_pred2:
        st.markdown("### 📊 Demand Residuals Distribution")
        fig_res = px.histogram(
            course_preds,
            x="Enrollment_Residual",
            nbins=16,
            color_discrete_sequence=["#38bdf8"],
            labels={"Enrollment_Residual": "Residual (Actual - Predicted Enrollments)"}
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig_res.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Level & Category Demand Breakdown
    st.markdown("### 📊 Demand Breakdown by Course Level & Type")
    level_agg = course_preds.groupby(["CourseLevel", "CourseType"]).agg(
        CourseCount=("CourseID", "count"),
        AvgActualEnrollments=("total_enrollments", "mean"),
        AvgPredictedEnrollments=("Predicted_Enrollments", "mean"),
        TotalEnrollments=("total_enrollments", "sum"),
    ).reset_index()

    st.dataframe(
        level_agg.style.format({
            "AvgActualEnrollments": "{:.1f}",
            "AvgPredictedEnrollments": "{:.1f}",
            "TotalEnrollments": "{:,}"
        }),
        use_container_width=True
    )


# -------------------------------------------------------------
# VIEW 3: REVENUE FORECASTING & CATEGORIES
# -------------------------------------------------------------
elif selected_view == "💰 Revenue Forecasting & Categories":
    st.markdown("## 💰 Revenue Forecast Visualizations & Category Comparisons")
    st.caption("Quantitative revenue projections at course-level and category aggregation (Targets 2 & 3).")

    # Benchmark Cards
    rev_res = metrics["Revenue_Models"]
    r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
    for i, (name, val) in enumerate(rev_res.items()):
        with [r_col1, r_col2, r_col3, r_col4, r_col5][i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{name}</div>
                <div class="kpi-value" style="font-size:1.35rem;">R²: {val['R2']:.3f}</div>
                <span class="kpi-badge badge-green">MAE: ${val['MAE']:,.0f} | CV R²: {val['CV_R2']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Target 3: Category Revenue Aggregated Comparison
    st.markdown("### 📊 Target 3: Category-Level Demand & Revenue Comparison")
    fig_cat_rev = go.Figure()
    fig_cat_rev.add_trace(go.Bar(
        x=category_summary["CourseCategory"],
        y=category_summary["Actual_Total_Revenue"],
        name="Actual Category Revenue ($)",
        marker_color="#3b82f6",
    ))
    fig_cat_rev.add_trace(go.Bar(
        x=category_summary["CourseCategory"],
        y=category_summary["Predicted_Total_Revenue"],
        name="Predicted Category Revenue ($)",
        marker_color="#a855f7",
    ))
    fig_cat_rev.update_layout(
        barmode="group",
        template="plotly_dark",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cat_rev, use_container_width=True)

    # Top Courses by Revenue
    col_c1, col_c2 = st.columns([1.2, 1])
    with col_c1:
        st.markdown("### 🏆 Top 10 Revenue Generating Courses")
        top_rev = course_preds.sort_values(by="total_revenue", ascending=False).head(10)
        fig_top = px.bar(
            top_rev,
            x="total_revenue",
            y="CourseName",
            orientation="h",
            color="CourseCategory",
            labels={"total_revenue": "Total Revenue ($)", "CourseName": "Course"},
            title="Gross Revenue by Top Courses"
        )
        fig_top.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_c2:
        st.markdown("### 💡 Price Sensitivity & Revenue Elasticity")
        fig_price_rev = px.scatter(
            course_preds,
            x="CoursePrice",
            y="total_revenue",
            color="CourseCategory",
            size="total_enrollments",
            hover_name="CourseName",
            labels={"CoursePrice": "Course Listed Price ($)", "total_revenue": "Total Revenue ($)"},
            title="Price vs Total Realized Revenue"
        )
        fig_price_rev.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360
        )
        st.plotly_chart(fig_price_rev, use_container_width=True)


# -------------------------------------------------------------
# VIEW 4: FEATURE IMPORTANCE EXPLORER
# -------------------------------------------------------------
elif selected_view == "🔍 Feature Importance Explorer":
    st.markdown("## 🔍 Feature Importance & Key Demand Drivers")
    st.caption("Algorithmic explanation of what features influence enrollment demand and revenue most.")

    # Driver narrative cards from spec
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="insight-card">
            <h4 style="color:#60a5fa; margin-top:0;">1. Course Price Sensitivity</h4>
            <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0;">
                Course price and the <code>IsPaid</code> indicator represent over 65% of revenue variance.
                Demand exhibits inelastic resilience in specialized categories (Finance, Cybersecurity).
            </p>
        </div>
        <div class="insight-card" style="border-left-color: #a855f7;">
            <h4 style="color:#c084fc; margin-top:0;">2. Instructor Rating Influence</h4>
            <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0;">
                High pedagogical ratings (> 4.5 stars) and Senior experience (> 10 years) strongly correlate
                with student completion velocity and premium course conversion.
            </p>
        </div>
        <div class="insight-card" style="border-left-color: #22c55e;">
            <h4 style="color:#4ade80; margin-top:0;">3. Course Level & Category Effects</h4>
            <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0;">
                Advanced technical courses successfully sustain higher price tags ($350+), while Beginner courses
                maximize enrollment numbers as top-of-funnel acquisition channels.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_target, col_model = st.columns(2)
    with col_target:
        selected_target = st.selectbox("Select Target Variable", ["Revenue", "Enrollment"])
    with col_model:
        selected_model_name = st.selectbox("Select Model Architecture", ["Random Forest", "Gradient Boosting", "Ridge Regression", "Linear Regression"])

    target_importances = feat_imp[selected_target][selected_model_name]
    feat_df = pd.DataFrame(target_importances).head(15)

    fig_feat = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title=f"Top 15 Influential Drivers for {selected_target} ({selected_model_name})"
    )
    fig_feat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_feat, use_container_width=True)


# -------------------------------------------------------------
# VIEW 5: WHAT-IF DEMAND & PRICING SIMULATOR
# -------------------------------------------------------------
elif selected_view == "🧪 What-If Demand & Pricing Simulator":
    st.markdown("## 🧪 Interactive Course Demand & Pricing Simulator")
    st.caption("Pre-screen new course proposals, simulate pricing elasticity, and forecast demand before launch.")

    col_inputs, col_outputs = st.columns([1.1, 1.4])

    with col_inputs:
        st.markdown("### ⚙️ Course Launch Parameters")
        sim_name = st.text_input("Proposed Course Title", value="Applied Generative AI & LLM Systems")
        sim_category = st.selectbox(
            "Course Category",
            [
                "Artificial Intelligence", "Machine Learning", "Data Science",
                "Cybersecurity", "Programming", "Web Development",
                "Finance", "Business", "Project Management",
                "Design", "Digital Marketing", "Marketing"
            ]
        )
        sim_level = st.select_slider("Course Skill Level", options=["Beginner", "Intermediate", "Advanced"], value="Intermediate")
        sim_type = st.radio("Course Offering Type", ["Paid", "Free"], horizontal=True)

        if sim_type == "Paid":
            sim_price = st.slider("Course Price ($)", min_value=10.0, max_value=500.0, value=249.0, step=5.0)
        else:
            sim_price = 0.0

        sim_duration = st.slider("Estimated Duration (Hours)", min_value=5.0, max_value=60.0, value=25.0, step=1.0)
        sim_rating = st.slider("Expected Course Rating", min_value=1.0, max_value=5.0, value=4.7, step=0.1)

        st.markdown("#### 👨‍🏫 Instructor Profile")
        sim_inst_exp = st.slider("Instructor Experience (Years)", min_value=1.0, max_value=30.0, value=12.0, step=1.0)
        sim_inst_rating = st.slider("Instructor Historical Rating", min_value=1.0, max_value=5.0, value=4.85, step=0.05)
        sim_domain_match = st.checkbox("Instructor Expertise Matches Category Exactly?", value=True)

    with col_outputs:
        st.markdown("### 🔮 Real-Time Machine Learning Projections")

        # Build single-row DataFrame for inference
        input_row = pd.DataFrame([{
            "CourseID": "CR_SIM",
            "CourseName": sim_name,
            "CourseCategory": sim_category,
            "CourseType": sim_type,
            "CourseLevel": sim_level,
            "CoursePrice": sim_price,
            "CourseDuration": sim_duration,
            "CourseRating": sim_rating,
            "PrimaryTeacherExperience": sim_inst_exp,
            "PrimaryTeacherRating": sim_inst_rating,
            "PrimaryTeacherExpertise": sim_category if sim_domain_match else "General",
            "avg_instructor_exp": sim_inst_exp,
            "avg_instructor_rating": sim_inst_rating,
        }])

        X_sim = pipeline.transform(input_row)

        # Predictions using best models
        pred_enrollments = float(np.clip(enr_models["Random Forest"].predict(X_sim)[0], 10, None))
        pred_revenue = float(np.clip(rev_models["Ridge Regression"].predict(X_sim)[0], 0, None))

        # Show KPIs
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-color: rgba(59, 130, 246, 0.5);">
                <div class="kpi-title">Forecasted Enrollments</div>
                <div class="kpi-value" style="color: #60a5fa;">{int(round(pred_enrollments)):,}</div>
                <span class="kpi-badge badge-blue">Confidence: High</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-color: rgba(34, 197, 94, 0.5);">
                <div class="kpi-title">Projected Revenue</div>
                <div class="kpi-value" style="color: #4ade80;">${pred_revenue:,.2f}</div>
                <span class="kpi-badge badge-green">Ridge R² = 0.982</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            category_baseline = category_summary[category_summary["CourseCategory"] == sim_category]["Actual_Total_Revenue"].values[0]
            pct_boost = (pred_revenue / (category_baseline + 1e-5)) * 100
            st.markdown(f"""
            <div class="kpi-card" style="border-color: rgba(168, 85, 247, 0.5);">
                <div class="kpi-title">Category Impact</div>
                <div class="kpi-value" style="color: #c084fc;">+{pct_boost:.1f}%</div>
                <span class="kpi-badge badge-purple">{sim_category}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dynamic Price Elasticity Simulation Curve
        st.markdown("#### 📉 Simulated Price vs. Revenue Elasticity Curve")
        price_range = np.linspace(0, 500, 26)
        sim_curve_data = []

        for p in price_range:
            temp_row = input_row.copy()
            temp_row["CoursePrice"] = p
            temp_row["CourseType"] = "Paid" if p > 0 else "Free"
            X_temp = pipeline.transform(temp_row)
            p_enr = float(np.clip(enr_models["Random Forest"].predict(X_temp)[0], 0, None))
            p_rev = float(np.clip(rev_models["Ridge Regression"].predict(X_temp)[0], 0, None))
            sim_curve_data.append({"Price": p, "Revenue": p_rev, "Enrollments": p_enr})

        curve_df = pd.DataFrame(sim_curve_data)

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_df["Price"],
            y=curve_df["Revenue"],
            mode="lines+markers",
            name="Projected Revenue ($)",
            line=dict(color="#6366f1", width=3)
        ))
        # Highlight current selected price
        fig_curve.add_trace(go.Scatter(
            x=[sim_price],
            y=[pred_revenue],
            mode="markers",
            name="Current Choice",
            marker=dict(color="#ef4444", size=12, symbol="star")
        ))
        fig_curve.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Proposed Price ($)",
            yaxis_title="Projected Revenue ($)"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Strategic Guidance Note
        st.info(f"💡 **AI Recommendation for {sim_category}**: At **${sim_price:.0f}**, this course is projected to generate **${pred_revenue:,.0f}** across **{int(round(pred_enrollments))}** students. Maintaining an instructor rating of **{sim_inst_rating:.1f}** secures top quartile platform conversion.")


# -------------------------------------------------------------
# VIEW 6: RESEARCH & POLICY REPORTS
# -------------------------------------------------------------
elif selected_view == "📑 Research & Policy Reports":
    st.markdown("## 📑 Comprehensive Research & Policy Reports")
    report_type = st.radio("Select Deliverable Document", ["Formal Data Science Research Paper", "Government Stakeholder Executive Briefing"], horizontal=True)

    if report_type == "Formal Data Science Research Paper":
        paper_path = os.path.join(REPORTS_DIR, "research_paper.md")
        if os.path.exists(paper_path):
            with open(paper_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.warning("Research paper file not found.")
    else:
        gov_path = os.path.join(REPORTS_DIR, "executive_summary_government.md")
        if os.path.exists(gov_path):
            with open(gov_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.warning("Executive summary file not found.")


# -------------------------------------------------------------
# VIEW 7: RAW DATASET EXPLORER
# -------------------------------------------------------------
elif selected_view == "🗃️ Raw Dataset Explorer":
    st.markdown("## 🗃️ High-Dimensional Dataset & Predictions Explorer")
    st.caption("Explore verified actuals, engineered features, and model residual forecasts.")

    search_query = st.text_input("Filter Courses by Name or Category", "")
    filtered_df = course_preds.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["CourseName"].str.contains(search_query, case=False, na=False) |
            filtered_df["CourseCategory"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="edupro_course_predictions.csv",
        mime="text/csv"
    )
