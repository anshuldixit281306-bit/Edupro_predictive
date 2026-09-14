"""
Model Training and Evaluation Pipeline for EduPro Online Platform.
Trains:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluates on targets:
1. Enrollment Count
2. Course Revenue
3. Category Revenue (aggregated)

Outputs MAE, RMSE, R2, feature importances, and saved joblib artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_raw_data, build_course_level_dataset, build_monthly_panel_dataset
from feature_engineering import EduProFeaturePipeline, engineer_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_predictions(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def train_and_evaluate_target(X_train, X_test, y_train, y_test, feature_names, target_name="Enrollment"):
    """
    Trains all candidate models on train set, evaluates on test set,
    and performs 5-fold cross validation.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=np.logspace(-3, 3, 50)),
        "Lasso Regression": LassoCV(alphas=np.logspace(-3, 3, 50), max_iter=5000, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42),
    }

    results = {}
    fitted_models = {}
    feature_importances = {}
    predictions = {"Actual": y_test.tolist()}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Test predictions
        y_pred = model.predict(X_test)
        # Prevent negative predictions for demand/revenue
        y_pred = np.clip(y_pred, 0, None)
        predictions[name] = y_pred.tolist()

        metrics = evaluate_predictions(y_test, y_pred)

        # Cross validation scores
        cv_scores = cross_validate(
            model,
            np.vstack([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            cv=kf,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
        )

        metrics["CV_MAE"] = float(-cv_scores["test_neg_mean_absolute_error"].mean())
        metrics["CV_RMSE"] = float(-cv_scores["test_neg_root_mean_squared_error"].mean())
        metrics["CV_R2"] = float(cv_scores["test_r2"].mean())

        results[name] = metrics

        # Extract feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).tolist()
        else:
            importances = [0.0] * len(feature_names)

        feat_imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        feature_importances[name] = feat_imp_df.to_dict(orient="records")

    return {
        "results": results,
        "fitted_models": fitted_models,
        "feature_importances": feature_importances,
        "predictions": predictions,
    }


def run_full_training():
    """Executes the complete model training workflow."""
    print("Loading data...")
    raw_data = load_raw_data()
    course_df = build_course_level_dataset(raw_data)
    monthly_panel_df = build_monthly_panel_dataset(raw_data)

    print(f"Loaded {len(course_df)} courses and {len(monthly_panel_df)} monthly records.")

    # -------------------------------------------------------------
    # 1. Course-Level Annual Modeling (For Launches & Strategic Demand)
    # -------------------------------------------------------------
    pipeline = EduProFeaturePipeline()
    X = pipeline.fit_transform(course_df)
    feature_names = pipeline.feature_names

    y_enrollments = course_df["total_enrollments"].values
    y_revenue = course_df["total_revenue"].values

    # Train / Test split 80-20 stratified by CourseType
    indices = np.arange(len(course_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=course_df["CourseType"]
    )

    X_train, X_test = X[train_idx], X[test_idx]
    test_courses = course_df.iloc[test_idx].copy().reset_index(drop=True)

    print("\n--- Training Models for Target 1: Enrollment Count ---")
    y_train_enr, y_test_enr = y_enrollments[train_idx], y_enrollments[test_idx]
    enr_eval = train_and_evaluate_target(X_train, X_test, y_train_enr, y_test_enr, feature_names, "Enrollments")
    for mod_name, metrics in enr_eval["results"].items():
        print(f"[{mod_name}] Test R2: {metrics['R2']:.3f}, MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f} | CV R2: {metrics['CV_R2']:.3f}")

    print("\n--- Training Models for Target 2: Course Revenue ---")
    y_train_rev, y_test_rev = y_revenue[train_idx], y_revenue[test_idx]
    rev_eval = train_and_evaluate_target(X_train, X_test, y_train_rev, y_test_rev, feature_names, "Revenue")
    for mod_name, metrics in rev_eval["results"].items():
        print(f"[{mod_name}] Test R2: {metrics['R2']:.3f}, MAE: ${metrics['MAE']:.2f}, RMSE: ${metrics['RMSE']:.2f} | CV R2: {metrics['CV_R2']:.3f}")

    # -------------------------------------------------------------
    # 2. Target 3: Category Revenue Forecasting & Aggregation
    # -------------------------------------------------------------
    # Predict revenue for all courses using best model (e.g. Gradient Boosting / Random Forest)
    best_rev_model = rev_eval["fitted_models"]["Gradient Boosting"]
    all_pred_revenue = np.clip(best_rev_model.predict(X), 0, None)
    best_enr_model = enr_eval["fitted_models"]["Random Forest"]
    all_pred_enrollments = np.clip(best_enr_model.predict(X), 0, None)

    course_predictions_df = course_df[[
        "CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel",
        "CoursePrice", "CourseDuration", "CourseRating", "PrimaryTeacherName",
        "total_enrollments", "total_revenue"
    ]].copy()
    course_predictions_df["Predicted_Enrollments"] = np.round(all_pred_enrollments).astype(int)
    course_predictions_df["Predicted_Revenue"] = np.round(all_pred_revenue, 2)
    course_predictions_df["Enrollment_Residual"] = course_predictions_df["total_enrollments"] - course_predictions_df["Predicted_Enrollments"]
    course_predictions_df["Revenue_Residual"] = course_predictions_df["total_revenue"] - course_predictions_df["Predicted_Revenue"]

    # Category Level Aggregation
    category_summary = course_predictions_df.groupby("CourseCategory").agg(
        CourseCount=("CourseID", "count"),
        Actual_Total_Enrollments=("total_enrollments", "sum"),
        Predicted_Total_Enrollments=("Predicted_Enrollments", "sum"),
        Actual_Total_Revenue=("total_revenue", "sum"),
        Predicted_Total_Revenue=("Predicted_Revenue", "sum"),
        Avg_Price=("CoursePrice", "mean"),
        Avg_Rating=("CourseRating", "mean"),
    ).reset_index()

    category_summary["Revenue_Variance_Pct"] = (
        (category_summary["Predicted_Total_Revenue"] - category_summary["Actual_Total_Revenue"])
        / (category_summary["Actual_Total_Revenue"] + 1e-5) * 100
    ).round(2)

    print("\n--- Target 3: Category Revenue Summary ---")
    print(category_summary[["CourseCategory", "Actual_Total_Revenue", "Predicted_Total_Revenue", "Revenue_Variance_Pct"]])

    # -------------------------------------------------------------
    # 3. Monthly Panel Modeling (Time Series Dynamics)
    # -------------------------------------------------------------
    print("\n--- Training Monthly Dynamic Panel Models ---")
    panel_feat_pipe = EduProFeaturePipeline(include_historical=True)
    X_panel = panel_feat_pipe.fit_transform(monthly_panel_df)
    y_panel_enr = monthly_panel_df["monthly_enrollments"].values
    y_panel_rev = monthly_panel_df["monthly_revenue"].values

    # Time-based split: Jan-Sep (months 1-9) as Train, Oct-Dec (months 10-12) as Test
    train_mask = monthly_panel_df["Month_num"] <= 9
    test_mask = monthly_panel_df["Month_num"] > 9

    X_p_train, X_p_test = X_panel[train_mask], X_panel[test_mask]
    y_p_train_enr, y_p_test_enr = y_panel_enr[train_mask], y_panel_enr[test_mask]
    y_p_train_rev, y_p_test_rev = y_panel_rev[train_mask], y_panel_rev[test_mask]

    monthly_enr_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    monthly_enr_model.fit(X_p_train, y_p_train_enr)
    y_p_pred_enr = np.clip(monthly_enr_model.predict(X_p_test), 0, None)
    monthly_enr_metrics = evaluate_predictions(y_p_test_enr, y_p_pred_enr)
    print(f"Monthly Enrollments Test R2: {monthly_enr_metrics['R2']:.3f}, MAE: {monthly_enr_metrics['MAE']:.2f}")

    monthly_rev_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    monthly_rev_model.fit(X_p_train, y_p_train_rev)
    y_p_pred_rev = np.clip(monthly_rev_model.predict(X_p_test), 0, None)
    monthly_rev_metrics = evaluate_predictions(y_p_test_rev, y_p_pred_rev)
    print(f"Monthly Revenue Test R2: {monthly_rev_metrics['R2']:.3f}, MAE: ${monthly_rev_metrics['MAE']:.2f}")

    # -------------------------------------------------------------
    # 4. Save Artifacts & Models
    # -------------------------------------------------------------
    print("\nSaving model files and metadata...")
    joblib.dump(pipeline, os.path.join(MODELS_DIR, "feature_pipeline.joblib"))
    joblib.dump(enr_eval["fitted_models"], os.path.join(MODELS_DIR, "models_enrollment.joblib"))
    joblib.dump(rev_eval["fitted_models"], os.path.join(MODELS_DIR, "models_revenue.joblib"))
    joblib.dump(panel_feat_pipe, os.path.join(MODELS_DIR, "panel_pipeline.joblib"))
    joblib.dump({
        "monthly_enr": monthly_enr_model,
        "monthly_rev": monthly_rev_model
    }, os.path.join(MODELS_DIR, "monthly_models.joblib"))

    # Save summary metrics JSON
    metrics_summary = {
        "Enrollment_Models": enr_eval["results"],
        "Revenue_Models": rev_eval["results"],
        "Monthly_Enrollments_Metrics": monthly_enr_metrics,
        "Monthly_Revenue_Metrics": monthly_rev_metrics,
    }
    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save feature importances JSON
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump({
            "Enrollment": enr_eval["feature_importances"],
            "Revenue": rev_eval["feature_importances"],
        }, f, indent=2)

    # Save predictions CSVs
    course_predictions_df.to_csv(os.path.join(MODELS_DIR, "course_predictions.csv"), index=False)
    category_summary.to_csv(os.path.join(MODELS_DIR, "category_forecast.csv"), index=False)
    monthly_panel_df.to_csv(os.path.join(MODELS_DIR, "monthly_panel_data.csv"), index=False)

    print("Model training and export completed successfully!")


if __name__ == "__main__":
    run_full_training()
