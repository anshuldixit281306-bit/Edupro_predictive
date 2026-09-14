"""
Data Loader module for EduPro Online Platform.
Ingests Users, Teachers, Courses, and Transactions from Excel.
Performs data validation, relational joining, and aggregation.
"""

import os
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "EduPro Online Platform.xlsx"
)


def load_raw_data(filepath=None):
    """Loads raw sheets from Excel workbook into pandas DataFrames."""
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    excel_file = pd.ExcelFile(filepath)
    users_df = excel_file.parse("Users")
    teachers_df = excel_file.parse("Teachers")
    courses_df = excel_file.parse("Courses")
    transactions_df = excel_file.parse("Transactions")

    # Clean data types
    users_df["UserID"] = users_df["UserID"].astype(str).str.strip()
    users_df["Age"] = pd.to_numeric(users_df["Age"], errors="coerce")

    teachers_df["TeacherID"] = teachers_df["TeacherID"].astype(str).str.strip()
    teachers_df["Age"] = pd.to_numeric(teachers_df["Age"], errors="coerce")
    teachers_df["YearsOfExperience"] = pd.to_numeric(teachers_df["YearsOfExperience"], errors="coerce")
    teachers_df["TeacherRating"] = pd.to_numeric(teachers_df["TeacherRating"], errors="coerce")

    courses_df["CourseID"] = courses_df["CourseID"].astype(str).str.strip()
    courses_df["CoursePrice"] = pd.to_numeric(courses_df["CoursePrice"], errors="coerce").fillna(0.0)
    courses_df["CourseDuration"] = pd.to_numeric(courses_df["CourseDuration"], errors="coerce")
    courses_df["CourseRating"] = pd.to_numeric(courses_df["CourseRating"], errors="coerce")

    transactions_df["TransactionID"] = transactions_df["TransactionID"].astype(str).str.strip()
    transactions_df["UserID"] = transactions_df["UserID"].astype(str).str.strip()
    transactions_df["CourseID"] = transactions_df["CourseID"].astype(str).str.strip()
    transactions_df["TeacherID"] = transactions_df["TeacherID"].astype(str).str.strip()
    transactions_df["Amount"] = pd.to_numeric(transactions_df["Amount"], errors="coerce").fillna(0.0)
    transactions_df["TransactionDate"] = pd.to_datetime(transactions_df["TransactionDate"], errors="coerce")

    return {
        "users": users_df,
        "teachers": teachers_df,
        "courses": courses_df,
        "transactions": transactions_df,
    }


def build_course_level_dataset(data_dict=None):
    """
    Aggregates transactions at the course level and joins with course and instructor features.
    """
    if data_dict is None:
        data_dict = load_raw_data()

    courses = data_dict["courses"].copy()
    teachers = data_dict["teachers"].copy()
    transactions = data_dict["transactions"].copy()

    # Aggregate transactions per course
    course_txn_summary = transactions.groupby("CourseID").agg(
        total_enrollments=("TransactionID", "count"),
        total_revenue=("Amount", "sum"),
        avg_revenue_per_sale=("Amount", "mean"),
        unique_students=("UserID", "nunique"),
        unique_teachers_count=("TeacherID", "nunique"),
        first_sale_date=("TransactionDate", "min"),
        last_sale_date=("TransactionDate", "max"),
    ).reset_index()

    # Determine primary teacher per course (the teacher with most transactions for that course)
    def get_primary_teacher(sub_df):
        mode_series = sub_df["TeacherID"].mode()
        return mode_series.iloc[0] if not mode_series.empty else np.nan

    primary_teachers = transactions.groupby("CourseID").apply(get_primary_teacher, include_groups=False).reset_index()
    primary_teachers.columns = ["CourseID", "PrimaryTeacherID"]

    # Calculate average teacher rating and experience for instructors teaching this course
    txn_with_teacher = transactions.merge(
        teachers[["TeacherID", "YearsOfExperience", "TeacherRating", "Expertise"]],
        on="TeacherID",
        how="left"
    )
    course_teacher_stats = txn_with_teacher.groupby("CourseID").agg(
        avg_instructor_exp=("YearsOfExperience", "mean"),
        avg_instructor_rating=("TeacherRating", "mean"),
    ).reset_index()

    # Merge into courses dataframe
    df = courses.merge(course_txn_summary, on="CourseID", how="left")
    df = df.merge(primary_teachers, on="CourseID", how="left")
    df = df.merge(course_teacher_stats, on="CourseID", how="left")
    df = df.merge(
        teachers.rename(columns={
            "TeacherName": "PrimaryTeacherName",
            "Expertise": "PrimaryTeacherExpertise",
            "YearsOfExperience": "PrimaryTeacherExperience",
            "TeacherRating": "PrimaryTeacherRating",
        }),
        left_on="PrimaryTeacherID",
        right_on="TeacherID",
        how="left"
    )

    # Clean fillna
    df["total_enrollments"] = df["total_enrollments"].fillna(0)
    df["total_revenue"] = df["total_revenue"].fillna(0.0)
    df["avg_revenue_per_sale"] = df["avg_revenue_per_sale"].fillna(0.0)
    df["avg_instructor_exp"] = df["avg_instructor_exp"].fillna(df["PrimaryTeacherExperience"]).fillna(df["PrimaryTeacherExperience"].mean())
    df["avg_instructor_rating"] = df["avg_instructor_rating"].fillna(df["PrimaryTeacherRating"]).fillna(df["PrimaryTeacherRating"].mean())

    return df


def build_monthly_panel_dataset(data_dict=None):
    """
    Builds a course-month panel dataset (60 courses x 12 months = 720 records)
    with historical performance features (lagged enrollments, rolling averages).
    """
    if data_dict is None:
        data_dict = load_raw_data()

    courses = data_dict["courses"].copy()
    teachers = data_dict["teachers"].copy()
    transactions = data_dict["transactions"].copy()

    # Add YearMonth
    transactions["YearMonth"] = transactions["TransactionDate"].dt.to_period("M")

    # Monthly aggregates
    monthly_agg = transactions.groupby(["CourseID", "YearMonth"]).agg(
        monthly_enrollments=("TransactionID", "count"),
        monthly_revenue=("Amount", "sum"),
        avg_trans_amount=("Amount", "mean"),
    ).reset_index()

    # Create complete grid of CourseID x All Months
    all_courses = courses["CourseID"].unique()
    all_months = pd.period_range("2025-01", "2025-12", freq="M")
    full_index = pd.MultiIndex.from_product([all_courses, all_months], names=["CourseID", "YearMonth"])
    panel = pd.DataFrame(index=full_index).reset_index()

    panel = panel.merge(monthly_agg, on=["CourseID", "YearMonth"], how="left")
    panel["monthly_enrollments"] = panel["monthly_enrollments"].fillna(0).astype(int)
    panel["monthly_revenue"] = panel["monthly_revenue"].fillna(0.0)
    panel["avg_trans_amount"] = panel["avg_trans_amount"].fillna(0.0)

    # Sort for time series features
    panel["YearMonth_str"] = panel["YearMonth"].astype(str)
    panel["Month_num"] = panel["YearMonth"].dt.month
    panel = panel.sort_values(["CourseID", "YearMonth"]).reset_index(drop=True)

    # Historical Performance Features per course:
    # 1. Past 1-month enrollment & revenue (Lag 1)
    # 2. Past 3-month rolling average enrollment & revenue
    # 3. Cumulative past enrollment count
    # 4. Cumulative past average revenue
    panel["past_enrollment_lag1"] = panel.groupby("CourseID")["monthly_enrollments"].shift(1)
    panel["past_revenue_lag1"] = panel.groupby("CourseID")["monthly_revenue"].shift(1)

    panel["past_enrollment_roll3"] = panel.groupby("CourseID")["monthly_enrollments"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    panel["past_revenue_roll3"] = panel.groupby("CourseID")["monthly_revenue"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    panel["past_cum_enrollments"] = panel.groupby("CourseID")["monthly_enrollments"].transform(
        lambda s: s.shift(1).cumsum()
    ).fillna(0)
    panel["past_cum_revenue"] = panel.groupby("CourseID")["monthly_revenue"].transform(
        lambda s: s.shift(1).cumsum()
    ).fillna(0.0)

    # Revenue per enrollment historical
    panel["past_revenue_per_enrollment"] = np.where(
        panel["past_cum_enrollments"] > 0,
        panel["past_cum_revenue"] / panel["past_cum_enrollments"],
        0.0
    )

    # Backfill/Fill initial lags with course mean or zeros
    panel["past_enrollment_lag1"] = panel.groupby("CourseID")["past_enrollment_lag1"].transform(
        lambda s: s.fillna(s.mean() if not pd.isna(s.mean()) else 0)
    )
    panel["past_revenue_lag1"] = panel.groupby("CourseID")["past_revenue_lag1"].transform(
        lambda s: s.fillna(s.mean() if not pd.isna(s.mean()) else 0)
    )
    panel["past_enrollment_roll3"] = panel["past_enrollment_roll3"].fillna(panel["past_enrollment_lag1"])
    panel["past_revenue_roll3"] = panel["past_revenue_roll3"].fillna(panel["past_revenue_lag1"])

    # Merge static course and teacher attributes
    course_level_df = build_course_level_dataset(data_dict)
    static_cols = [
        "CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel",
        "CoursePrice", "CourseDuration", "CourseRating",
        "PrimaryTeacherID", "PrimaryTeacherName", "PrimaryTeacherExpertise",
        "PrimaryTeacherExperience", "PrimaryTeacherRating",
        "avg_instructor_exp", "avg_instructor_rating"
    ]
    panel = panel.merge(course_level_df[static_cols], on="CourseID", how="left")

    return panel


if __name__ == "__main__":
    print("Testing data loader...")
    raw = load_raw_data()
    print("Raw loaded. Courses:", len(raw["courses"]), "Transactions:", len(raw["transactions"]))
    c_df = build_course_level_dataset(raw)
    print("Course-level aggregated shape:", c_df.shape)
    p_df = build_monthly_panel_dataset(raw)
    print("Monthly panel shape:", p_df.shape)
    print("Sample panel columns:", list(p_df.columns)[:12])
