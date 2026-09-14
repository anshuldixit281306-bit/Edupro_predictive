"""
Feature Engineering module for EduPro Online Platform.
Implements:
- Price bands (low / medium / high / free)
- Duration buckets (short / medium / long)
- Rating tiers (low / average / high)
- Course level encoding (ordinal / one-hot)
- Instructor experience buckets (junior / mid / senior)
- Teacher rating score
- Expertise-category match score
- Historical performance features (past enrollments, past revenue, revenue per enrollment)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def categorize_price(price):
    """Categorizes course price into bands: Free, Low, Medium, High."""
    if price <= 0.01:
        return "Free"
    elif price < 100:
        return "Low"
    elif price <= 250:
        return "Medium"
    else:
        return "High"


def categorize_duration(duration):
    """Categorizes course duration into buckets: Short (<15h), Medium (15-30h), Long (>30h)."""
    if duration < 15:
        return "Short"
    elif duration <= 30:
        return "Medium"
    else:
        return "Long"


def categorize_rating(rating):
    """Categorizes course or teacher rating into tiers: Low (<3.0), Average (3.0-4.0), High (>4.0)."""
    if rating < 3.0:
        return "Low"
    elif rating <= 4.0:
        return "Average"
    else:
        return "High"


def categorize_experience(exp):
    """Categorizes instructor experience into buckets: Junior (<5y), Mid (5-10y), Senior (>10y)."""
    if exp < 5:
        return "Junior"
    elif exp <= 10:
        return "Mid"
    else:
        return "Senior"


def calculate_expertise_match(course_category, teacher_expertise):
    """
    Computes semantic/exact match score (1.0 or 0.0) between course category and instructor expertise.
    Handles fuzzy matches like 'Programming' / 'Python Basics' or 'Machine Learning' / 'Artificial Intelligence'.
    """
    if pd.isna(course_category) or pd.isna(teacher_expertise):
        return 0.0

    c_cat = str(course_category).strip().lower()
    t_exp = str(teacher_expertise).strip().lower()

    if c_cat == t_exp:
        return 1.0

    # Domain synonyms/overlaps
    tech_aliases = {
        "programming": ["programming", "web development", "software"],
        "web development": ["web development", "programming", "design"],
        "machine learning": ["machine learning", "data science", "artificial intelligence"],
        "artificial intelligence": ["artificial intelligence", "machine learning", "data science"],
        "data science": ["data science", "machine learning", "artificial intelligence", "analytics"],
        "marketing": ["marketing", "digital marketing"],
        "digital marketing": ["digital marketing", "marketing"],
        "project management": ["project management", "business"],
        "business": ["business", "finance", "project management"],
        "finance": ["finance", "business"],
    }

    if c_cat in tech_aliases and t_exp in tech_aliases[c_cat]:
        return 0.8

    if c_cat in t_exp or t_exp in c_cat:
        return 0.7

    return 0.0


def engineer_features(df, is_panel=False):
    """
    Adds all specified engineered features to the input dataframe.
    Works seamlessly on both course-level aggregate df and monthly panel df.
    """
    feat_df = df.copy()

    # 1. Course features
    feat_df["PriceBand"] = feat_df["CoursePrice"].apply(categorize_price)
    feat_df["DurationBucket"] = feat_df["CourseDuration"].apply(categorize_duration)
    feat_df["CourseRatingTier"] = feat_df["CourseRating"].apply(categorize_rating)

    # Course level ordinal encoding
    level_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    feat_df["CourseLevelEncoded"] = feat_df["CourseLevel"].map(level_map).fillna(1).astype(int)

    # Course type binary
    feat_df["IsPaid"] = (feat_df["CourseType"] == "Paid").astype(int)

    # 2. Instructor features
    inst_exp_col = "avg_instructor_exp" if "avg_instructor_exp" in feat_df.columns else "PrimaryTeacherExperience"
    inst_rat_col = "avg_instructor_rating" if "avg_instructor_rating" in feat_df.columns else "PrimaryTeacherRating"

    feat_df["InstructorExperienceBucket"] = feat_df[inst_exp_col].apply(categorize_experience)
    feat_df["InstructorRatingTier"] = feat_df[inst_rat_col].apply(categorize_rating)
    feat_df["TeacherRatingScore"] = feat_df[inst_rat_col].astype(float)
    feat_df["TeacherExperienceYears"] = feat_df[inst_exp_col].astype(float)

    # Expertise-Category Match Score
    if "PrimaryTeacherExpertise" in feat_df.columns:
        feat_df["ExpertiseCategoryMatchScore"] = feat_df.apply(
            lambda r: calculate_expertise_match(r["CourseCategory"], r["PrimaryTeacherExpertise"]),
            axis=1
        )
    else:
        feat_df["ExpertiseCategoryMatchScore"] = 0.5

    # 3. Historical performance features (only for panel data with true historical lags)
    if is_panel:
        feat_df["PastEnrollmentCount"] = feat_df["past_enrollment_roll3"].fillna(0)
        feat_df["PastAverageRevenue"] = feat_df["past_revenue_roll3"].fillna(0.0)
        feat_df["RevenuePerEnrollment"] = feat_df["past_revenue_per_enrollment"].fillna(0.0)
    else:
        # For course static profile (avoiding target leakage of total_enrollments/total_revenue)
        feat_df["PastEnrollmentCount"] = 0.0
        feat_df["PastAverageRevenue"] = 0.0
        feat_df["RevenuePerEnrollment"] = 0.0

    return feat_df


class EduProFeaturePipeline:
    """
    Full preprocessing pipeline:
    - Categorical One-Hot Encoding
    - Numerical Standard Scaling
    - Handles transformation for inference
    """

    def __init__(self, include_historical=False):
        self.include_historical = include_historical
        self.cat_cols = [
            "CourseCategory", "PriceBand", "DurationBucket",
            "CourseRatingTier", "InstructorExperienceBucket"
        ]
        self.num_cols = [
            "CoursePrice", "CourseDuration", "CourseRating",
            "CourseLevelEncoded", "IsPaid",
            "TeacherExperienceYears", "TeacherRatingScore",
            "ExpertiseCategoryMatchScore"
        ]
        if self.include_historical:
            self.num_cols += ["PastEnrollmentCount", "PastAverageRevenue", "RevenuePerEnrollment"]

        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X_df):
        engineered = engineer_features(X_df, is_panel=self.include_historical)
        self.scaler.fit(engineered[self.num_cols])
        self.ohe.fit(engineered[self.cat_cols])
        ohe_names = list(self.ohe.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + ohe_names
        self.is_fitted = True
        return self

    def transform(self, X_df):
        if not self.is_fitted:
            raise ValueError("Pipeline has not been fitted yet.")
        engineered = engineer_features(X_df, is_panel=self.include_historical)
        num_scaled = self.scaler.transform(engineered[self.num_cols])
        cat_encoded = self.ohe.transform(engineered[self.cat_cols])
        X_mat = np.hstack([num_scaled, cat_encoded])
        return X_mat

    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)


if __name__ == "__main__":
    from data_loader import load_raw_data, build_course_level_dataset
    raw = load_raw_data()
    course_df = build_course_level_dataset(raw)
    feat_df = engineer_features(course_df)
    print("Engineered features sample:\n", feat_df[["CourseName", "PriceBand", "DurationBucket", "ExpertiseCategoryMatchScore"]].head())

    pipe = EduProFeaturePipeline()
    X = pipe.fit_transform(course_df)
    print("Transformed matrix shape:", X.shape)
    print("Number of feature names:", len(pipe.feature_names))
