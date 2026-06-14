# =========================================================
# IMPORTS
# =========================================================

from pathlib import Path
from PIL import Image, ImageChops
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Engagement & Product Utilization Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# FILE PATH CONFIGURATION
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

ASSETS_DIR = APP_DIR / "assets"
CSS_FILE = ASSETS_DIR / "ecb_dashboard.css"

IMAGE_DIR = ASSETS_DIR / "images"

ECB_LOGO_FILE = IMAGE_DIR / "ecb_logo.svg"
ECB_LOGO_LINE_FILE = IMAGE_DIR / "ecb_logo_line.svg"
UNIFIED_MENTOR_LOGO_FILE = IMAGE_DIR / "unified_mentor.png"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RETENTION_STRENGTH_DIR = PROCESSED_DATA_DIR / "retention_strength_scoring"
TRENDS_INSIGHTS_DIR = PROCESSED_DATA_DIR / "trends_insights_recommendations"

FINAL_MASTER_DATA_FILE = RETENTION_STRENGTH_DIR / "retention_strength_master_dataset.csv"

# =========================================================
# LOGO HELPER FUNCTION
# =========================================================
def prepare_sidebar_logo_image(logo_path: Path):
    """
    Prepare Unified Mentor logo for sidebar display by trimming extra padding.
    Works best for PNG/JPG/JPEG/WEBP.
    """
    try:
        logo_image = Image.open(logo_path).convert("RGBA")

        alpha_channel = logo_image.getchannel("A")
        alpha_bbox = alpha_channel.getbbox()

        if alpha_bbox:
            logo_image = logo_image.crop(alpha_bbox)

        background = Image.new("RGBA", logo_image.size, logo_image.getpixel((0, 0)))
        diff = ImageChops.difference(logo_image, background)
        bbox = diff.getbbox()

        if bbox:
            logo_image = logo_image.crop(bbox)

        padding_x = 8
        padding_y = 5

        final_logo = Image.new(
            "RGBA",
            (
                logo_image.width + padding_x * 2,
                logo_image.height + padding_y * 2,
            ),
            (255, 255, 255, 0),
        )

        final_logo.paste(logo_image, (padding_x, padding_y), logo_image)

        return final_logo

    except Exception:
        return str(logo_path)

# =========================================================
# LOAD CUSTOM CSS
# =========================================================

def load_css(css_path: Path) -> None:
    """
    Load external CSS safely.

    Current rule:
    - CSS file is connected in Step 1A.
    - CSS file can stay blank during app.py development.
    - Styling work will start only after full app.py approval.
    """
    if not css_path.exists():
        st.warning("CSS file not found: app/assets/ecb_dashboard.css")
        return

    css_content = css_path.read_text(encoding="utf-8").strip()

    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


load_css(CSS_FILE)

# =========================================================
# REBUILD STEP 1B — DYNAMIC NATIVE ECB HEADER SECTION
# =========================================================

def render_ecb_header(
    logo_path: Path,
    logo_line_path: Path,
    brand_caption: str = "BANKING RETENTION INTELLIGENCE",
    dashboard_title: str = "Customer Engagement & Product Utilization Analytics",
    dashboard_subtitle: str = (
        "Retention strategy dashboard analyzing engagement behavior, product depth, "
        "high-value disengagement, and relationship strength."
    ),
) -> None:
    """
    Render the dashboard header using only native Streamlit components.

    Structure:
    - Full-width ECB institutional logo-line row
    - Compact institutional caption row
    - Divider
    - ECB flag logo + dashboard title row
    - Divider

    Important:
    - Logo-line is intentionally not placed inside narrow columns.
    - This prevents mobile/tablet clipping.
    """
    with st.container(key="ecb_header"):

        # Full-width ECB logo-line row
        if logo_line_path.exists():
            st.image(
                str(logo_line_path),
                width=330,
           )
        else:
            st.markdown("**EUROPEAN CENTRAL BANK | EUROSYSTEM**")

        st.caption(brand_caption)

        st.divider()

        logo_col, title_col = st.columns(
            [0.13, 0.87],
            vertical_alignment="center",
        )

        with logo_col:
            if logo_path.exists():
                st.image(str(logo_path), width=86)
            else:
                st.warning("ECB logo file not found.")

        with title_col:
            st.title(dashboard_title)
            st.write(dashboard_subtitle)

        st.divider()
        
render_ecb_header(
    logo_path=ECB_LOGO_FILE,
    logo_line_path=ECB_LOGO_LINE_FILE,
)
    
# =========================================================
# REBUILD STEP 1C — DATA LOADING + REQUIRED COLUMN VALIDATION
# =========================================================

REQUIRED_DASHBOARD_COLUMNS = [
    # Raw dataset foundation
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",

    # Dashboard-ready derived columns
    "Churn_Status",
    "Engagement_Status",
    "Credit_Card_Status",
    "Product_Depth_Segment",
    "High_Value_Customer_Status",
    "High_Value_Disengaged_Status",
    "Salary_Balance_Mismatch_Status",
    "Customer_Risk_Segment",
    "Relationship_Strength_Index",
    "Relationship_Strength_Tier",
    "Retention_Strength_Score",
    "Retention_Strength_Tier",
]


@st.cache_data(show_spinner="Loading final retention master dataset...")
def load_final_master_dataset(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def validate_required_columns(
    data_frame: pd.DataFrame,
    required_columns: list[str],
) -> tuple[bool, list[str]]:
    """
    Check whether all required dashboard columns are available.
    """
    missing_columns = [
        column for column in required_columns
        if column not in data_frame.columns
    ]

    return len(missing_columns) == 0, missing_columns


def validate_binary_columns(data_frame: pd.DataFrame) -> dict[str, list]:
    """
    Validate binary dashboard columns.

    Expected values:
    - HasCrCard: 0 or 1
    - IsActiveMember: 0 or 1
    - Exited: 0 or 1
    """
    binary_columns = ["HasCrCard", "IsActiveMember", "Exited"]
    validation_issues = {}

    for column in binary_columns:
        if column in data_frame.columns:
            unique_values = sorted(data_frame[column].dropna().unique().tolist())
            unexpected_values = [
                value for value in unique_values
                if value not in [0, 1]
            ]

            if unexpected_values:
                validation_issues[column] = unexpected_values

    return validation_issues


if not FINAL_MASTER_DATA_FILE.exists():
    st.error(
        "Final master dataset not found. Please verify this path:"
    )
    st.write(FINAL_MASTER_DATA_FILE)
    st.stop()


df_master = load_final_master_dataset(FINAL_MASTER_DATA_FILE)

columns_valid, missing_required_columns = validate_required_columns(
    df_master,
    REQUIRED_DASHBOARD_COLUMNS,
)

binary_validation_issues = validate_binary_columns(df_master)

if not columns_valid:
    st.error("Required dashboard columns are missing from the final master dataset.")
    st.write("Missing columns:", missing_required_columns)
    st.stop()

if binary_validation_issues:
    st.error("Binary column validation failed.")
    st.write(binary_validation_issues)
    st.stop()
    
# =========================================================
# REBUILD STEP 1D — HELPER FUNCTIONS + REQUIRED FILTER OPTION PREPARATION
# Project guideline compliant:
# 1. Engagement filters
# 2. Product count slider
# 3. Balance and salary thresholds
# =========================================================

def get_clean_sorted_options(data_frame: pd.DataFrame, column_name: str) -> list:
    """
    Return clean, sorted unique values from a column for required filters.
    """
    if data_frame.empty or column_name not in data_frame.columns:
        return []

    values = (
        data_frame[column_name]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]

    return sorted(values.unique().tolist())


def safe_divide(numerator: float, denominator: float, multiplier: float = 1.0) -> float:
    """
    Safely divide two values and return 0 when denominator is zero or invalid.
    """
    try:
        if denominator is None or pd.isna(denominator) or denominator == 0:
            return 0.0

        return (numerator / denominator) * multiplier

    except Exception:
        return 0.0


def calculate_churn_rate(data_frame: pd.DataFrame) -> float:
    """
    Calculate churn rate percentage from Exited column.
    """
    if data_frame.empty or "Exited" not in data_frame.columns:
        return 0.0

    exited_values = pd.to_numeric(data_frame["Exited"], errors="coerce").fillna(0)

    return safe_divide(
        numerator=exited_values.sum(),
        denominator=len(data_frame),
        multiplier=100,
    )


def calculate_retention_rate(data_frame: pd.DataFrame) -> float:
    """
    Calculate retention rate percentage from churn rate.
    """
    if data_frame.empty:
        return 0.0

    return 100 - calculate_churn_rate(data_frame)


def calculate_customer_count(data_frame: pd.DataFrame) -> int:
    """
    Calculate unique customers when CustomerId exists, otherwise row count.
    """
    if data_frame.empty:
        return 0

    if "CustomerId" in data_frame.columns:
        return data_frame["CustomerId"].nunique()

    return len(data_frame)


def create_segment_summary(
    data_frame: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    """
    Create segment-level customer, churn, retention, product, and financial summary.
    This is not used for sidebar filtering.
    It will support the 4 dashboard modules later.
    """
    required_columns = [
        "CustomerId",
        "Exited",
        "Retention_Strength_Score",
        "Relationship_Strength_Index",
        "NumOfProducts",
        "Balance",
        "EstimatedSalary",
    ]

    if (
        data_frame.empty
        or segment_column not in data_frame.columns
        or any(column not in data_frame.columns for column in required_columns)
    ):
        return pd.DataFrame()

    summary = (
        data_frame
        .groupby(segment_column, dropna=False)
        .agg(
            Customer_Count=("CustomerId", "nunique"),
            Churned_Customers=("Exited", "sum"),
            Avg_Retention_Strength_Score=("Retention_Strength_Score", "mean"),
            Avg_Relationship_Strength_Index=("Relationship_Strength_Index", "mean"),
            Avg_Num_Products=("NumOfProducts", "mean"),
            Avg_Balance=("Balance", "mean"),
            Avg_Estimated_Salary=("EstimatedSalary", "mean"),
        )
        .reset_index()
    )

    summary["Churn_Rate"] = summary.apply(
        lambda row: safe_divide(
            row["Churned_Customers"],
            row["Customer_Count"],
            100,
        ),
        axis=1,
    )

    summary["Retention_Rate"] = 100 - summary["Churn_Rate"]

    numeric_columns = [
        "Churn_Rate",
        "Retention_Rate",
        "Avg_Retention_Strength_Score",
        "Avg_Relationship_Strength_Index",
        "Avg_Num_Products",
        "Avg_Balance",
        "Avg_Estimated_Salary",
    ]

    for column in numeric_columns:
        if column in summary.columns:
            summary[column] = summary[column].round(2)

    return summary


# =========================================================
# REQUIRED SIDEBAR FILTER OPTION PREPARATION
# Only project-required user capabilities are prepared here.
# =========================================================

engagement_status_options = get_clean_sorted_options(
    df_master,
    "Engagement_Status",
)

if not engagement_status_options:
    st.error(
        "Engagement filter options could not be prepared because Engagement_Status is missing or empty."
    )
    st.stop()


product_count_min = int(pd.to_numeric(df_master["NumOfProducts"], errors="coerce").min())
product_count_max = int(pd.to_numeric(df_master["NumOfProducts"], errors="coerce").max())

balance_min = float(pd.to_numeric(df_master["Balance"], errors="coerce").min())
balance_max = float(pd.to_numeric(df_master["Balance"], errors="coerce").max())

salary_min = float(pd.to_numeric(df_master["EstimatedSalary"], errors="coerce").min())
salary_max = float(pd.to_numeric(df_master["EstimatedSalary"], errors="coerce").max())


# Streamlit sliders will start from 0 for cleaner threshold control.
balance_slider_min = 0.0
salary_slider_min = 0.0

balance_slider_max = float(np.ceil(balance_max / 1000) * 1000)
salary_slider_max = float(np.ceil(salary_max / 1000) * 1000)


# Guard against invalid slider ranges
if product_count_min == product_count_max:
    product_count_max = product_count_min + 1

if balance_slider_max <= balance_slider_min:
    balance_slider_max = balance_slider_min + 1000.0

if salary_slider_max <= salary_slider_min:
    salary_slider_max = salary_slider_min + 1000.0
    
# =========================================================
# REBUILD STEP 1E — GUIDELINE-COMPLIANT DYNAMIC SIDEBAR
# Only required user capabilities:
# 1. Engagement filters
# 2. Product count slider
# 3. Balance and salary thresholds
# =========================================================

def render_sidebar_filters(
    engagement_options: list,
    product_min: int,
    product_max: int,
    balance_min_value: float,
    balance_max_value: float,
    salary_min_value: float,
    salary_max_value: float,
    unified_mentor_logo_path: Path,
) -> dict:
    """
    Render only project-required sidebar filters.

    Locked sidebar scope:
    - Keep the same sidebar items only.
    - Keep all filter return keys unchanged.
    - Keep filters fully connected with the dashboard filtering system.
    """
    selected_filters = {}

    with st.sidebar:
        with st.container(key="sidebar_logo_panel"):
            if unified_mentor_logo_path.exists():
                sidebar_logo_image = prepare_sidebar_logo_image(unified_mentor_logo_path)
                st.image(
                    sidebar_logo_image,
                    width=225,
                )

        with st.container(key="sidebar_header_panel"):
            st.title("Dashboard Filters")

            st.caption(
                "Use the required controls to filter customer engagement, product count, "
                "balance threshold, and salary threshold."
            )

        st.divider()

        with st.container(key="sidebar_engagement_panel"):
            st.subheader("Engagement Filters")

            selected_filters["engagement_status"] = st.multiselect(
                label="Engagement Status",
                options=engagement_options,
                default=engagement_options,
                help=(
                    "Filter customers by engagement behavior. "
                    "Select both Active and Inactive members to compare engagement retention ratio."
                ),
            )

        st.divider()

        with st.container(key="sidebar_product_panel"):
            st.subheader("Product Count Slider")

            selected_filters["product_count_range"] = st.slider(
                label="Product Count Range",
                min_value=product_min,
                max_value=product_max,
                value=(product_min, product_max),
                step=1,
                format="%d",
                key="filter_product_count_range",
                help="Filter customers by number of banking products used.",
            )

        st.divider()

        with st.container(key="sidebar_threshold_panel"):
            st.subheader("Balance & Salary Thresholds")

            selected_filters["min_balance"] = st.slider(
                label="Minimum Balance Threshold",
                min_value=balance_min_value,
                max_value=balance_max_value,
                value=balance_min_value,
                step=1000.0,
                format="%.0f",
                key="filter_min_balance",
                help="Show customers with balance greater than or equal to this threshold.",
            )

            selected_filters["min_salary"] = st.slider(
                label="Minimum Salary Threshold",
                min_value=salary_min_value,
                max_value=salary_max_value,
                value=salary_min_value,
                step=1000.0,
                format="%.0f",
                key="filter_min_salary",
                help="Show customers with estimated salary greater than or equal to this threshold.",
            )

        st.divider()
        

    return selected_filters

dashboard_filters = render_sidebar_filters(
    engagement_options=engagement_status_options,
    product_min=product_count_min,
    product_max=product_count_max,
    balance_min_value=balance_slider_min,
    balance_max_value=balance_slider_max,
    salary_min_value=salary_slider_min,
    salary_max_value=salary_slider_max,
    unified_mentor_logo_path=UNIFIED_MENTOR_LOGO_FILE,
)


selected_engagement_status = dashboard_filters["engagement_status"]
selected_product_count_range = dashboard_filters["product_count_range"]
selected_min_balance = dashboard_filters["min_balance"]
selected_min_salary = dashboard_filters["min_salary"]
    
# =========================================================
# REBUILD STEP 1F — GUIDELINE-COMPLIANT FILTERED DATAFRAME SYSTEM
# Uses only required sidebar filters:
# 1. Engagement filters
# 2. Product count slider
# 3. Balance and salary thresholds
# =========================================================

def apply_dashboard_filters(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply only project-required dashboard filters.

    Project guideline alignment:
    - Engagement_Status filter
    - NumOfProducts range slider
    - Balance minimum threshold
    - EstimatedSalary minimum threshold
    """
    if data_frame.empty:
        return data_frame.copy()

    filtered_data = data_frame.copy()

    # Engagement filter
    if selected_engagement_status and "Engagement_Status" in filtered_data.columns:
        filtered_data = filtered_data[
            filtered_data["Engagement_Status"].isin(selected_engagement_status)
        ]

    # Product count range filter
    if "NumOfProducts" in filtered_data.columns:
        filtered_data["NumOfProducts"] = pd.to_numeric(
            filtered_data["NumOfProducts"],
            errors="coerce",
        )

        filtered_data = filtered_data[
            filtered_data["NumOfProducts"].between(
                selected_product_count_range[0],
                selected_product_count_range[1],
                inclusive="both",
            )
        ]

    # Minimum balance threshold
    if "Balance" in filtered_data.columns:
        filtered_data["Balance"] = pd.to_numeric(
            filtered_data["Balance"],
            errors="coerce",
        )

        filtered_data = filtered_data[
            filtered_data["Balance"] >= selected_min_balance
        ]

    # Minimum salary threshold
    if "EstimatedSalary" in filtered_data.columns:
        filtered_data["EstimatedSalary"] = pd.to_numeric(
            filtered_data["EstimatedSalary"],
            errors="coerce",
        )

        filtered_data = filtered_data[
            filtered_data["EstimatedSalary"] >= selected_min_salary
        ]

    return filtered_data.reset_index(drop=True)


filtered_df = apply_dashboard_filters(df_master)


if filtered_df.empty:
    st.warning(
        "No customers match the selected filters. Please reduce the balance/salary threshold "
        "or widen the product count range."
    )
    st.stop()
    
# =========================================================
# REBUILD STEP 1G — REQUIRED KPI CALCULATION FUNCTIONS ONLY
# NumPy-safe KPI logic | KPI display will be added in Step 1H
# =========================================================

def safe_np_divide(
    numerator: float,
    denominator: float,
    multiplier: float = 1.0,
    unavailable_as_nan: bool = False,
) -> float:
    """
    NumPy-safe division helper.

    unavailable_as_nan=True:
    - Used when a KPI should be treated as unavailable instead of showing misleading 0.
    """
    try:
        if denominator is None or pd.isna(denominator) or float(denominator) == 0:
            return np.nan if unavailable_as_nan else 0.0

        result = np.divide(float(numerator), float(denominator)) * multiplier

        if not np.isfinite(result):
            return np.nan if unavailable_as_nan else 0.0

        return float(result)

    except Exception:
        return np.nan if unavailable_as_nan else 0.0


def safe_np_round(value: float, decimals: int = 2) -> float:
    """
    Round numeric values safely while preserving np.nan.
    """
    try:
        if value is None or pd.isna(value) or not np.isfinite(value):
            return np.nan

        return round(float(value), decimals)

    except Exception:
        return np.nan


def normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Normalize text values for reliable KPI status matching.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def calculate_retention_rate_np(data_frame: pd.DataFrame) -> float:
    """
    NumPy-safe retention rate.

    Exited:
    - 1 = Churned
    - 0 = Retained
    """
    if data_frame.empty or "Exited" not in data_frame.columns:
        return np.nan

    customer_count = len(data_frame)

    if customer_count == 0:
        return np.nan

    churned_customers = pd.to_numeric(
        data_frame["Exited"],
        errors="coerce",
    ).fillna(0).sum()

    churn_rate = safe_np_divide(
        numerator=churned_customers,
        denominator=customer_count,
        multiplier=100.0,
        unavailable_as_nan=True,
    )

    if pd.isna(churn_rate):
        return np.nan

    retention_rate = np.subtract(100.0, churn_rate)

    return safe_np_round(retention_rate, 2)


def calculate_segment_retention_rate_np(
    data_frame: pd.DataFrame,
    segment_column: str,
    segment_value: str,
) -> float:
    """
    Calculate retention rate for a specific segment.
    Returns np.nan when the segment is unavailable after filtering.
    """
    if data_frame.empty or segment_column not in data_frame.columns:
        return np.nan

    segment_data = data_frame[
        data_frame[segment_column].astype(str).str.strip().eq(segment_value)
    ]

    if segment_data.empty:
        return np.nan

    return calculate_retention_rate_np(segment_data)


def calculate_engagement_retention_ratio(data_frame: pd.DataFrame) -> float:
    """
    Required KPI:
    Engagement Retention Ratio

    Business Meaning:
    Active customer retention compared with inactive customer retention.

    Formula:
    Active Member Retention Rate / Inactive Member Retention Rate

    If Active or Inactive segment is unavailable after filtering, return np.nan.
    """
    if data_frame.empty or "Engagement_Status" not in data_frame.columns:
        return np.nan

    active_retention_rate = calculate_segment_retention_rate_np(
        data_frame=data_frame,
        segment_column="Engagement_Status",
        segment_value="Active Member",
    )

    inactive_retention_rate = calculate_segment_retention_rate_np(
        data_frame=data_frame,
        segment_column="Engagement_Status",
        segment_value="Inactive Member",
    )

    ratio = safe_np_divide(
        numerator=active_retention_rate,
        denominator=inactive_retention_rate,
        multiplier=1.0,
        unavailable_as_nan=True,
    )

    return safe_np_round(ratio, 2)


def calculate_product_depth_index(data_frame: pd.DataFrame) -> float:
    """
    Required KPI:
    Product Depth Index

    Business Meaning:
    Product usage depth among retained customers compared with the filtered base.

    Formula:
    Average NumOfProducts among retained customers
    /
    Average NumOfProducts overall
    × 100
    """
    required_columns = ["NumOfProducts", "Exited"]

    if data_frame.empty or any(column not in data_frame.columns for column in required_columns):
        return np.nan

    working_df = data_frame.copy()
    working_df["NumOfProducts"] = pd.to_numeric(
        working_df["NumOfProducts"],
        errors="coerce",
    )

    overall_avg_products = working_df["NumOfProducts"].mean()

    retained_df = working_df[working_df["Exited"] == 0]

    if retained_df.empty:
        return np.nan

    retained_avg_products = retained_df["NumOfProducts"].mean()

    product_depth_index = safe_np_divide(
        numerator=retained_avg_products,
        denominator=overall_avg_products,
        multiplier=100.0,
        unavailable_as_nan=True,
    )

    return safe_np_round(product_depth_index, 2)


def get_high_value_customer_mask(data_frame: pd.DataFrame) -> pd.Series:
    """
    Build a safe high-value customer mask.

    Preferred exact match:
    - High Value Customer

    Negative statuses such as regular/standard/non/no are excluded.
    """
    if data_frame.empty or "High_Value_Customer_Status" not in data_frame.columns:
        return pd.Series(False, index=data_frame.index)

    status = normalize_text_series(data_frame["High_Value_Customer_Status"])

    exact_high_value = status.eq("high value customer")

    fallback_high_value = (
        status.str.contains("high", na=False)
        & status.str.contains("value", na=False)
        & ~status.str.contains("no high|non high|not high|regular|standard", regex=True, na=False)
    )

    return exact_high_value | fallback_high_value


def get_high_value_disengaged_mask(data_frame: pd.DataFrame) -> pd.Series:
    """
    Build a safe high-value disengaged mask.

    Preferred exact match:
    - High Value Disengaged

    Important:
    - This avoids counting 'No High Value Disengaged' as disengaged.
    """
    if data_frame.empty or "High_Value_Disengaged_Status" not in data_frame.columns:
        return pd.Series(False, index=data_frame.index)

    status = normalize_text_series(data_frame["High_Value_Disengaged_Status"])

    exact_disengaged = status.eq("high value disengaged")

    fallback_disengaged = (
        status.str.contains("high", na=False)
        & status.str.contains("value", na=False)
        & status.str.contains("disengaged", na=False)
        & ~status.str.contains("no high|non high|not high|not disengaged|no disengaged", regex=True, na=False)
    )

    return exact_disengaged | fallback_disengaged


def calculate_high_balance_disengagement_rate(data_frame: pd.DataFrame) -> float:
    """
    Required KPI:
    High-Balance Disengagement Rate

    Business Meaning:
    Premium churn risk among high-value customers.

    Formula:
    High Value Disengaged Customers / High Value Customers × 100

    Uses:
    - High_Value_Customer_Status
    - High_Value_Disengaged_Status

    Does NOT use contains('Disengaged') directly.
    """
    required_columns = [
        "CustomerId",
        "High_Value_Customer_Status",
        "High_Value_Disengaged_Status",
    ]

    if data_frame.empty or any(column not in data_frame.columns for column in required_columns):
        return np.nan

    high_value_mask = get_high_value_customer_mask(data_frame)
    high_value_disengaged_mask = get_high_value_disengaged_mask(data_frame)

    high_value_df = data_frame[high_value_mask]
    high_value_disengaged_df = data_frame[high_value_mask & high_value_disengaged_mask]

    high_value_customer_count = high_value_df["CustomerId"].nunique()
    high_value_disengaged_count = high_value_disengaged_df["CustomerId"].nunique()

    rate = safe_np_divide(
        numerator=high_value_disengaged_count,
        denominator=high_value_customer_count,
        multiplier=100.0,
        unavailable_as_nan=True,
    )

    return safe_np_round(rate, 2)


def get_credit_card_customer_mask(data_frame: pd.DataFrame) -> pd.Series:
    """
    Build safe credit card holder mask.

    Preferred exact match:
    - Has Credit Card

    Avoids accidentally counting 'No Credit Card'.
    """
    if data_frame.empty or "Credit_Card_Status" not in data_frame.columns:
        return pd.Series(False, index=data_frame.index)

    status = normalize_text_series(data_frame["Credit_Card_Status"])

    exact_has_card = status.eq("has credit card")

    fallback_has_card = (
        status.str.contains("credit", na=False)
        & status.str.contains("card", na=False)
        & ~status.str.contains("no credit|without|not", regex=True, na=False)
    )

    return exact_has_card | fallback_has_card


def calculate_credit_card_stickiness_score(data_frame: pd.DataFrame) -> float:
    """
    Required KPI:
    Credit Card Stickiness Score

    Business Meaning:
    Retention rate among customers who have a credit card.

    Formula:
    Retained credit-card customers / total credit-card customers × 100
    """
    required_columns = ["Credit_Card_Status", "Exited"]

    if data_frame.empty or any(column not in data_frame.columns for column in required_columns):
        return np.nan

    credit_card_mask = get_credit_card_customer_mask(data_frame)
    credit_card_df = data_frame[credit_card_mask]

    if credit_card_df.empty:
        return np.nan

    return calculate_retention_rate_np(credit_card_df)


def calculate_relationship_strength_index(data_frame: pd.DataFrame) -> float:
    """
    Required KPI:
    Relationship Strength Index

    Business Meaning:
    Average relationship strength score for the current filtered customer base.
    """
    if data_frame.empty or "Relationship_Strength_Index" not in data_frame.columns:
        return np.nan

    relationship_index = pd.to_numeric(
        data_frame["Relationship_Strength_Index"],
        errors="coerce",
    ).mean()

    return safe_np_round(relationship_index, 2)


def calculate_required_dashboard_kpis(data_frame: pd.DataFrame) -> dict:
    """
    Calculate all project-required KPIs from filtered_df.
    """
    return {
        "Engagement Retention Ratio": calculate_engagement_retention_ratio(data_frame),
        "Product Depth Index": calculate_product_depth_index(data_frame),
        "High-Balance Disengagement Rate": calculate_high_balance_disengagement_rate(data_frame),
        "Credit Card Stickiness Score": calculate_credit_card_stickiness_score(data_frame),
        "Relationship Strength Index": calculate_relationship_strength_index(data_frame),
    }


required_kpis = calculate_required_dashboard_kpis(filtered_df)
    
# =========================================================
# REBUILD STEP 1H — REQUIRED KPI SUMMARY STRIP / DISPLAY CARDS
# Displays the 5 project-required KPIs from required_kpis.
# =========================================================

def is_kpi_unavailable(value) -> bool:
    """
    Check whether KPI value is unavailable.
    Used to avoid showing raw NaN on the dashboard.
    """
    try:
        return value is None or pd.isna(value) or not np.isfinite(float(value))
    except Exception:
        return True


def format_kpi_value(
    value,
    suffix: str = "",
    decimals: int = 2,
    unavailable_label: str = "N/A",
) -> str:
    """
    Format KPI values safely for Streamlit metric cards.
    """
    if is_kpi_unavailable(value):
        return unavailable_label

    return f"{float(value):.{decimals}f}{suffix}"


def calculate_kpi_delta(
    current_value,
    benchmark_value,
    suffix: str = "",
    decimals: int = 2,
) -> str | None:
    """
    Calculate simple delta between filtered KPI and full dataset KPI.
    Returns None when comparison is not possible.
    """
    if is_kpi_unavailable(current_value) or is_kpi_unavailable(benchmark_value):
        return None

    delta_value = float(current_value) - float(benchmark_value)

    return f"{delta_value:+.{decimals}f}{suffix}"


# Full dataset KPI benchmark for comparison
benchmark_kpis = calculate_required_dashboard_kpis(df_master)


# =========================================================
# DYNAMIC KPI SUMMARY SECTION
# =========================================================

def render_kpi_summary(
    current_kpis: dict,
    baseline_kpis: dict,
) -> None:
    """
    Render the five required project KPIs using Streamlit-native components.

    Final structure:
    - Row 1: 3 large KPI cards
    - Row 2: 2 centered KPI cards
    - KPI heading rendered separately for full readability
    - st.metric label collapsed to avoid Streamlit label truncation
    - Formula reference shown once below KPI cards
    - No custom HTML is used
    """
    st.subheader("Key Performance Indicators (KPIs)")

    kpi_definitions = [
        {
            "label": "Engagement Retention Ratio",
            "key": "Engagement Retention Ratio",
            "css_key": "engagement_retention_ratio",
            "suffix": "",
            "decimals": 2,
            "delta_color": "normal",
            "help": (
                "Active Member retention rate divided by Inactive Member retention rate. "
                "Shows N/A when either Active or Inactive segment is unavailable after filtering."
            ),
        },
        {
            "label": "Product Depth Index",
            "key": "Product Depth Index",
            "css_key": "product_depth_index",
            "suffix": "",
            "decimals": 2,
            "delta_color": "normal",
            "help": (
                "Average product count among retained customers divided by overall average "
                "product count, multiplied by 100."
            ),
        },
        {
            "label": "High-Balance Disengagement Rate",
            "key": "High-Balance Disengagement Rate",
            "css_key": "high_balance_disengagement_rate",
            "suffix": "%",
            "decimals": 2,
            "delta_color": "inverse",
            "help": (
                "High Value Disengaged customers divided by total High Value customers, "
                "multiplied by 100. Lower is better for this risk KPI."
            ),
        },
        {
            "label": "Credit Card Stickiness Score",
            "key": "Credit Card Stickiness Score",
            "css_key": "credit_card_stickiness_score",
            "suffix": "%",
            "decimals": 2,
            "delta_color": "normal",
            "help": "Retention rate among customers who have a credit card.",
        },
        {
            "label": "Relationship Strength Index",
            "key": "Relationship Strength Index",
            "css_key": "relationship_strength_index",
            "suffix": "",
            "decimals": 2,
            "delta_color": "normal",
            "help": "Average Relationship Strength Index for the current filtered customer base.",
        },
    ]

    def render_single_kpi_card(kpi: dict) -> None:
        current_value = current_kpis.get(kpi["key"])
        baseline_value = baseline_kpis.get(kpi["key"])

        with st.container(key=f"kpi_card_{kpi['css_key']}"):
            st.markdown(f"#### {kpi['label']}")

            st.metric(
                label=kpi["label"],
                value=format_kpi_value(
                    current_value,
                    suffix=kpi["suffix"],
                    decimals=kpi["decimals"],
                ),
                delta=calculate_kpi_delta(
                    current_value,
                    baseline_value,
                    suffix=kpi["suffix"],
                    decimals=kpi["decimals"],
                ),
                delta_color=kpi["delta_color"],
                help=kpi["help"],
                label_visibility="collapsed",
            )

    first_row_columns = st.columns(3, gap="medium")

    for column, kpi in zip(first_row_columns, kpi_definitions[:3]):
        with column:
            render_single_kpi_card(kpi)

    st.write("")

    left_spacer, second_col_1, second_col_2, right_spacer = st.columns(
        [0.5, 1, 1, 0.5],
        gap="medium",
    )

    for column, kpi in zip([second_col_1, second_col_2], kpi_definitions[3:]):
        with column:
            render_single_kpi_card(kpi)

    if is_kpi_unavailable(current_kpis.get("Engagement Retention Ratio")):
        st.info(
            "Engagement Retention Ratio is shown as N/A because the current filters do not include both "
            "Active Member and Inactive Member segments. Select both segments to compare retention strength."
        )

    with st.expander("KPI Formula Reference", expanded=False):
        st.markdown("#### How the KPI metrics are calculated")

        st.write(
            "These KPI metrics are recalculated dynamically from the currently filtered customer base. "
            "They help compare retention quality, product depth, high-value disengagement risk, "
            "credit-card stickiness, and overall relationship strength."
        )

        formula_rows = [
            {
                "KPI": "Engagement Retention Ratio",
                "Formula": "Active Member Retention Rate ÷ Inactive Member Retention Rate",
                "Meaning": "Compares how strongly active members retain compared with inactive members.",
            },
            {
                "KPI": "Product Depth Index",
                "Formula": "Retained Avg NumOfProducts ÷ Overall Avg NumOfProducts × 100",
                "Meaning": "Shows whether retained customers hold deeper product relationships than the overall customer base.",
            },
            {
                "KPI": "High-Balance Disengagement Rate",
                "Formula": "High Value Disengaged Customers ÷ High Value Customers × 100",
                "Meaning": "Measures the risk share of valuable customers who are financially important but behaviorally disengaged.",
            },
            {
                "KPI": "Credit Card Stickiness Score",
                "Formula": "Retention Rate among customers with a credit card",
                "Meaning": "Shows how sticky credit-card ownership is for customer retention.",
            },
            {
                "KPI": "Relationship Strength Index",
                "Formula": "Average Relationship_Strength_Index",
                "Meaning": "Summarizes the overall relationship depth and strength across the filtered customer base.",
            },
        ]

        st.dataframe(
            formula_rows,
            use_container_width=True,
            hide_index=True,
        )

render_kpi_summary(
    current_kpis=required_kpis,
    baseline_kpis=benchmark_kpis,
)

# =========================================================
# REBUILD STEP 1N-V1 — PREMIUM DYNAMIC VISUAL SYSTEM HELPERS
# Reusable Plotly visual foundation for all dashboard modules.
# No CSS, no HTML, no module replacement in this step.
# =========================================================

ECB_CHART_COLORS = {
    "navy": "#003299",
    "blue": "#005BBA",
    "light_blue": "#6BB7F2",
    "soft_blue": "#DCEBFA",
    "gold": "#F7C948",
    "green": "#00875A",
    "red": "#D64545",
    "orange": "#F2994A",
    "purple": "#7C3AED",
    "teal": "#00A3A3",
    "gray": "#667085",
    "light_gray": "#EEF3F8",
    "dark_text": "#172033",
}

ENGAGEMENT_COLOR_MAP = {
    "Active Member": ECB_CHART_COLORS["blue"],
    "Inactive Member": ECB_CHART_COLORS["red"],
}

CUSTOMER_OUTCOME_COLOR_MAP = {
    "Retained Customers": ECB_CHART_COLORS["green"],
    "Churned Customers": ECB_CHART_COLORS["red"],
}

RETENTION_TIER_ORDER = [
    "Critical Retention Risk",
    "Weak Retention",
    "Moderate Retention",
    "Strong Retention",
    "Very Strong Retention",
]

RETENTION_TIER_COLOR_MAP = {
    "Critical Retention Risk": ECB_CHART_COLORS["red"],
    "Weak Retention": ECB_CHART_COLORS["orange"],
    "Moderate Retention": ECB_CHART_COLORS["gold"],
    "Strong Retention": ECB_CHART_COLORS["blue"],
    "Very Strong Retention": ECB_CHART_COLORS["green"],
}

PRODUCT_DEPTH_ORDER = [
    "Single Product",
    "Optimal Product Depth",
    "High Product Load",
]

PRODUCT_DEPTH_COLOR_MAP = {
    "Single Product": ECB_CHART_COLORS["orange"],
    "Optimal Product Depth": ECB_CHART_COLORS["blue"],
    "High Product Load": ECB_CHART_COLORS["purple"],
}

RISK_SEGMENT_ORDER = [
    "Sticky Active Customer",
    "Standard Monitoring",
    "At-Risk Premium",
    "Inactive Single-Product Risk",
    "High Product Load Risk",
]

RISK_SEGMENT_COLOR_MAP = {
    "Sticky Active Customer": ECB_CHART_COLORS["green"],
    "Standard Monitoring": ECB_CHART_COLORS["blue"],
    "At-Risk Premium": ECB_CHART_COLORS["red"],
    "Inactive Single-Product Risk": ECB_CHART_COLORS["orange"],
    "High Product Load Risk": ECB_CHART_COLORS["purple"],
}

HIGH_VALUE_COLOR_MAP = {
    "High Value Customer": ECB_CHART_COLORS["red"],
    "Standard Value Customer": ECB_CHART_COLORS["blue"],
    "High Value Disengaged": ECB_CHART_COLORS["red"],
    "High Value Engaged / Stable": ECB_CHART_COLORS["green"],
}

MISMATCH_COLOR_MAP = {
    "Salary-Balance Mismatch": ECB_CHART_COLORS["orange"],
    "No Salary-Balance Mismatch": ECB_CHART_COLORS["blue"],
}


def format_compact_number(value) -> str:
    """
    Format numbers for chart labels and hover text.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"

        value = float(value)

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"

        return f"{value:,.0f}"

    except Exception:
        return "N/A"


def format_percentage(value, decimals: int = 2) -> str:
    """
    Format percentage values safely.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"

        return f"{float(value):.{decimals}f}%"

    except Exception:
        return "N/A"


def format_currency_value(value, decimals: int = 0) -> str:
    """
    Format financial values for dashboard display.
    Currency symbol is avoided to keep the project globally neutral.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"

        return f"{float(value):,.{decimals}f}"

    except Exception:
        return "N/A"


def get_existing_category_order(
    data_frame: pd.DataFrame,
    column_name: str,
    preferred_order: list[str],
) -> list[str]:
    """
    Return only categories available in the current filtered data,
    while preserving preferred business order.
    """
    if data_frame.empty or column_name not in data_frame.columns:
        return []

    available_categories = (
        data_frame[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    ordered_categories = [
        category for category in preferred_order
        if category in available_categories
    ]

    remaining_categories = sorted(
        category for category in available_categories
        if category not in ordered_categories
    )

    return ordered_categories + remaining_categories


def apply_premium_plotly_layout(
    fig: go.Figure,
    title: str,
    height: int = 460,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """
    Apply a consistent premium institutional Plotly layout across all dashboard visuals.
    """
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 18,
                "color": ECB_CHART_COLORS["dark_text"],
            },
        },
        height=height,
        margin=dict(l=24, r=24, t=76, b=54),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title=legend_title,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=ECB_CHART_COLORS["soft_blue"],
            font_size=12,
            font_family="Arial, sans-serif",
        ),
    )

    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=False,
        zeroline=False,
        linecolor="#D0D7E2",
        tickfont=dict(color=ECB_CHART_COLORS["gray"]),
        title_font=dict(color=ECB_CHART_COLORS["gray"]),
    )

    fig.update_yaxes(
        title_text=yaxis_title,
        showgrid=True,
        gridcolor="#E6ECF3",
        zeroline=False,
        linecolor="#D0D7E2",
        tickfont=dict(color=ECB_CHART_COLORS["gray"]),
        title_font=dict(color=ECB_CHART_COLORS["gray"]),
    )

    return fig


def add_overall_churn_reference_line(
    fig: go.Figure,
    overall_churn_rate: float,
    axis_type: str = "x",
    label: str = "Overall Churn",
) -> go.Figure:
    """
    Add a reusable overall churn reference line to Plotly charts.

    axis_type:
    - "x" for vertical reference line
    - "y" for horizontal reference line
    """
    if overall_churn_rate is None or pd.isna(overall_churn_rate):
        return fig

    if axis_type == "x":
        fig.add_vline(
            x=overall_churn_rate,
            line_dash="dot",
            line_color=ECB_CHART_COLORS["gray"],
            annotation_text=f"{label}: {overall_churn_rate:.2f}%",
            annotation_position="top",
        )

    elif axis_type == "y":
        fig.add_hline(
            y=overall_churn_rate,
            line_dash="dot",
            line_color=ECB_CHART_COLORS["gray"],
            annotation_text=f"{label}: {overall_churn_rate:.2f}%",
            annotation_position="top right",
        )

    return fig


def create_empty_visual_message(module_name: str) -> None:
    """
    Standard empty-state message for dynamic filtered visuals.
    """
    st.warning(
        f"{module_name} cannot be displayed because the current filters do not "
        "contain enough records for this visual. Please widen the sidebar filters."
    )


# =========================================================
# STEP 1N-V5 — FULL 4-MODULE VISUALIZATION DIVERSITY UPGRADE
# Safe-mode replacement for the dashboard modules only.
#
# Project modules covered:
# 1. Engagement vs churn overview
# 2. Product utilization impact analysis
# 3. High-value disengaged customer detector
# 4. Retention strength scoring panels
#
# Rules:
# - Streamlit-native only
# - Plotly interactive visuals
# - No custom HTML
# - Sidebar filters remain connected through filtered_df
# - Each module uses different visualization types where meaningful
# =========================================================

MODULE_COLOR_SEQUENCE = [
    ECB_CHART_COLORS["blue"],
    ECB_CHART_COLORS["green"],
    ECB_CHART_COLORS["orange"],
    ECB_CHART_COLORS["purple"],
    ECB_CHART_COLORS["teal"],
    ECB_CHART_COLORS["gold"],
    ECB_CHART_COLORS["red"],
    ECB_CHART_COLORS["light_blue"],
]

DIVERGENT_RISK_COLORSCALE = [
    [0.00, ECB_CHART_COLORS["green"]],
    [0.35, ECB_CHART_COLORS["gold"]],
    [0.65, ECB_CHART_COLORS["orange"]],
    [1.00, ECB_CHART_COLORS["red"]],
]


def build_dynamic_color_map(
    categories: list,
    preferred_map: dict | None = None,
) -> dict:
    """
    Create a stable dynamic color map for available categories.
    Existing business colors are respected first, then fallback colors are assigned.
    """
    preferred_map = preferred_map or {}
    color_map = {}

    fallback_index = 0

    for category in categories:
        category_text = str(category)

        if category_text in preferred_map:
            color_map[category_text] = preferred_map[category_text]
        else:
            color_map[category_text] = MODULE_COLOR_SEQUENCE[
                fallback_index % len(MODULE_COLOR_SEQUENCE)
            ]
            fallback_index += 1

    return color_map


def format_display_number(value) -> str:
    """
    Format whole-number values for tables and insights.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"


def format_display_decimal(value, decimals: int = 2) -> str:
    """
    Format decimal values for tables and insights.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def format_display_percent(value, decimals: int = 2) -> str:
    """
    Format percentage values for tables and insights.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "N/A"


def render_business_explanation_block(
    title: str,
    what: str,
    why: str,
    how: str,
    when: str,
) -> None:
    """
    Streamlit-native explanation block for each visual.
    """
    with st.expander(title, expanded=False):
        st.markdown("**What this shows**")
        st.write(what)

        st.markdown("**Why it matters**")
        st.write(why)

        st.markdown("**How to read it**")
        st.write(how)

        st.markdown("**When to take action**")
        st.write(when)


def create_rate_summary(
    data_frame: pd.DataFrame,
    group_columns: list[str],
    extra_aggregations: dict | None = None,
) -> pd.DataFrame:
    """
    Reusable grouped customer, churn, and retention summary.
    """
    required_columns = ["CustomerId", "Exited"] + group_columns

    if (
        data_frame.empty
        or any(column not in data_frame.columns for column in required_columns)
    ):
        return pd.DataFrame()

    working_df = data_frame.copy()
    working_df["Exited"] = pd.to_numeric(
        working_df["Exited"],
        errors="coerce",
    ).fillna(0)

    aggregations = {
        "Customer_Count": ("CustomerId", "nunique"),
        "Churned_Customers": ("Exited", "sum"),
    }

    if extra_aggregations:
        aggregations.update(extra_aggregations)

    summary = (
        working_df
        .groupby(group_columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    summary["Churned_Customers"] = summary["Churned_Customers"].astype(int)
    summary["Retained_Customers"] = (
        summary["Customer_Count"] - summary["Churned_Customers"]
    )

    summary["Churn_Rate"] = summary.apply(
        lambda row: safe_divide(
            row["Churned_Customers"],
            row["Customer_Count"],
            100,
        ),
        axis=1,
    )

    summary["Retention_Rate"] = 100 - summary["Churn_Rate"]

    numeric_columns = summary.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        if column not in ["Customer_Count", "Churned_Customers", "Retained_Customers"]:
            summary[column] = summary[column].round(2)

    return summary


def create_churn_heatmap(
    matrix_summary: pd.DataFrame,
    row_column: str,
    column_column: str,
    row_order: list[str],
    column_order: list[str],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 450,
) -> go.Figure:
    """
    Reusable churn-rate heatmap with text labels.
    """
    available_rows = get_existing_category_order(
        matrix_summary,
        row_column,
        row_order,
    )

    available_columns = get_existing_category_order(
        matrix_summary,
        column_column,
        column_order,
    )

    churn_matrix = (
        matrix_summary
        .pivot_table(
            index=row_column,
            columns=column_column,
            values="Churn_Rate",
            aggfunc="mean",
        )
        .reindex(index=available_rows, columns=available_columns)
    )

    customer_matrix = (
        matrix_summary
        .pivot_table(
            index=row_column,
            columns=column_column,
            values="Customer_Count",
            aggfunc="sum",
        )
        .reindex(index=available_rows, columns=available_columns)
    )

    text_matrix = pd.DataFrame(
        "",
        index=churn_matrix.index,
        columns=churn_matrix.columns,
        dtype=object,
    )

    for row_label in churn_matrix.index:
        for column_label in churn_matrix.columns:
            churn_value = churn_matrix.loc[row_label, column_label]
            customer_value = customer_matrix.loc[row_label, column_label]

            if pd.isna(churn_value) or pd.isna(customer_value):
                text_matrix.loc[row_label, column_label] = "No Data"
            else:
                text_matrix.loc[row_label, column_label] = (
                    f"{churn_value:.1f}%<br>{int(customer_value):,} customers"
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=churn_matrix.values,
            x=churn_matrix.columns,
            y=churn_matrix.index,
            text=text_matrix.values,
            customdata=customer_matrix.values,
            texttemplate="%{text}",
            colorscale=DIVERGENT_RISK_COLORSCALE,
            zmin=0,
            zmax=100,
            colorbar=dict(
                title="Churn Rate (%)",
                ticksuffix="%",
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{xaxis_title}: " + "%{x}<br>"
                "Churn Rate: %{z:.2f}%<br>"
                "Customers: %{customdata:,}<extra></extra>"
            ),
        )
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title=title,
        height=height,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        show_legend=False,
    )

    fig.update_xaxes(tickangle=-22)

    fig.update_layout(
        margin=dict(l=24, r=80, t=76, b=95),
    )

    return fig


# =========================================================
# MODULE 1 — ENGAGEMENT VS CHURN OVERVIEW
# Visualization mix:
# 1. Grouped bar
# 2. 100% stacked churn contribution bar
# 3. Heatmap risk matrix
# =========================================================

def create_engagement_churn_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement-level churn and retention summary.
    """
    extra_aggregations = {
        "Avg_Product_Count": ("NumOfProducts", "mean"),
        "Avg_Balance": ("Balance", "mean"),
        "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
    }

    required_columns = list(extra_aggregations.keys())

    if any(
        source_column not in data_frame.columns
        for source_column in ["NumOfProducts", "Balance", "Retention_Strength_Score"]
    ):
        return pd.DataFrame()

    summary = create_rate_summary(
        data_frame=data_frame,
        group_columns=["Engagement_Status"],
        extra_aggregations=extra_aggregations,
    )

    if summary.empty:
        return summary

    return summary[
        [
            "Engagement_Status",
            "Customer_Count",
            "Retained_Customers",
            "Churned_Customers",
            "Retention_Rate",
            "Churn_Rate",
            "Avg_Product_Count",
            "Avg_Balance",
            "Avg_Retention_Strength_Score",
        ]
    ]


def create_engagement_outcome_chart(engagement_summary: pd.DataFrame) -> go.Figure:
    """
    Premium grouped bar chart: retained vs churned by engagement segment.

    V2 layout direction:
    - Removes bottom/right legend to prevent truncation.
    - Adds a compact color guide annotation.
    - Uses direct value labels above bars.
    - Keeps chart clean, stable, and responsive.
    """
    chart_data = engagement_summary.melt(
        id_vars="Engagement_Status",
        value_vars=["Retained_Customers", "Churned_Customers"],
        var_name="Customer_Status",
        value_name="Customers",
    )

    chart_data["Customer_Status"] = chart_data["Customer_Status"].replace(
        {
            "Retained_Customers": "Retained Customers",
            "Churned_Customers": "Churned Customers",
        }
    )

    category_order = get_existing_category_order(
        engagement_summary,
        "Engagement_Status",
        ["Active Member", "Inactive Member"],
    )

    fig = px.bar(
        chart_data,
        x="Engagement_Status",
        y="Customers",
        color="Customer_Status",
        barmode="group",
        text="Customers",
        category_orders={
            "Engagement_Status": category_order,
            "Customer_Status": ["Retained Customers", "Churned Customers"],
        },
        color_discrete_map=CUSTOMER_OUTCOME_COLOR_MAP,
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        textfont=dict(
            size=13,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Outcome: %{fullData.name}<br>"
            "Customers: %{y:,}<extra></extra>"
        ),
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Engagement Outcome Split: Retained vs Churned Customers",
        height=500,
        xaxis_title="Engagement Status",
        yaxis_title="Customer Count",
        legend_title=None,
    )

    fig.update_layout(
        showlegend=False,
        bargap=0.34,
        bargroupgap=0.12,
        margin=dict(l=82, r=62, t=92, b=86),
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )

    fig.add_annotation(
        x=1,
        y=1.12,
        xref="paper",
        yref="paper",
        text="<span style='color:#00875A'>■ Retained</span> &nbsp;&nbsp; <span style='color:#D64545'>■ Churned</span>",
        showarrow=False,
        align="right",
        font=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
    )

    fig.update_xaxes(
        automargin=True,
        tickangle=0,
        title_standoff=18,
        tickfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        title_font=dict(
            size=13,
            color=ECB_CHART_COLORS["gray"],
        ),
    )

    fig.update_yaxes(
        automargin=True,
        title_standoff=18,
        tickfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        title_font=dict(
            size=13,
            color=ECB_CHART_COLORS["gray"],
        ),
    )

    return fig


def create_engagement_churn_contribution_donut(
    engagement_summary: pd.DataFrame,
) -> go.Figure:
    """
    Premium 100% stacked bar: churn contribution by engagement segment.

    V2 layout direction:
    - Removes legend because segment labels are already inside the bar.
    - Prevents bottom legend/axis collision.
    - Keeps the chart compact, centered, and business-readable.
    """
    chart_data = engagement_summary[
        engagement_summary["Churned_Customers"] > 0
    ].copy()

    if chart_data.empty:
        chart_data = engagement_summary.copy()

    chart_data["Engagement_Status"] = chart_data["Engagement_Status"].astype(str)
    total_churned = chart_data["Churned_Customers"].sum()

    chart_data["Contribution_Share"] = chart_data["Churned_Customers"].apply(
        lambda value: safe_divide(value, total_churned, 100)
    )

    color_map = build_dynamic_color_map(
        chart_data["Engagement_Status"].tolist(),
        ENGAGEMENT_COLOR_MAP,
    )

    fig = go.Figure()

    for _, row in chart_data.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Contribution_Share"]],
                y=["Churn Contribution"],
                orientation="h",
                name=row["Engagement_Status"],
                marker=dict(
                    color=color_map.get(
                        row["Engagement_Status"],
                        ECB_CHART_COLORS["blue"],
                    ),
                    line=dict(width=0),
                ),
                text=[
                    f"{row['Engagement_Status']}<br>"
                    f"{row['Contribution_Share']:.1f}% | {int(row['Churned_Customers']):,}"
                ],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=13,
                    color=ECB_CHART_COLORS["dark_text"],
                ),
                hovertemplate=(
                    f"<b>{row['Engagement_Status']}</b><br>"
                    f"Churned Customers: {int(row['Churned_Customers']):,}<br>"
                    f"Contribution Share: {row['Contribution_Share']:.2f}%<br>"
                    f"Segment Churn Rate: {row['Churn_Rate']:.2f}%<extra></extra>"
                ),
            )
        )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Churn Contribution Share by Engagement Segment",
        height=360,
        xaxis_title="Share of Churned Customers (%)",
        yaxis_title=None,
        legend_title=None,
    )

    fig.update_layout(
        showlegend=False,
        barmode="stack",
        bargap=0.68,
        margin=dict(l=54, r=54, t=88, b=76),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )

    fig.add_annotation(
        x=0,
        y=1.12,
        xref="paper",
        yref="paper",
        text="Higher share = larger contribution to total churn",
        showarrow=False,
        align="left",
        font=dict(
            size=12,
            color=ECB_CHART_COLORS["gray"],
        ),
    )

    fig.update_xaxes(
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E6ECF3",
        automargin=True,
        title_standoff=18,
        tickfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        title_font=dict(
            size=13,
            color=ECB_CHART_COLORS["gray"],
        ),
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )

    return fig


def create_engagement_matrix_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement Status × Retention Strength Tier churn matrix summary.
    """
    return create_rate_summary(
        data_frame=data_frame,
        group_columns=["Engagement_Status", "Retention_Strength_Tier"],
    )


def create_engagement_matrix_chart(matrix_summary: pd.DataFrame) -> go.Figure:
    """
    Premium heatmap: engagement × retention strength tier.

    V2 layout direction:
    - Gives heatmap more visual dominance.
    - Keeps x-axis labels readable.
    - Keeps colorbar compact and visible.
    - Avoids CSS-dependent Plotly fixes.
    """
    fig = create_churn_heatmap(
        matrix_summary=matrix_summary,
        row_column="Engagement_Status",
        column_column="Retention_Strength_Tier",
        row_order=["Active Member", "Inactive Member"],
        column_order=RETENTION_TIER_ORDER,
        title="Engagement × Retention Tier Churn Risk Matrix",
        xaxis_title="Retention Strength Tier",
        yaxis_title="Engagement Status",
        height=560,
    )

    fig.update_layout(
        height=560,
        margin=dict(l=118, r=118, t=82, b=138),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        tickangle=-22,
        automargin=True,
        title_standoff=30,
        tickfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        title_font=dict(
            size=13,
            color=ECB_CHART_COLORS["gray"],
        ),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        automargin=True,
        title_standoff=28,
        tickfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        title_font=dict(
            size=13,
            color=ECB_CHART_COLORS["gray"],
        ),
        showgrid=False,
        zeroline=False,
    )

    fig.update_traces(
        textfont=dict(
            size=12,
            color=ECB_CHART_COLORS["dark_text"],
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Retention Tier: %{x}<br>"
            "Churn Rate: %{z:.2f}%<extra></extra>"
        ),
        colorbar=dict(
            title=dict(
                text="Churn Rate (%)",
                side="right",
                font=dict(
                    size=12,
                    color=ECB_CHART_COLORS["dark_text"],
                ),
            ),
            tickfont=dict(
                size=11,
                color=ECB_CHART_COLORS["dark_text"],
            ),
            ticksuffix="%",
            thickness=16,
            len=0.68,
            x=1.02,
            y=0.50,
            outlinewidth=0,
        ),
        selector=dict(type="heatmap"),
    )

    return fig


def calculate_engagement_decision_signals(
    engagement_summary: pd.DataFrame,
    matrix_summary: pd.DataFrame,
) -> dict:
    """
    Dynamic business signals for Module 1.
    """
    signals = {
        "active_row": None,
        "inactive_row": None,
        "highest_churn_row": None,
        "strongest_retention_row": None,
        "retention_gap": np.nan,
        "churn_gap": np.nan,
        "comparison_available": False,
        "highest_rate_segment": None,
        "highest_impact_segment": None,
    }

    if engagement_summary.empty:
        return signals

    for status in ["Active Member", "Inactive Member"]:
        matching_rows = engagement_summary[
            engagement_summary["Engagement_Status"].astype(str).eq(status)
        ]

        if not matching_rows.empty:
            if status == "Active Member":
                signals["active_row"] = matching_rows.iloc[0]
            else:
                signals["inactive_row"] = matching_rows.iloc[0]

    signals["highest_churn_row"] = engagement_summary.sort_values(
        ["Churn_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    signals["strongest_retention_row"] = engagement_summary.sort_values(
        ["Retention_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    if signals["active_row"] is not None and signals["inactive_row"] is not None:
        signals["retention_gap"] = (
            float(signals["active_row"]["Retention_Rate"])
            - float(signals["inactive_row"]["Retention_Rate"])
        )
        signals["churn_gap"] = (
            float(signals["inactive_row"]["Churn_Rate"])
            - float(signals["active_row"]["Churn_Rate"])
        )
        signals["comparison_available"] = True

    if not matrix_summary.empty:
        signals["highest_rate_segment"] = matrix_summary.sort_values(
            ["Churn_Rate", "Customer_Count"],
            ascending=[False, False],
        ).iloc[0]

        signals["highest_impact_segment"] = matrix_summary.sort_values(
            ["Churned_Customers", "Customer_Count", "Churn_Rate"],
            ascending=[False, False, False],
        ).iloc[0]

    return signals


def create_engagement_display_table(engagement_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Business-readable Module 1 table.
    """
    if engagement_summary.empty:
        return pd.DataFrame()

    display_table = engagement_summary.rename(
        columns={
            "Engagement_Status": "Engagement Segment",
            "Customer_Count": "Customers",
            "Retained_Customers": "Retained",
            "Churned_Customers": "Churned",
            "Retention_Rate": "Retention Rate",
            "Churn_Rate": "Churn Rate",
            "Avg_Product_Count": "Avg Products",
            "Avg_Balance": "Avg Balance",
            "Avg_Retention_Strength_Score": "Avg Retention Score",
        }
    ).copy()

    for column in ["Customers", "Retained", "Churned"]:
        display_table[column] = display_table[column].map(format_display_number)

    for column in ["Retention Rate", "Churn Rate"]:
        display_table[column] = display_table[column].map(format_display_percent)

    for column in ["Avg Products", "Avg Retention Score"]:
        display_table[column] = display_table[column].map(format_display_decimal)

    display_table["Avg Balance"] = display_table["Avg Balance"].map(format_display_number)

    return display_table


def render_engagement_recommendations(signals: dict) -> None:
    """
    Dynamic business recommendation for Module 1.
    """
    st.markdown("#### Business Interpretation & Retention Actions")

    highest_churn_row = signals["highest_churn_row"]
    strongest_retention_row = signals["strongest_retention_row"]

    if highest_churn_row is None or strongest_retention_row is None:
        st.info("Engagement recommendation is unavailable for the current filtered view.")
        return

    st.success(
        f"Engagement is acting as a measurable retention signal. "
        f"**{strongest_retention_row['Engagement_Status']}** has the strongest retention rate at "
        f"**{strongest_retention_row['Retention_Rate']:.2f}%**, while "
        f"**{highest_churn_row['Engagement_Status']}** has the highest churn rate at "
        f"**{highest_churn_row['Churn_Rate']:.2f}%**."
    )

    if signals["comparison_available"]:
        if signals["churn_gap"] > 5:
            st.warning(
                f"Inactive churn lift is **{signals['churn_gap']:.2f} percentage points**. "
                "Inactive customers should be treated as an early-warning retention segment."
            )
        else:
            st.info(
                f"The active vs inactive churn gap is **{signals['churn_gap']:.2f} percentage points** "
                "in the current filtered view."
            )
    else:
        st.info(
            "Select both Active Member and Inactive Member in the sidebar to compare engagement-driven churn lift."
        )

    if signals["highest_rate_segment"] is not None and signals["highest_impact_segment"] is not None:
        rate_segment = signals["highest_rate_segment"]
        impact_segment = signals["highest_impact_segment"]

        col_1, col_2 = st.columns(2, gap="medium")

        with col_1:
            st.markdown("**Highest Rate Risk Segment**")
            st.write(
                f"**{rate_segment['Engagement_Status']} + {rate_segment['Retention_Strength_Tier']}** "
                f"has **{rate_segment['Churn_Rate']:.2f}% churn** across "
                f"**{int(rate_segment['Customer_Count']):,} customers**."
            )

        with col_2:
            st.markdown("**Highest Business Impact Segment**")
            st.write(
                f"**{impact_segment['Engagement_Status']} + {impact_segment['Retention_Strength_Tier']}** "
                f"has the largest churn impact with **{int(impact_segment['Churned_Customers']):,} churned customers** "
                f"out of **{int(impact_segment['Customer_Count']):,} customers**."
            )

        st.error(
            f"First engagement-retention priority: **{impact_segment['Engagement_Status']} + "
            f"{impact_segment['Retention_Strength_Tier']}**. Start with reactivation nudges, "
            "relationship outreach, product education, and campaign-cycle monitoring."
        )


def render_engagement_vs_churn_module(data_frame: pd.DataFrame) -> None:
    """
    Render Module 1: Engagement vs Churn Overview.
    """
    st.subheader("Engagement vs Churn Overview")

    st.caption(
        "Guideline focus: Evaluate the relationship between engagement behavior and churn, "
        "identify inactive risk, and support engagement-driven retention strategy."
    )

    st.write(
        "This module shows whether active customers retain better than inactive customers, "
        "which engagement segment contributes most to churn, and which engagement-retention "
        "micro-segment should receive first campaign priority."
    )

    engagement_summary = create_engagement_churn_summary(data_frame)

    if engagement_summary.empty:
        create_empty_visual_message("Engagement vs Churn Overview")
        return

    matrix_summary = create_engagement_matrix_summary(data_frame)
    signals = calculate_engagement_decision_signals(
        engagement_summary=engagement_summary,
        matrix_summary=matrix_summary,
    )

    total_customers = int(engagement_summary["Customer_Count"].sum())
    retained_customers = int(engagement_summary["Retained_Customers"].sum())
    churned_customers = int(engagement_summary["Churned_Customers"].sum())
    overall_retention_rate = safe_divide(retained_customers, total_customers, 100)
    overall_churn_rate = safe_divide(churned_customers, total_customers, 100)

    st.markdown("#### Current Engagement Retention Snapshot")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

    with metric_col_1:
        st.metric("Customers Analyzed", f"{total_customers:,}")

    with metric_col_2:
        st.metric("Retained Customers", f"{retained_customers:,}")

    with metric_col_3:
        st.metric("Churned Customers", f"{churned_customers:,}")

    with metric_col_4:
        st.metric(
            "Overall Retention Rate",
            f"{overall_retention_rate:.2f}%",
            delta=f"Churn {overall_churn_rate:.2f}%",
            delta_color="inverse",
        )

    st.markdown("#### Engagement Decision Signals")

    signal_col_1, signal_col_2, signal_col_3, signal_col_4 = st.columns(4, gap="medium")

    with signal_col_1:
        st.metric(
            "Retention Gap",
            f"{signals['retention_gap']:.2f} pp" if not pd.isna(signals["retention_gap"]) else "N/A",
        )

    with signal_col_2:
        st.metric(
            "Inactive Churn Lift",
            f"{signals['churn_gap']:.2f} pp" if not pd.isna(signals["churn_gap"]) else "N/A",
        )

    with signal_col_3:
        st.metric(
            "Highest Churn Segment",
            str(signals["highest_churn_row"]["Engagement_Status"]),
            delta=f"{signals['highest_churn_row']['Churn_Rate']:.2f}% churn",
            delta_color="inverse",
        )

    with signal_col_4:
        st.metric(
            "Best Retention Segment",
            str(signals["strongest_retention_row"]["Engagement_Status"]),
            delta=f"{signals['strongest_retention_row']['Retention_Rate']:.2f}% retained",
        )

    st.markdown("#### Engagement Outcome Split")

    st.plotly_chart(
        create_engagement_outcome_chart(engagement_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 1: Engagement Outcome Split",
        what="This grouped bar chart compares retained and churned customers across engagement segments.",
        why="It verifies whether engagement behavior is associated with stronger customer retention.",
        how="Compare retained and churned bars inside each engagement segment. Higher churn means higher retention intervention need.",
        when="Take action when inactive churn is meaningfully higher or when a filtered segment shows rising churn volume.",
    )

    st.markdown("#### Churn Contribution Share")

    st.plotly_chart(
        create_engagement_churn_contribution_donut(engagement_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 2: Churn Contribution Share",
        what="This 100% stacked bar shows which engagement segment contributes the largest share of churned customers.",
        why="A segment can have high churn rate but low volume. Churn contribution helps prioritize business impact.",
        how="Use the full-width segments to compare churn contribution share, then validate priority with churn rate and customer volume.",
        when="Take action when one engagement segment dominates churn contribution after filters are applied.",
    )

    st.markdown("#### Engagement × Retention Tier Risk Matrix")

    if matrix_summary.empty:
        create_empty_visual_message("Engagement Risk Matrix")
    else:
        st.plotly_chart(
            create_engagement_matrix_chart(matrix_summary),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        render_business_explanation_block(
            title="How to read Visual 3: Engagement × Retention Tier Risk Matrix",
            what="This heatmap combines engagement status with retention strength tier to identify micro-segment churn risk.",
            why="It shows where engagement behavior and retention weakness overlap, which improves campaign targeting.",
            how="Use red cells for high risk, green cells for safer zones, and customer counts to judge business impact.",
            when="Prioritize cells with high churn rate and meaningful customer count.",
        )

    st.markdown("#### Engagement-Level Summary Table")

    st.dataframe(
        create_engagement_display_table(engagement_summary),
        use_container_width=True,
        hide_index=True,
    )

    render_engagement_recommendations(signals)


# =========================================================
# MODULE 2 — PRODUCT UTILIZATION IMPACT ANALYSIS
# Visualization mix:
# 1. Combo chart
# 2. 100% stacked horizontal bar
# 3. Bubble scatter
# 4. Heatmap risk matrix
# =========================================================

def create_product_utilization_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Product-count-level churn and retention summary.
    """
    extra_aggregations = {
        "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
        "Avg_Relationship_Strength_Index": ("Relationship_Strength_Index", "mean"),
    }

    summary = create_rate_summary(
        data_frame=data_frame,
        group_columns=["NumOfProducts"],
        extra_aggregations=extra_aggregations,
    )

    if summary.empty:
        return summary

    summary["NumOfProducts"] = pd.to_numeric(
        summary["NumOfProducts"],
        errors="coerce",
    ).astype(int)

    return summary.sort_values("NumOfProducts")


def create_product_count_combo_chart(product_summary: pd.DataFrame) -> go.Figure:
    """
    Combo chart: customer volume + retention/churn rates by product count.
    """
    chart_data = product_summary.copy()
    chart_data["Product_Count_Label"] = chart_data["NumOfProducts"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_data["Product_Count_Label"],
            y=chart_data["Customer_Count"],
            name="Customer Count",
            text=chart_data["Customer_Count"],
            texttemplate="%{text:,}",
            textposition="outside",
            marker_color=ECB_CHART_COLORS["blue"],
            yaxis="y",
            hovertemplate="<b>%{x} Products</b><br>Customers: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data["Product_Count_Label"],
            y=chart_data["Retention_Rate"],
            name="Retention Rate",
            mode="lines+markers+text",
            text=chart_data["Retention_Rate"].map(lambda value: f"{value:.1f}%"),
            textposition="top center",
            line=dict(width=3, color=ECB_CHART_COLORS["green"]),
            marker=dict(size=9),
            yaxis="y2",
            hovertemplate="<b>%{x} Products</b><br>Retention Rate: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data["Product_Count_Label"],
            y=chart_data["Churn_Rate"],
            name="Churn Rate",
            mode="lines+markers+text",
            text=chart_data["Churn_Rate"].map(lambda value: f"{value:.1f}%"),
            textposition="bottom center",
            line=dict(width=3, color=ECB_CHART_COLORS["red"]),
            marker=dict(size=9),
            yaxis="y2",
            hovertemplate="<b>%{x} Products</b><br>Churn Rate: %{y:.2f}%<extra></extra>",
        )
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Product Count Impact on Retention and Churn",
        height=460,
        xaxis_title="Number of Products",
        yaxis_title="Customer Count",
        legend_title="Metric",
    )

    fig.update_xaxes(type="category")
    fig.update_layout(
        yaxis2=dict(
            title="Rate (%)",
            overlaying="y",
            side="right",
            rangemode="tozero",
            ticksuffix="%",
            showgrid=False,
        ),
        bargap=0.35,
        margin=dict(l=24, r=72, t=76, b=60),
    )

    return fig


def create_single_multi_product_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Single-product vs multi-product summary.
    """
    required_columns = [
        "CustomerId",
        "NumOfProducts",
        "Exited",
        "Retention_Strength_Score",
        "Relationship_Strength_Index",
    ]

    if (
        data_frame.empty
        or any(column not in data_frame.columns for column in required_columns)
    ):
        return pd.DataFrame()

    working_df = data_frame.copy()
    working_df["NumOfProducts"] = pd.to_numeric(
        working_df["NumOfProducts"],
        errors="coerce",
    )
    working_df["Exited"] = pd.to_numeric(
        working_df["Exited"],
        errors="coerce",
    ).fillna(0)

    working_df["Product_Relationship_Type"] = np.where(
        working_df["NumOfProducts"].eq(1),
        "Single-Product Customers",
        "Multi-Product Customers",
    )

    summary = create_rate_summary(
        data_frame=working_df,
        group_columns=["Product_Relationship_Type"],
        extra_aggregations={
            "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
            "Avg_Relationship_Strength_Index": ("Relationship_Strength_Index", "mean"),
        },
    )

    if summary.empty:
        return summary

    summary["Product_Relationship_Type"] = pd.Categorical(
        summary["Product_Relationship_Type"],
        categories=["Single-Product Customers", "Multi-Product Customers"],
        ordered=True,
    )

    return summary.sort_values("Product_Relationship_Type")


def create_single_multi_stacked_chart(product_type_summary: pd.DataFrame) -> go.Figure:
    """
    100% horizontal stacked bar: retained/churn share.
    """
    chart_data = product_type_summary.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=chart_data["Product_Relationship_Type"].astype(str),
            x=chart_data["Retention_Rate"],
            name="Retention Rate",
            orientation="h",
            text=chart_data["Retention_Rate"].map(lambda value: f"{value:.1f}%"),
            textposition="inside",
            marker_color=ECB_CHART_COLORS["green"],
            hovertemplate="<b>%{y}</b><br>Retention Rate: %{x:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=chart_data["Product_Relationship_Type"].astype(str),
            x=chart_data["Churn_Rate"],
            name="Churn Rate",
            orientation="h",
            text=chart_data["Churn_Rate"].map(lambda value: f"{value:.1f}%"),
            textposition="inside",
            marker_color=ECB_CHART_COLORS["red"],
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.2f}%<extra></extra>",
        )
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Single vs Multi-Product Retention Split",
        height=350,
        xaxis_title="Customer Outcome Share (%)",
        yaxis_title=None,
        legend_title="Customer Outcome",
    )

    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        margin=dict(l=24, r=24, t=76, b=50),
    )

    return fig


def create_product_strength_bubble_chart(product_summary: pd.DataFrame) -> go.Figure:
    """
    Bubble scatter: product count vs relationship strength, bubble size = customers.
    """
    chart_data = product_summary.copy()
    chart_data["Product_Count_Label"] = chart_data["NumOfProducts"].astype(str)

    fig = px.scatter(
        chart_data,
        x="Avg_Relationship_Strength_Index",
        y="Avg_Retention_Strength_Score",
        size="Customer_Count",
        color="Churn_Rate",
        text="Product_Count_Label",
        color_continuous_scale=[
            ECB_CHART_COLORS["green"],
            ECB_CHART_COLORS["gold"],
            ECB_CHART_COLORS["orange"],
            ECB_CHART_COLORS["red"],
        ],
        hover_data={
            "Customer_Count": ":,",
            "Retention_Rate": ":.2f",
            "Churn_Rate": ":.2f",
            "NumOfProducts": True,
            "Avg_Relationship_Strength_Index": ":.2f",
            "Avg_Retention_Strength_Score": ":.2f",
        },
    )

    fig.update_traces(
        textposition="middle center",
        marker=dict(
            sizemode="area",
            opacity=0.78,
            line=dict(width=1, color="white"),
        ),
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Product Count Relationship Strength Bubble Map",
        height=460,
        xaxis_title="Avg Relationship Strength Index",
        yaxis_title="Avg Retention Strength Score",
        show_legend=False,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Churn Rate (%)",
            ticksuffix="%",
        )
    )

    return fig


def create_product_depth_matrix_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Product depth × retention tier matrix summary.
    """
    return create_rate_summary(
        data_frame=data_frame,
        group_columns=["Product_Depth_Segment", "Retention_Strength_Tier"],
    )


def create_product_depth_matrix_chart(matrix_summary: pd.DataFrame) -> go.Figure:
    """
    Heatmap: product depth × retention tier.
    """
    return create_churn_heatmap(
        matrix_summary=matrix_summary,
        row_column="Product_Depth_Segment",
        column_column="Retention_Strength_Tier",
        row_order=PRODUCT_DEPTH_ORDER,
        column_order=RETENTION_TIER_ORDER,
        title="Product Depth × Retention Tier Churn Risk Matrix",
        xaxis_title="Retention Strength Tier",
        yaxis_title="Product Depth Segment",
        height=450,
    )


def calculate_product_decision_signals(
    product_summary: pd.DataFrame,
    product_type_summary: pd.DataFrame,
    product_matrix_summary: pd.DataFrame,
) -> dict:
    """
    Dynamic Module 2 business signals.
    """
    signals = {
        "strongest_product_row": None,
        "highest_churn_product_row": None,
        "single_multi_gap": np.nan,
        "best_relationship_type": None,
        "weakest_relationship_type": None,
        "highest_depth_rate_segment": None,
        "highest_depth_impact_segment": None,
    }

    if product_summary.empty:
        return signals

    signals["strongest_product_row"] = product_summary.sort_values(
        ["Retention_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    signals["highest_churn_product_row"] = product_summary.sort_values(
        ["Churn_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    if not product_type_summary.empty:
        signals["best_relationship_type"] = product_type_summary.sort_values(
            ["Retention_Rate", "Customer_Count"],
            ascending=[False, False],
        ).iloc[0]

        signals["weakest_relationship_type"] = product_type_summary.sort_values(
            ["Churn_Rate", "Customer_Count"],
            ascending=[False, False],
        ).iloc[0]

        single_row = product_type_summary[
            product_type_summary["Product_Relationship_Type"].astype(str).eq("Single-Product Customers")
        ]

        multi_row = product_type_summary[
            product_type_summary["Product_Relationship_Type"].astype(str).eq("Multi-Product Customers")
        ]

        if not single_row.empty and not multi_row.empty:
            signals["single_multi_gap"] = (
                float(multi_row.iloc[0]["Retention_Rate"])
                - float(single_row.iloc[0]["Retention_Rate"])
            )

    if not product_matrix_summary.empty:
        signals["highest_depth_rate_segment"] = product_matrix_summary.sort_values(
            ["Churn_Rate", "Customer_Count"],
            ascending=[False, False],
        ).iloc[0]

        signals["highest_depth_impact_segment"] = product_matrix_summary.sort_values(
            ["Churned_Customers", "Customer_Count", "Churn_Rate"],
            ascending=[False, False, False],
        ).iloc[0]

    return signals


def create_product_display_table(product_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Business-readable Module 2 table.
    """
    if product_summary.empty:
        return pd.DataFrame()

    display_table = product_summary.rename(
        columns={
            "NumOfProducts": "Products",
            "Customer_Count": "Customers",
            "Retained_Customers": "Retained",
            "Churned_Customers": "Churned",
            "Retention_Rate": "Retention Rate",
            "Churn_Rate": "Churn Rate",
            "Avg_Retention_Strength_Score": "Avg Retention Score",
            "Avg_Relationship_Strength_Index": "Avg Relationship Index",
        }
    ).copy()

    for column in ["Products", "Customers", "Retained", "Churned"]:
        display_table[column] = display_table[column].map(format_display_number)

    for column in ["Retention Rate", "Churn Rate"]:
        display_table[column] = display_table[column].map(format_display_percent)

    for column in ["Avg Retention Score", "Avg Relationship Index"]:
        display_table[column] = display_table[column].map(format_display_decimal)

    return display_table


def render_product_recommendations(signals: dict) -> None:
    """
    Dynamic product utilization recommendation.
    """
    st.markdown("#### Business Interpretation & Product Strategy Actions")

    strongest_row = signals["strongest_product_row"]
    highest_churn_row = signals["highest_churn_product_row"]

    if strongest_row is None or highest_churn_row is None:
        st.info("Product utilization recommendation is unavailable for the current filtered view.")
        return

    st.success(
        f"Customers with **{int(strongest_row['NumOfProducts'])} products** show the strongest retention "
        f"at **{strongest_row['Retention_Rate']:.2f}%**, while customers with "
        f"**{int(highest_churn_row['NumOfProducts'])} products** show the highest churn at "
        f"**{highest_churn_row['Churn_Rate']:.2f}%**."
    )

    if not pd.isna(signals["single_multi_gap"]):
        if signals["single_multi_gap"] > 0:
            st.info(
                f"Multi-product customers retain **{signals['single_multi_gap']:.2f} percentage points** "
                "better than single-product customers. Product bundling is supported in this filtered view."
            )
        else:
            st.warning(
                f"Multi-product retention is not outperforming single-product retention. "
                f"Current gap: **{signals['single_multi_gap']:.2f} percentage points**."
            )

    if signals["highest_depth_rate_segment"] is not None and signals["highest_depth_impact_segment"] is not None:
        rate_segment = signals["highest_depth_rate_segment"]
        impact_segment = signals["highest_depth_impact_segment"]

        col_1, col_2 = st.columns(2, gap="medium")

        with col_1:
            st.markdown("**Highest Product-Depth Rate Risk**")
            st.write(
                f"**{rate_segment['Product_Depth_Segment']} + {rate_segment['Retention_Strength_Tier']}** "
                f"has **{rate_segment['Churn_Rate']:.2f}% churn** across "
                f"**{int(rate_segment['Customer_Count']):,} customers**."
            )

        with col_2:
            st.markdown("**Highest Product-Depth Business Impact**")
            st.write(
                f"**{impact_segment['Product_Depth_Segment']} + {impact_segment['Retention_Strength_Tier']}** "
                f"has **{int(impact_segment['Churned_Customers']):,} churned customers** out of "
                f"**{int(impact_segment['Customer_Count']):,} customers**."
            )

        st.error(
            f"First product-utilization priority: **{impact_segment['Product_Depth_Segment']} + "
            f"{impact_segment['Retention_Strength_Tier']}**. Use product education, product-fit review, "
            "relevant bundling, and relationship-deepening outreach."
        )


def render_product_utilization_module(data_frame: pd.DataFrame) -> None:
    """
    Render Module 2: Product Utilization Impact Analysis.
    """
    st.subheader("Product Utilization Impact Analysis")

    st.caption(
        "Guideline focus: Measure retention impact of product count and product mix, "
        "compare single-product versus multi-product behavior, and support product bundling decisions."
    )

    st.write(
        "This module checks whether product depth improves loyalty, whether some product-count groups create churn risk, "
        "and where product strategy should focus first."
    )

    product_summary = create_product_utilization_summary(data_frame)

    if product_summary.empty:
        create_empty_visual_message("Product Utilization Impact Analysis")
        return

    product_type_summary = create_single_multi_product_summary(data_frame)
    product_matrix_summary = create_product_depth_matrix_summary(data_frame)
    signals = calculate_product_decision_signals(
        product_summary=product_summary,
        product_type_summary=product_type_summary,
        product_matrix_summary=product_matrix_summary,
    )

    avg_product_count = pd.to_numeric(
        data_frame["NumOfProducts"],
        errors="coerce",
    ).mean()

    single_row = product_type_summary[
        product_type_summary["Product_Relationship_Type"].astype(str).eq("Single-Product Customers")
    ] if not product_type_summary.empty else pd.DataFrame()

    multi_row = product_type_summary[
        product_type_summary["Product_Relationship_Type"].astype(str).eq("Multi-Product Customers")
    ] if not product_type_summary.empty else pd.DataFrame()

    single_churn = single_row.iloc[0]["Churn_Rate"] if not single_row.empty else np.nan
    multi_retention = multi_row.iloc[0]["Retention_Rate"] if not multi_row.empty else np.nan

    st.markdown("#### Current Product Utilization Snapshot")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

    with metric_col_1:
        st.metric("Average Product Count", format_display_decimal(avg_product_count))

    with metric_col_2:
        st.metric("Single-Product Churn Rate", format_display_percent(single_churn))

    with metric_col_3:
        st.metric("Multi-Product Retention Rate", format_display_percent(multi_retention))

    with metric_col_4:
        strongest_row = signals["strongest_product_row"]
        st.metric(
            "Strongest Product Count",
            f"{int(strongest_row['NumOfProducts'])} Products" if strongest_row is not None else "N/A",
        )

    st.markdown("#### Product Decision Signals")

    signal_col_1, signal_col_2, signal_col_3 = st.columns(3, gap="medium")

    with signal_col_1:
        st.metric(
            "Best Retention Product Count",
            f"{int(signals['strongest_product_row']['NumOfProducts'])} Products",
            delta=f"{signals['strongest_product_row']['Retention_Rate']:.2f}% retained",
        )

    with signal_col_2:
        st.metric(
            "Highest Churn Product Count",
            f"{int(signals['highest_churn_product_row']['NumOfProducts'])} Products",
            delta=f"{signals['highest_churn_product_row']['Churn_Rate']:.2f}% churn",
            delta_color="inverse",
        )

    with signal_col_3:
        st.metric(
            "Multi vs Single Retention Gap",
            f"{signals['single_multi_gap']:.2f} pp" if not pd.isna(signals["single_multi_gap"]) else "N/A",
        )

    st.markdown("#### Product Count Impact")

    st.plotly_chart(
        create_product_count_combo_chart(product_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 1: Product Count Impact",
        what="This combo chart compares customer volume, retention rate, and churn rate by product count.",
        why="It tests whether more products improve retention or create product-load risk.",
        how="Use bars for volume and lines for retention/churn rates. High churn with meaningful volume needs action.",
        when="Take action when a product-count group has high churn, low retention, or large exposure.",
    )

    st.markdown("#### Single vs Multi-Product Retention Split")

    if product_type_summary.empty:
        create_empty_visual_message("Single vs Multi-Product Retention Split")
    else:
        st.plotly_chart(
            create_single_multi_stacked_chart(product_type_summary),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        render_business_explanation_block(
            title="How to read Visual 2: Single vs Multi-Product Retention Split",
            what="This 100% stacked bar compares retained and churned share for single-product and multi-product customers.",
            why="It validates whether deeper product relationships improve retention.",
            how="Larger green share means stronger retention; larger red share means higher churn risk.",
            when="Take action when single-product churn is materially higher or multi-product customers underperform.",
        )

    st.markdown("#### Product Relationship Strength Bubble Map")

    st.plotly_chart(
        create_product_strength_bubble_chart(product_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 3: Product Relationship Strength Bubble Map",
        what="This bubble map shows product count groups by relationship strength, retention score, churn rate, and customer volume.",
        why="It helps avoid blindly pushing more products by showing whether product count actually improves relationship quality.",
        how="Large bubbles represent more customers; redder bubbles represent higher churn risk.",
        when="Take action when a large product-count bubble has weak relationship/retention scores or high churn.",
    )

    st.markdown("#### Product Depth × Retention Tier Risk Matrix")

    if product_matrix_summary.empty:
        create_empty_visual_message("Product Depth Risk Matrix")
    else:
        st.plotly_chart(
            create_product_depth_matrix_chart(product_matrix_summary),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        render_business_explanation_block(
            title="How to read Visual 4: Product Depth × Retention Tier Risk Matrix",
            what="This heatmap combines product depth segment with retention strength tier.",
            why="It identifies product-depth micro-segments where churn risk and retention weakness overlap.",
            how="Red cells mean high churn risk; use customer counts to judge business impact.",
            when="Prioritize cells with high churn rate and meaningful customer count.",
        )

    st.markdown("#### Product Utilization Summary Table")

    st.dataframe(
        create_product_display_table(product_summary),
        use_container_width=True,
        hide_index=True,
    )

    render_product_recommendations(signals)


# =========================================================
# MODULE 3 — HIGH-VALUE DISENGAGED CUSTOMER DETECTOR
# Visualization mix:
# 1. Donut chart
# 2. Treemap
# 3. Bubble scatter
# 4. Heatmap risk matrix
# =========================================================

def create_high_value_working_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Clean high-value customer working frame.
    """
    required_columns = [
        "CustomerId",
        "High_Value_Customer_Status",
        "High_Value_Disengaged_Status",
        "Engagement_Status",
        "Exited",
        "Balance",
        "EstimatedSalary",
        "Salary_Balance_Mismatch_Status",
        "Retention_Strength_Score",
        "Retention_Strength_Tier",
        "Relationship_Strength_Index",
    ]

    if (
        data_frame.empty
        or any(column not in data_frame.columns for column in required_columns)
    ):
        return pd.DataFrame()

    working_df = data_frame.copy()

    for column in [
        "Exited",
        "Balance",
        "EstimatedSalary",
        "Retention_Strength_Score",
        "Relationship_Strength_Index",
    ]:
        working_df[column] = pd.to_numeric(
            working_df[column],
            errors="coerce",
        )

    working_df["Exited"] = working_df["Exited"].fillna(0)

    high_value_df = working_df[get_high_value_customer_mask(working_df)].copy()

    if high_value_df.empty:
        return pd.DataFrame()

    high_value_df["Detector_Group"] = np.where(
        get_high_value_disengaged_mask(high_value_df),
        "High Value Disengaged",
        "High Value Engaged / Stable",
    )

    return high_value_df.reset_index(drop=True)


def create_high_value_detector_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    High-value engaged vs disengaged summary.
    """
    high_value_df = create_high_value_working_frame(data_frame)

    if high_value_df.empty:
        return pd.DataFrame()

    summary = create_rate_summary(
        data_frame=high_value_df,
        group_columns=["Detector_Group"],
        extra_aggregations={
            "Total_Balance_Exposure": ("Balance", "sum"),
            "Avg_Balance": ("Balance", "mean"),
            "Avg_Estimated_Salary": ("EstimatedSalary", "mean"),
            "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
            "Avg_Relationship_Strength_Index": ("Relationship_Strength_Index", "mean"),
        },
    )

    return summary


def create_high_value_salary_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Salary-balance mismatch summary for high-value customers.
    """
    high_value_df = create_high_value_working_frame(data_frame)

    if high_value_df.empty:
        return pd.DataFrame()

    return create_rate_summary(
        data_frame=high_value_df,
        group_columns=["Salary_Balance_Mismatch_Status"],
        extra_aggregations={
            "Total_Balance_Exposure": ("Balance", "sum"),
            "Avg_Balance": ("Balance", "mean"),
            "Avg_Estimated_Salary": ("EstimatedSalary", "mean"),
        },
    )


def create_high_value_matrix_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    High-value engagement × retention tier summary.
    """
    high_value_df = create_high_value_working_frame(data_frame)

    if high_value_df.empty:
        return pd.DataFrame()

    return create_rate_summary(
        data_frame=high_value_df,
        group_columns=["Engagement_Status", "Retention_Strength_Tier"],
        extra_aggregations={
            "Total_Balance_Exposure": ("Balance", "sum"),
            "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
            "Avg_Relationship_Strength_Index": ("Relationship_Strength_Index", "mean"),
        },
    )


def create_high_value_segment_donut(detector_summary: pd.DataFrame) -> go.Figure:
    """
    Premium 100% stacked bar: high-value customer segment mix.

    Note:
    - Function name is intentionally kept unchanged for safe call-order stability.
    - This replaces the previous donut because two-category donuts created too much
      empty left/right space in the Streamlit wide layout.
    """
    chart_data = detector_summary.copy()
    chart_data["Detector_Group"] = chart_data["Detector_Group"].astype(str)

    total_customers = chart_data["Customer_Count"].sum()
    chart_data["Segment_Share"] = chart_data["Customer_Count"].apply(
        lambda value: safe_divide(value, total_customers, 100)
    )

    color_map = build_dynamic_color_map(
        chart_data["Detector_Group"].tolist(),
        HIGH_VALUE_COLOR_MAP,
    )

    fig = go.Figure()

    for _, row in chart_data.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Segment_Share"]],
                y=["High-Value Mix"],
                orientation="h",
                name=row["Detector_Group"],
                marker_color=color_map.get(row["Detector_Group"], ECB_CHART_COLORS["blue"]),
                text=[
                    f"{row['Detector_Group']}<br>"
                    f"{row['Segment_Share']:.1f}% | {int(row['Customer_Count']):,}"
                ],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    f"<b>{row['Detector_Group']}</b><br>"
                    f"Customers: {int(row['Customer_Count']):,}<br>"
                    f"Segment Share: {row['Segment_Share']:.2f}%<br>"
                    f"Churn Rate: {row['Churn_Rate']:.2f}%<br>"
                    f"Balance Exposure: {row['Total_Balance_Exposure']:,.0f}<extra></extra>"
                ),
            )
        )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="High-Value Customer Segment Mix",
        height=300,
        xaxis_title="Share of High-Value Customers (%)",
        yaxis_title=None,
        legend_title="Detector Group",
    )

    fig.update_layout(
        barmode="stack",
        bargap=0.55,
        margin=dict(l=24, r=24, t=76, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.5,
            title="Detector Group",
        ),
    )

    fig.update_xaxes(
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E6ECF3",
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    return fig


def create_high_value_balance_treemap(detector_summary: pd.DataFrame) -> go.Figure:
    """
    Treemap: balance exposure by high-value detector group.
    """
    chart_data = detector_summary.copy()

    fig = px.treemap(
        chart_data,
        path=["Detector_Group"],
        values="Total_Balance_Exposure",
        color="Churn_Rate",
        color_continuous_scale=[
            ECB_CHART_COLORS["green"],
            ECB_CHART_COLORS["gold"],
            ECB_CHART_COLORS["orange"],
            ECB_CHART_COLORS["red"],
        ],
        hover_data={
            "Customer_Count": ":,",
            "Churn_Rate": ":.2f",
            "Total_Balance_Exposure": ":,.0f",
        },
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,.0f}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Balance Exposure: %{value:,.0f}<br>"
            "Churn Rate: %{color:.2f}%<extra></extra>"
        ),
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Balance Exposure by High-Value Segment",
        height=430,
        show_legend=False,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Churn Rate (%)",
            ticksuffix="%",
        )
    )

    return fig


def create_salary_balance_bubble_chart(salary_summary: pd.DataFrame) -> go.Figure:
    """
    Bubble scatter: salary-balance mismatch risk.
    """
    chart_data = salary_summary.copy()

    color_map = build_dynamic_color_map(
        chart_data["Salary_Balance_Mismatch_Status"].astype(str).tolist(),
        MISMATCH_COLOR_MAP,
    )

    fig = px.scatter(
        chart_data,
        x="Avg_Balance",
        y="Churn_Rate",
        size="Customer_Count",
        color="Salary_Balance_Mismatch_Status",
        text="Customer_Count",
        color_discrete_map=color_map,
        hover_data={
            "Customer_Count": ":,",
            "Total_Balance_Exposure": ":,.0f",
            "Avg_Estimated_Salary": ":,.0f",
            "Churn_Rate": ":.2f",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="middle center",
        marker=dict(
            opacity=0.76,
            line=dict(width=1, color="white"),
        ),
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Salary-Balance Mismatch Risk Bubble Map",
        height=460,
        xaxis_title="Average Balance",
        yaxis_title="Churn Rate (%)",
        legend_title="Salary-Balance Status",
    )

    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(tickformat=",")

    return fig


def create_high_value_matrix_chart(matrix_summary: pd.DataFrame) -> go.Figure:
    """
    Heatmap: high-value engagement × retention tier.
    """
    return create_churn_heatmap(
        matrix_summary=matrix_summary,
        row_column="Engagement_Status",
        column_column="Retention_Strength_Tier",
        row_order=["Active Member", "Inactive Member"],
        column_order=RETENTION_TIER_ORDER,
        title="High-Value Engagement × Retention Tier Risk Matrix",
        xaxis_title="Retention Strength Tier",
        yaxis_title="Engagement Status",
        height=450,
    )


def calculate_high_value_signals(
    detector_summary: pd.DataFrame,
    salary_summary: pd.DataFrame,
    matrix_summary: pd.DataFrame,
) -> dict:
    """
    Dynamic Module 3 business signals.
    """
    signals = {
        "disengaged_row": None,
        "stable_row": None,
        "churn_gap": np.nan,
        "highest_salary_impact": None,
        "highest_rate_segment": None,
        "highest_impact_segment": None,
    }

    if not detector_summary.empty:
        disengaged_rows = detector_summary[
            detector_summary["Detector_Group"].eq("High Value Disengaged")
        ]

        stable_rows = detector_summary[
            detector_summary["Detector_Group"].eq("High Value Engaged / Stable")
        ]

        if not disengaged_rows.empty:
            signals["disengaged_row"] = disengaged_rows.iloc[0]

        if not stable_rows.empty:
            signals["stable_row"] = stable_rows.iloc[0]

        if signals["disengaged_row"] is not None and signals["stable_row"] is not None:
            signals["churn_gap"] = (
                float(signals["disengaged_row"]["Churn_Rate"])
                - float(signals["stable_row"]["Churn_Rate"])
            )

    if not salary_summary.empty:
        signals["highest_salary_impact"] = salary_summary.sort_values(
            ["Churned_Customers", "Total_Balance_Exposure", "Churn_Rate"],
            ascending=[False, False, False],
        ).iloc[0]

    if not matrix_summary.empty:
        signals["highest_rate_segment"] = matrix_summary.sort_values(
            ["Churn_Rate", "Customer_Count"],
            ascending=[False, False],
        ).iloc[0]

        signals["highest_impact_segment"] = matrix_summary.sort_values(
            ["Churned_Customers", "Total_Balance_Exposure", "Customer_Count"],
            ascending=[False, False, False],
        ).iloc[0]

    return signals


def create_high_value_display_table(detector_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Business-readable Module 3 table.
    """
    if detector_summary.empty:
        return pd.DataFrame()

    display_table = detector_summary.rename(
        columns={
            "Detector_Group": "High-Value Segment",
            "Customer_Count": "Customers",
            "Retained_Customers": "Retained",
            "Churned_Customers": "Churned",
            "Retention_Rate": "Retention Rate",
            "Churn_Rate": "Churn Rate",
            "Total_Balance_Exposure": "Total Balance Exposure",
            "Avg_Balance": "Avg Balance",
            "Avg_Estimated_Salary": "Avg Salary",
            "Avg_Retention_Strength_Score": "Avg Retention Score",
            "Avg_Relationship_Strength_Index": "Avg Relationship Index",
        }
    ).copy()

    for column in ["Customers", "Retained", "Churned"]:
        display_table[column] = display_table[column].map(format_display_number)

    for column in ["Retention Rate", "Churn Rate"]:
        display_table[column] = display_table[column].map(format_display_percent)

    for column in ["Total Balance Exposure", "Avg Balance", "Avg Salary"]:
        display_table[column] = display_table[column].map(format_display_number)

    for column in ["Avg Retention Score", "Avg Relationship Index"]:
        display_table[column] = display_table[column].map(format_display_decimal)

    return display_table


def render_high_value_recommendations(signals: dict) -> None:
    """
    Dynamic premium retention recommendation.
    """
    st.markdown("#### Business Interpretation & Premium Retention Actions")

    disengaged_row = signals["disengaged_row"]
    stable_row = signals["stable_row"]

    if disengaged_row is not None:
        st.error(
            f"High-value disengaged customers represent **{int(disengaged_row['Customer_Count']):,} customers** "
            f"with **{format_display_number(disengaged_row['Total_Balance_Exposure'])} balance exposure** "
            f"and **{disengaged_row['Churn_Rate']:.2f}% churn**."
        )

    if stable_row is not None:
        st.success(
            f"High-value engaged/stable customers show **{stable_row['Retention_Rate']:.2f}% retention**, "
            "which should be used as the premium relationship benchmark."
        )

    if not pd.isna(signals["churn_gap"]):
        st.warning(
            f"High-value disengagement churn gap is **{signals['churn_gap']:.2f} percentage points**. "
            "Premium customers should not be treated as safe only because they have high balances."
        )

    if signals["highest_rate_segment"] is not None and signals["highest_impact_segment"] is not None:
        rate_segment = signals["highest_rate_segment"]
        impact_segment = signals["highest_impact_segment"]

        col_1, col_2 = st.columns(2, gap="medium")

        with col_1:
            st.markdown("**Highest Rate Premium Risk**")
            st.write(
                f"**{rate_segment['Engagement_Status']} + {rate_segment['Retention_Strength_Tier']}** "
                f"has **{rate_segment['Churn_Rate']:.2f}% churn** across "
                f"**{int(rate_segment['Customer_Count']):,} high-value customers**."
            )

        with col_2:
            st.markdown("**Highest Premium Business Impact**")
            st.write(
                f"**{impact_segment['Engagement_Status']} + {impact_segment['Retention_Strength_Tier']}** "
                f"has **{int(impact_segment['Churned_Customers']):,} churned customers** and "
                f"**{format_display_number(impact_segment['Total_Balance_Exposure'])} balance exposure**."
            )

        st.error(
            f"First premium-retention priority: **{impact_segment['Engagement_Status']} + "
            f"{impact_segment['Retention_Strength_Tier']}**. Use relationship-manager outreach, "
            "personalized financial check-ins, product-fit review, and early-warning monitoring."
        )

    if signals["highest_salary_impact"] is not None:
        salary_row = signals["highest_salary_impact"]

        st.markdown("**Salary-Balance Risk Interpretation**")
        st.write(
            f"Highest salary-balance risk appears in **{salary_row['Salary_Balance_Mismatch_Status']}**, "
            f"with **{int(salary_row['Churned_Customers']):,} churned customers**, "
            f"**{salary_row['Churn_Rate']:.2f}% churn**, and "
            f"**{format_display_number(salary_row['Total_Balance_Exposure'])} balance exposure**."
        )


def render_high_value_disengaged_module(data_frame: pd.DataFrame) -> None:
    """
    Render Module 3: High-Value Disengaged Customer Detector.
    """
    st.subheader("High-Value Disengaged Customer Detector")

    st.caption(
        "Guideline focus: Identify disengaged yet high-value customers, detect premium silent churn risk, "
        "and support targeted retention actions for financially valuable segments."
    )

    st.write(
        "This module identifies premium customers who may look financially strong but still carry churn risk because "
        "of weak engagement, salary-balance mismatch, or low retention strength."
    )

    high_value_df = create_high_value_working_frame(data_frame)
    detector_summary = create_high_value_detector_summary(data_frame)

    if high_value_df.empty or detector_summary.empty:
        create_empty_visual_message("High-Value Disengaged Customer Detector")
        return

    salary_summary = create_high_value_salary_summary(data_frame)
    matrix_summary = create_high_value_matrix_summary(data_frame)
    signals = calculate_high_value_signals(
        detector_summary=detector_summary,
        salary_summary=salary_summary,
        matrix_summary=matrix_summary,
    )

    high_value_customers = high_value_df["CustomerId"].nunique()
    disengaged_customers = high_value_df[
        high_value_df["Detector_Group"].eq("High Value Disengaged")
    ]["CustomerId"].nunique()
    disengagement_rate = safe_divide(disengaged_customers, high_value_customers, 100)

    disengaged_balance = high_value_df[
        high_value_df["Detector_Group"].eq("High Value Disengaged")
    ]["Balance"].sum()

    avg_disengaged_salary = high_value_df[
        high_value_df["Detector_Group"].eq("High Value Disengaged")
    ]["EstimatedSalary"].mean()

    st.markdown("#### Current High-Value Customer Risk Snapshot")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

    with metric_col_1:
        st.metric("High-Value Customers", f"{high_value_customers:,}")

    with metric_col_2:
        st.metric("High-Value Disengaged", f"{disengaged_customers:,}")

    with metric_col_3:
        st.metric("Disengagement Rate", format_display_percent(disengagement_rate))

    with metric_col_4:
        st.metric("Balance at Risk", format_display_number(disengaged_balance))

    st.markdown("#### Premium Exposure Signals")

    exposure_col_1, exposure_col_2, exposure_col_3 = st.columns(3, gap="medium")

    with exposure_col_1:
        st.metric(
            "Avg Disengaged Balance",
            format_display_number(
                high_value_df[
                    high_value_df["Detector_Group"].eq("High Value Disengaged")
                ]["Balance"].mean()
            ),
        )

    with exposure_col_2:
        st.metric("Avg Disengaged Salary", format_display_number(avg_disengaged_salary))

    with exposure_col_3:
        if signals["highest_salary_impact"] is not None:
            st.metric(
                "Highest Salary-Risk Group",
                str(signals["highest_salary_impact"]["Salary_Balance_Mismatch_Status"]),
                delta=f"{signals['highest_salary_impact']['Churn_Rate']:.2f}% churn",
                delta_color="inverse",
            )
        else:
            st.metric("Highest Salary-Risk Group", "N/A")

    st.markdown("#### High-Value Customer Segment Mix")

    st.plotly_chart(
        create_high_value_segment_donut(detector_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 1: High-Value Customer Segment Mix",
        what="This 100% stacked bar shows the share of high-value customers who are disengaged versus engaged/stable.",
        why="It reveals whether premium disengagement is a small edge case or a major customer-base issue.",
        how="Read the segment width as customer share. A larger disengaged share means more premium customers need relationship reactivation.",
        when="Take action when high-value disengaged share is meaningful after filters are applied.",
    )

    st.markdown("#### Balance Exposure by High-Value Segment")

    st.plotly_chart(
        create_high_value_balance_treemap(detector_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 2: Balance Exposure by High-Value Segment",
        what="This treemap shows how total balance exposure is distributed across high-value segments.",
        why="Premium retention strategy should consider financial exposure, not only customer count.",
        how="Larger tiles indicate larger balance exposure; stronger risk color indicates higher churn.",
        when="Take action when a large exposure segment also has elevated churn risk.",
    )

    st.markdown("#### Salary-Balance Mismatch Risk Bubble Map")

    if salary_summary.empty:
        create_empty_visual_message("Salary-Balance Mismatch Bubble Map")
    else:
        st.plotly_chart(
            create_salary_balance_bubble_chart(salary_summary),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        render_business_explanation_block(
            title="How to read Visual 3: Salary-Balance Mismatch Risk Bubble Map",
            what="This bubble map compares mismatch groups by average balance, churn rate, and customer volume.",
            why="It detects high-value customers whose financial profile may indicate unstable relationship depth.",
            how="Large bubbles mean more customers; higher bubbles mean higher churn; right-side bubbles mean higher average balance.",
            when="Take action when a mismatch group has high churn and meaningful customer volume or exposure.",
        )

    st.markdown("#### High-Value Engagement × Retention Tier Risk Matrix")

    if matrix_summary.empty:
        create_empty_visual_message("High-Value Risk Matrix")
    else:
        st.plotly_chart(
            create_high_value_matrix_chart(matrix_summary),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        render_business_explanation_block(
            title="How to read Visual 4: High-Value Engagement × Retention Tier Risk Matrix",
            what="This heatmap combines engagement status with retention tier for high-value customers.",
            why="It identifies premium micro-segments where engagement weakness and retention risk overlap.",
            how="Red cells show high churn; customer counts show business impact.",
            when="Prioritize cells with high churn, high count, and high-value exposure.",
        )

    st.markdown("#### High-Value Detector Summary Table")

    st.dataframe(
        create_high_value_display_table(detector_summary),
        use_container_width=True,
        hide_index=True,
    )

    render_high_value_recommendations(signals)


# =========================================================
# MODULE 4 — RETENTION STRENGTH SCORING PANELS
# Visualization mix:
# 1. Distribution histogram
# 2. Box plot
# 3. Scatter plot
# 4. Tier bar ranking
# =========================================================

def create_retention_strength_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Retention-tier-level scoring summary.
    """
    extra_aggregations = {
        "Avg_Retention_Strength_Score": ("Retention_Strength_Score", "mean"),
        "Avg_Relationship_Strength_Index": ("Relationship_Strength_Index", "mean"),
        "Avg_Product_Count": ("NumOfProducts", "mean"),
        "Avg_Balance": ("Balance", "mean"),
    }

    summary = create_rate_summary(
        data_frame=data_frame,
        group_columns=["Retention_Strength_Tier"],
        extra_aggregations=extra_aggregations,
    )

    if summary.empty:
        return summary

    category_order = get_existing_category_order(
        summary,
        "Retention_Strength_Tier",
        RETENTION_TIER_ORDER,
    )

    summary["Retention_Strength_Tier"] = pd.Categorical(
        summary["Retention_Strength_Tier"],
        categories=category_order,
        ordered=True,
    )

    return summary.sort_values("Retention_Strength_Tier")


def create_retention_score_histogram(data_frame: pd.DataFrame) -> go.Figure:
    """
    Histogram: retention strength score distribution.
    """
    working_df = data_frame.copy()
    working_df["Retention_Strength_Score"] = pd.to_numeric(
        working_df["Retention_Strength_Score"],
        errors="coerce",
    )

    color_map = build_dynamic_color_map(
        get_existing_category_order(
            working_df,
            "Retention_Strength_Tier",
            RETENTION_TIER_ORDER,
        ),
        RETENTION_TIER_COLOR_MAP,
    )

    fig = px.histogram(
        working_df.dropna(subset=["Retention_Strength_Score"]),
        x="Retention_Strength_Score",
        color="Retention_Strength_Tier",
        nbins=24,
        marginal="rug",
        color_discrete_map=color_map,
        category_orders={"Retention_Strength_Tier": RETENTION_TIER_ORDER},
        hover_data=["Retention_Strength_Tier", "Engagement_Status", "NumOfProducts"],
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Retention Strength Score Distribution",
        height=450,
        xaxis_title="Retention Strength Score",
        yaxis_title="Customer Count",
        legend_title="Retention Tier",
    )

    return fig


def create_retention_tier_box_plot(data_frame: pd.DataFrame) -> go.Figure:
    """
    Box plot: relationship strength distribution by retention tier.
    """
    working_df = data_frame.copy()
    working_df["Relationship_Strength_Index"] = pd.to_numeric(
        working_df["Relationship_Strength_Index"],
        errors="coerce",
    )

    tier_order = get_existing_category_order(
        working_df,
        "Retention_Strength_Tier",
        RETENTION_TIER_ORDER,
    )

    color_map = build_dynamic_color_map(
        tier_order,
        RETENTION_TIER_COLOR_MAP,
    )

    fig = px.box(
        working_df.dropna(subset=["Relationship_Strength_Index"]),
        x="Retention_Strength_Tier",
        y="Relationship_Strength_Index",
        color="Retention_Strength_Tier",
        points="outliers",
        category_orders={"Retention_Strength_Tier": tier_order},
        color_discrete_map=color_map,
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Relationship Strength Distribution by Retention Tier",
        height=450,
        xaxis_title="Retention Strength Tier",
        yaxis_title="Relationship Strength Index",
        legend_title="Retention Tier",
        show_legend=False,
    )

    fig.update_xaxes(tickangle=-22)

    return fig


def create_retention_relationship_scatter(data_frame: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: relationship strength vs retention score.
    """
    working_df = data_frame.copy()

    working_df["Relationship_Strength_Index"] = pd.to_numeric(
        working_df["Relationship_Strength_Index"],
        errors="coerce",
    )

    working_df["Retention_Strength_Score"] = pd.to_numeric(
        working_df["Retention_Strength_Score"],
        errors="coerce",
    )

    working_df["Exited"] = pd.to_numeric(
        working_df["Exited"],
        errors="coerce",
    ).fillna(0)

    working_df["Customer_Outcome"] = np.where(
        working_df["Exited"].eq(1),
        "Churned Customers",
        "Retained Customers",
    )

    fig = px.scatter(
        working_df.dropna(
            subset=["Relationship_Strength_Index", "Retention_Strength_Score"]
        ),
        x="Relationship_Strength_Index",
        y="Retention_Strength_Score",
        color="Customer_Outcome",
        size="NumOfProducts",
        opacity=0.62,
        color_discrete_map=CUSTOMER_OUTCOME_COLOR_MAP,
        hover_data={
            "CustomerId": True,
            "Engagement_Status": True,
            "NumOfProducts": True,
            "Retention_Strength_Tier": True,
            "Balance": ":,.0f",
        },
    )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="white"))
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Relationship Strength vs Retention Strength Map",
        height=500,
        xaxis_title="Relationship Strength Index",
        yaxis_title="Retention Strength Score",
        legend_title="Customer Outcome",
    )

    return fig


def create_retention_tier_ranking_chart(retention_summary: pd.DataFrame) -> go.Figure:
    """
    Horizontal ranking bar: churn rate by retention tier.
    """
    chart_data = retention_summary.copy()

    chart_data = chart_data.sort_values(
        ["Churn_Rate", "Customer_Count"],
        ascending=[True, True],
    )

    color_map = build_dynamic_color_map(
        chart_data["Retention_Strength_Tier"].astype(str).tolist(),
        RETENTION_TIER_COLOR_MAP,
    )

    fig = px.bar(
        chart_data,
        x="Churn_Rate",
        y=chart_data["Retention_Strength_Tier"].astype(str),
        orientation="h",
        color=chart_data["Retention_Strength_Tier"].astype(str),
        text="Churn_Rate",
        color_discrete_map=color_map,
        hover_data={
            "Customer_Count": ":,",
            "Churned_Customers": ":,",
            "Retention_Rate": ":.2f",
            "Avg_Retention_Strength_Score": ":.2f",
            "Avg_Relationship_Strength_Index": ":.2f",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Churn Rate: %{x:.2f}%<br>"
            "Customers: %{customdata[0]:,}<br>"
            "Churned: %{customdata[1]:,}<extra></extra>"
        ),
    )

    fig = apply_premium_plotly_layout(
        fig=fig,
        title="Retention Tier Churn Risk Ranking",
        height=430,
        xaxis_title="Churn Rate (%)",
        yaxis_title="Retention Strength Tier",
        show_legend=False,
    )

    fig.update_xaxes(ticksuffix="%")

    return fig


def calculate_retention_signals(
    retention_summary: pd.DataFrame,
    data_frame: pd.DataFrame,
) -> dict:
    """
    Dynamic Module 4 business signals.
    """
    signals = {
        "avg_retention_score": np.nan,
        "avg_relationship_index": np.nan,
        "strongest_tier": None,
        "highest_risk_tier": None,
        "largest_tier": None,
    }

    if data_frame.empty or retention_summary.empty:
        return signals

    signals["avg_retention_score"] = pd.to_numeric(
        data_frame["Retention_Strength_Score"],
        errors="coerce",
    ).mean()

    signals["avg_relationship_index"] = pd.to_numeric(
        data_frame["Relationship_Strength_Index"],
        errors="coerce",
    ).mean()

    signals["strongest_tier"] = retention_summary.sort_values(
        ["Retention_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    signals["highest_risk_tier"] = retention_summary.sort_values(
        ["Churn_Rate", "Customer_Count"],
        ascending=[False, False],
    ).iloc[0]

    signals["largest_tier"] = retention_summary.sort_values(
        ["Customer_Count", "Churn_Rate"],
        ascending=[False, False],
    ).iloc[0]

    return signals


def create_retention_display_table(retention_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Business-readable Module 4 table.
    """
    if retention_summary.empty:
        return pd.DataFrame()

    display_table = retention_summary.rename(
        columns={
            "Retention_Strength_Tier": "Retention Tier",
            "Customer_Count": "Customers",
            "Retained_Customers": "Retained",
            "Churned_Customers": "Churned",
            "Retention_Rate": "Retention Rate",
            "Churn_Rate": "Churn Rate",
            "Avg_Retention_Strength_Score": "Avg Retention Score",
            "Avg_Relationship_Strength_Index": "Avg Relationship Index",
            "Avg_Product_Count": "Avg Product Count",
            "Avg_Balance": "Avg Balance",
        }
    ).copy()

    for column in ["Customers", "Retained", "Churned"]:
        display_table[column] = display_table[column].map(format_display_number)

    for column in ["Retention Rate", "Churn Rate"]:
        display_table[column] = display_table[column].map(format_display_percent)

    for column in ["Avg Retention Score", "Avg Relationship Index", "Avg Product Count"]:
        display_table[column] = display_table[column].map(format_display_decimal)

    display_table["Avg Balance"] = display_table["Avg Balance"].map(format_display_number)

    return display_table


def render_retention_recommendations(signals: dict) -> None:
    """
    Dynamic retention scoring recommendation.
    """
    st.markdown("#### Business Interpretation & Retention Scoring Actions")

    strongest_tier = signals["strongest_tier"]
    highest_risk_tier = signals["highest_risk_tier"]
    largest_tier = signals["largest_tier"]

    if strongest_tier is None or highest_risk_tier is None:
        st.info("Retention scoring recommendation is unavailable for the current filtered view.")
        return

    st.success(
        f"**{strongest_tier['Retention_Strength_Tier']}** is the strongest tier with "
        f"**{strongest_tier['Retention_Rate']:.2f}% retention**."
    )

    st.error(
        f"**{highest_risk_tier['Retention_Strength_Tier']}** is the highest risk tier with "
        f"**{highest_risk_tier['Churn_Rate']:.2f}% churn** and "
        f"**{int(highest_risk_tier['Churned_Customers']):,} churned customers**."
    )

    if largest_tier is not None:
        st.info(
            f"The largest tier is **{largest_tier['Retention_Strength_Tier']}** with "
            f"**{int(largest_tier['Customer_Count']):,} customers**. "
            "This segment should be monitored because large volume can create meaningful business impact."
        )

    st.markdown("**Recommended retention scoring strategy**")
    st.write(
        "Use the retention strength score as a prioritization layer. Start with the highest churn tier, "
        "then review large-volume moderate/weak tiers for scalable retention campaigns. "
        "Use relationship-strength improvement, product-fit review, engagement nudges, and customer-service outreach "
        "to move customers from critical or weak tiers into stronger retention tiers."
    )


def render_retention_strength_module(data_frame: pd.DataFrame) -> None:
    """
    Render Module 4: Retention Strength Scoring Panels.
    """
    st.subheader("Retention Strength Scoring Panels")

    st.caption(
        "Guideline focus: Assess sticky customer profiles, measure churn stability across retention tiers, "
        "and identify relationship-strength thresholds linked to retention."
    )

    st.write(
        "This module converts engagement, product depth, and relationship signals into retention-strength panels "
        "that help prioritize customers for retention actions."
    )

    retention_summary = create_retention_strength_summary(data_frame)

    if retention_summary.empty:
        create_empty_visual_message("Retention Strength Scoring Panels")
        return

    signals = calculate_retention_signals(
        retention_summary=retention_summary,
        data_frame=data_frame,
    )

    st.markdown("#### Current Retention Strength Snapshot")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4, gap="medium")

    with metric_col_1:
        st.metric("Avg Retention Strength Score", format_display_decimal(signals["avg_retention_score"]))

    with metric_col_2:
        st.metric("Avg Relationship Strength", format_display_decimal(signals["avg_relationship_index"]))

    with metric_col_3:
        st.metric(
            "Strongest Retention Tier",
            str(signals["strongest_tier"]["Retention_Strength_Tier"]),
            delta=f"{signals['strongest_tier']['Retention_Rate']:.2f}% retained",
        )

    with metric_col_4:
        st.metric(
            "Highest Risk Tier",
            str(signals["highest_risk_tier"]["Retention_Strength_Tier"]),
            delta=f"{signals['highest_risk_tier']['Churn_Rate']:.2f}% churn",
            delta_color="inverse",
        )

    st.markdown("#### Retention Strength Score Distribution")

    st.plotly_chart(
        create_retention_score_histogram(data_frame),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 1: Retention Strength Score Distribution",
        what="This histogram shows how customers are distributed across retention strength scores.",
        why="It reveals whether the customer base is concentrated in weak, moderate, or strong retention zones.",
        how="Look for heavy concentration in low-score ranges or strong-score ranges after filters are applied.",
        when="Take action when many customers cluster in low or weak score areas.",
    )

    st.markdown("#### Relationship Strength Distribution by Retention Tier")

    st.plotly_chart(
        create_retention_tier_box_plot(data_frame),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 2: Relationship Strength Distribution by Retention Tier",
        what="This box plot compares relationship strength distribution across retention tiers.",
        why="It shows whether stronger retention tiers have more stable relationship strength patterns.",
        how="Higher boxes indicate stronger relationship quality; wide boxes or many outliers indicate inconsistency.",
        when="Take action when weak or critical tiers show low relationship strength distribution.",
    )

    st.markdown("#### Relationship Strength vs Retention Strength Map")

    st.plotly_chart(
        create_retention_relationship_scatter(data_frame),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 3: Relationship Strength vs Retention Strength Map",
        what="This scatter plot maps customers by relationship strength and retention strength score.",
        why="It helps locate churned and retained customers inside the relationship-retention space.",
        how="Churned customers in the low-score area show immediate risk; retained customers in the high-score area show sticky profiles.",
        when="Take action when many churned customers appear at low relationship and retention strength.",
    )

    st.markdown("#### Retention Tier Churn Risk Ranking")

    st.plotly_chart(
        create_retention_tier_ranking_chart(retention_summary),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    render_business_explanation_block(
        title="How to read Visual 4: Retention Tier Churn Risk Ranking",
        what="This horizontal ranking chart orders retention tiers by churn risk.",
        why="It converts scoring output into campaign-priority order.",
        how="Longer bars mean higher churn rate. Use customer count and churned customers in hover to judge business impact.",
        when="Take action on the highest churn tier first, then monitor large-volume weak or moderate tiers.",
    )

    st.markdown("#### Retention Strength Tier Summary Table")

    st.dataframe(
        create_retention_display_table(retention_summary),
        use_container_width=True,
        hide_index=True,
    )

    render_retention_recommendations(signals)
# =========================================================
# REBUILD STEP 1I — GUIDELINE-COMPLIANT DASHBOARD MODULE TABS
# Core modules required by project guidelines:
# 1. Engagement vs Churn Overview
# 2. Product Utilization Impact Analysis
# 3. High-Value Disengaged Customer Detector
# 4. Retention Strength Scoring Panels
# =========================================================

def render_dashboard_module_tabs(data_frame: pd.DataFrame) -> None:
    """
    Render the four required dashboard modules using Streamlit-native tabs.

    Important:
    - No custom HTML
    - No CSS changes
    - No mobile/tablet changes
    - All modules receive filtered_df, so every module stays connected to sidebar filters
    """
    st.divider()

    st.subheader("Dashboard Analysis Modules")

    st.caption(
        "Explore the filtered customer base through four required retention analytics modules: "
        "engagement behavior, product utilization, high-value disengagement risk, "
        "and retention strength scoring."
    )

    module_1_tab, module_2_tab, module_3_tab, module_4_tab = st.tabs(
        [
            "Engagement vs Churn",
            "Product Utilization",
            "High-Value Disengaged",
            "Retention Strength",
        ]
    )

    with module_1_tab:
        render_engagement_vs_churn_module(data_frame)

    with module_2_tab:
        render_product_utilization_module(data_frame)

    with module_3_tab:
        render_high_value_disengaged_module(data_frame)

    with module_4_tab:
        render_retention_strength_module(data_frame)


render_dashboard_module_tabs(filtered_df)


# -------------------------
# FOOTER SECTION
# -------------------------
st.markdown("---")

st.markdown("### 📌 Project Information & Credits")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
**👨‍💻 Developed by:** Mohit Gupta
  
**🎯 Role:** Data Analyst Intern
        """
    )

with c2:
    st.markdown(
        """
**📊 Project:** Customer Engagement & Product Utilization Analytics for Retention Strategy

**🏢 Organization:** Unified Mentor Pvt. Ltd.
        """
    )

with c3:
    st.markdown(
        """
**👨‍🏫 Mentor:** Saiprasad Kagne
  
**📅 Year:** 2026
        """
    )

st.markdown(
    """
<div style="
    text-align: center;
    margin-top: 10px;
    color: #6b563d;
    font-size: 14px;
    font-weight: 600;
">
    💡 Built using Python, Pandas, Plotly & Streamlit
</div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)