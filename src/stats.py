import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, List

def calculate_significant_stats(df: pd.DataFrame, measure_col: str, control_group: str = "T") -> pd.DataFrame:
    """
    Calculates Mann-Whitney U test p-values comparing each condition to the control group.

    Args:
        df: DataFrame containing the data.
        measure_col: Name of the column with the measurement values.
        control_group: Name of the control group condition.

    Returns:
        DataFrame with statistical results.
    """
    stats_results = []

    if control_group not in df["Condition"].unique():
        return pd.DataFrame()

    control_data = df[df["Condition"] == control_group][measure_col].dropna()

    for condition in df["Condition"].unique():
        if condition == control_group:
            continue

        cond_data = df[df["Condition"] == condition][measure_col].dropna()

        if len(cond_data) > 1 and len(control_data) > 1:
            try:
                stat, p_value = stats.mannwhitneyu(control_data, cond_data, alternative='two-sided')

                if p_value < 0.001: stars = "***"
                elif p_value < 0.01: stars = "**"
                elif p_value < 0.05: stars = "*"
                else: stars = "ns"

                stats_results.append({
                    "Comparaison": f"{control_group} vs {condition}",
                    "Médiane Témoin": round(control_data.median(), 3),
                    "Médiane Cond.": round(cond_data.median(), 3),
                    "P-value": p_value,
                    "P-value (str)": f"{p_value:.4f}",
                    "Significativité": stars
                })
            except ValueError:
                # Handle cases where all numbers are identical
                pass

    return pd.DataFrame(stats_results)

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Adds a 'Z-Score' column and flags outliers based on the threshold.
    Calculates Z-score group by 'Condition' to respect biological variability between groups.

    Args:
        df: Input DataFrame.
        column: Column to check for outliers (e.g., 'Rapport').
        threshold: Z-score threshold (default 3.0).

    Returns:
        DataFrame containing only the outliers.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    # Calculate Z-score per condition group
    # We use transform to keep the index aligned
    try:
        z_scores = df.groupby("Condition")[column].transform(lambda x: stats.zscore(x, nan_policy='omit'))
        df_out = df.copy()
        df_out['Z_Score'] = z_scores

        # Filter
        outliers = df_out[np.abs(df_out['Z_Score']) > threshold].copy()

        # Sort by Z-score magnitude
        outliers = outliers.sort_values(by='Z_Score', key=abs, ascending=False)

        return outliers
    except Exception as e:
        print(f"Error calculating outliers: {e}")
        return pd.DataFrame()
