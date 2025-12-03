import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, List

def calculate_significant_stats(df: pd.DataFrame, measure_col: str, control_group: str = "T") -> pd.DataFrame:
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
                pass

    return pd.DataFrame(stats_results)

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    try:
        z_scores = df.groupby("Condition")[column].transform(lambda x: stats.zscore(x, nan_policy='omit'))
        df_out = df.copy()
        df_out['Z_Score'] = z_scores
        outliers = df_out[np.abs(df_out['Z_Score']) > threshold].copy()
        outliers = outliers.sort_values(by='Z_Score', key=abs, ascending=False)

        return outliers
    except Exception as e:
        print(f"Error calculating outliers: {e}")
        return pd.DataFrame()
