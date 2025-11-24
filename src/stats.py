import pandas as pd
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
