import unittest
import pandas as pd
from src.stats import calculate_significant_stats, detect_outliers_zscore

class TestStats(unittest.TestCase):
    def test_calculate_significant_stats(self):
        # Increased sample size to get statistical significance
        data = {
            "Condition": ["T"]*5 + ["A"]*5 + ["B"]*5,
            "Value": [10, 10.1, 9.9, 10.05, 9.95,  # T ~ 10
                      15, 15.1, 14.9, 15.05, 14.95, # A ~ 15 (Distinct)
                      10, 10.1, 9.9, 10.05, 9.95]   # B ~ 10 (Same)
        }
        df = pd.DataFrame(data)

        # Test comparison between T and A -> Should be significant
        results = calculate_significant_stats(df, "Value", control_group="T")

        self.assertFalse(results.empty)
        row_a = results[results["Comparaison"] == "T vs A"].iloc[0]
        self.assertIn("*", row_a["Significativité"])

        # Test comparison between T and B -> Should be ns
        row_b = results[results["Comparaison"] == "T vs B"].iloc[0]
        self.assertEqual(row_b["Significativité"], "ns")

    def test_missing_control(self):
        data = {
            "Condition": ["A", "A", "B", "B"],
            "Value": [1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        results = calculate_significant_stats(df, "Value", control_group="T")
        self.assertTrue(results.empty)

    def test_detect_outliers(self):
        # Create data with one obvious outlier in Group A
        data = {
            "Condition": ["A"]*6,
            "Value": [10, 10, 10, 10, 10, 100] # 100 is outlier
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers_zscore(df, "Value", threshold=2.0)
        self.assertFalse(outliers.empty)
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers.iloc[0]["Value"], 100)

if __name__ == '__main__':
    unittest.main()
