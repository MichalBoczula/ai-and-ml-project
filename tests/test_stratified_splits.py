from math import ceil
from pathlib import Path
from unittest import TestCase

import pandas as pd
from sklearn.metrics import accuracy_score

from main import predict_with_svc, split_dataset


class StratifiedSplitTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "pulsar_stars.csv"
        cls.df = pd.read_csv(dataset_path)
        cls.class_proportions = cls.df["target_class"].value_counts(normalize=True)

    def check_split(self, split_label: str, test_size: float) -> None:
        X_train, X_test, y_train, y_test = split_dataset(self.df, test_size=test_size)
        expected_test_rows = ceil(len(self.df) * test_size)

        self.assertEqual(len(X_test), expected_test_rows)
        self.assertEqual(len(X_train), len(self.df) - expected_test_rows)
        self.assertTrue(X_train.index.intersection(X_test.index).empty)

        for labels in (y_train, y_test):
            counts = labels.value_counts()
            for target_class, proportion in self.class_proportions.items():
                self.assertAlmostEqual(
                    counts.get(target_class, 0), len(labels) * proportion, delta=1
                )

        y_pred = predict_with_svc(X_train, y_train, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, y_test.value_counts(normalize=True).max())

        test_counts = y_test.value_counts()
        print(
            f"{split_label} | train={len(y_train)} | test={len(y_test)} | "
            f"test class 0={test_counts[0]} | test class 1={test_counts[1]} | "
            f"accuracy={accuracy:.2%}"
        )

    def test_90_10(self) -> None:
        self.check_split("90/10", test_size=0.10)

    def test_85_15(self) -> None:
        self.check_split("85/15", test_size=0.15)

    def test_80_20(self) -> None:
        self.check_split("80/20", test_size=0.20)

    def test_75_25(self) -> None:
        self.check_split("75/25", test_size=0.25)
