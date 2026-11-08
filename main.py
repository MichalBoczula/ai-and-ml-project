from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "data" / "pulsar_stars.csv"
    df = pd.read_csv(dataset_path)

    print(f"DataFrame shape: {df.shape}")
    print(df.head().to_string())

    df.columns = df.columns.str.lstrip()
    df = df.rename(
        columns={
            "Mean of the integrated profile": "profile_mean",
            "Standard deviation of the integrated profile": "profile_std",
            "Excess kurtosis of the integrated profile": "profile_excess_kurtosis",
            "Skewness of the integrated profile": "profile_skewness",
            "Mean of the DM-SNR curve": "dm_snr_mean",
            "Standard deviation of the DM-SNR curve": "dm_snr_std",
            "Excess kurtosis of the DM-SNR curve": "dm_snr_excess_kurtosis",
            "Skewness of the DM-SNR curve": "dm_snr_skewness",
        },
        errors="raise",
    )

    print("\nColumn names after cleanup:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
