from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)

    features = df.drop(columns="target_class")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for column, ax in zip(features.columns, axes.flat, strict=True):
        sns.boxplot(data=features, x=column, ax=ax)
    fig.suptitle("Pulsar feature boxplots")
    fig.savefig(output_dir / "boxplots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained")
    for column, ax in zip(df.columns, axes.flat, strict=True):
        is_target = column == "target_class"
        sns.histplot(data=df, x=column, bins=30, discrete=is_target, ax=ax)
        if is_target:
            ax.set_xticks([0, 1])
    fig.suptitle("Pulsar dataset histograms")
    fig.savefig(output_dir / "histograms.png", dpi=150)
    plt.close(fig)


def split_dataset(df: pd.DataFrame, *, test_size: float):
    X = df.drop(columns="target_class")
    y = df["target_class"]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def predict_with_svc(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svc = SVC()
    svc.fit(X_train_scaled, y_train)
    return svc.predict(X_test_scaled)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "data" / "pulsar_stars.csv"
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

    print("\nTarget class counts:")
    target_counts = df["target_class"].value_counts().reindex([0, 1], fill_value=0)
    print(target_counts.to_string())

    print("\nMissing values by column:")
    missing_values = df.isna().sum()
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum()}")

    print("\nDescriptive statistics:")
    print(df.describe().to_string())

    output_dir = project_dir / "plots"
    save_plots(df, output_dir)
    print(f"\nPlots saved to {output_dir}")

    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2)
    y_pred = predict_with_svc(X_train, y_train, X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTraining rows: {len(y_train)}, test rows: {len(y_test)}")
    print(f"SVC test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
