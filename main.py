from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
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


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def predict_with_svc(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    svc = SVC()
    svc.fit(X_train_scaled, y_train)
    return svc.predict(X_test_scaled)


def compare_svc_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, np.ndarray], GridSearchCV]:
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    linear_svc = SVC(kernel="linear")
    models = {
        "RBF C=1": SVC(),
        "RBF C=100": SVC(C=100),
        "Linear C=1": linear_svc,
        "Poly C=1": SVC(kernel="poly"),
        "Poly C=100": SVC(kernel="poly", C=100),
        "Sigmoid C=1": SVC(kernel="sigmoid"),
        "Sigmoid C=100": SVC(kernel="sigmoid", C=100),
    }

    scores = []
    matrices = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_accuracy = (
            linear_svc.score(X_test_scaled, y_test)
            if model is linear_svc
            else accuracy_score(y_test, y_pred)
        )
        scores.append({
            "Model": name,
            "Kernel": model.kernel,
            "C": model.C,
            "Train accuracy": model.score(X_train_scaled, y_train),
            "Test accuracy": test_accuracy,
        })
        matrices[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC()),
        param_grid=[
            {"svc__kernel": ["linear"], "svc__C": [1, 10, 100]},
            {
                "svc__kernel": ["rbf", "poly", "sigmoid"],
                "svc__C": [1, 10, 100],
                "svc__gamma": ["scale", 0.01],
            },
        ],
        cv=cv,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    scores.append({
        "Model": "Grid search winner",
        "Kernel": grid.best_params_["svc__kernel"],
        "C": grid.best_params_["svc__C"],
        "Train accuracy": grid.score(X_train, y_train),
        "Test accuracy": accuracy_score(y_test, y_pred),
    })
    matrices["Grid search winner"] = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return pd.DataFrame(scores), matrices, grid


def save_confusion_heatmaps(matrices: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for (name, matrix), ax in zip(matrices.items(), axes.flat, strict=True):
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(
            rates, annot=matrix, fmt="d", cmap="Blues", vmin=0, vmax=1,
            square=True, cbar=False, xticklabels=[0, 1], yticklabels=[0, 1], ax=ax,
        )
        ax.set(title=name, xlabel="Predicted class", ylabel="Actual class")
    fig.suptitle("Test confusion matrices\nColor shows the share within each actual class")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_excel_report(
    path: Path, tables: dict[str, pd.DataFrame], plot_dir: Path
) -> None:
    path.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        percent = writer.book.add_format({"num_format": "0.00%"})
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(table), len(table.columns) - 1)
            for index, column in enumerate(table.columns):
                width = max(len(column) + 3, 16)
                if column in {"Model", "Column", "Metric"}:
                    width = 29
                cell_format = percent if "accuracy" in column.lower() else None
                sheet.set_column(index, index, width, cell_format)

        charts = writer.book.add_worksheet("Charts")
        charts.hide_gridlines(2)
        charts.set_zoom(85)
        for cell, filename in (
            ("A2", "confusion_matrices.png"),
            ("A28", "boxplots.png"),
            ("A53", "histograms.png"),
        ):
            charts.insert_image(cell, str(plot_dir / filename), {"x_scale": 0.72, "y_scale": 0.72})


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

    descriptive_stats = df.describe()
    print("\nDescriptive statistics:")
    print(descriptive_stats.to_string())

    output_dir = project_dir / "plots"
    save_plots(df, output_dir)
    print(f"\nPlots saved to {output_dir}")

    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2)
    print(f"\nTraining rows: {len(y_train)}, test rows: {len(y_test)}")
    print("Training the SVC variants and running five-fold grid search...")
    scores, matrices, grid = compare_svc_models(X_train, y_train, X_test, y_test)
    print(scores.to_string(index=False, formatters={
        "Train accuracy": "{:.2%}".format, "Test accuracy": "{:.2%}".format,
    }))
    print(f"\nBest CV accuracy: {grid.best_score_:.2%}")
    print(f"Best parameters: {grid.best_params_}")

    matrix_rows = []
    for name, matrix in matrices.items():
        print(f"\n{name} confusion matrix (actual rows, predicted columns):")
        print(matrix)
        for actual, (predicted_0, predicted_1) in enumerate(matrix):
            matrix_rows.append((name, actual, predicted_0, predicted_1))
    save_confusion_heatmaps(matrices, output_dir / "confusion_matrices.png")

    cv_results = pd.DataFrame(grid.cv_results_).rename(columns={
        "rank_test_score": "Rank", "param_svc__kernel": "Kernel",
        "param_svc__C": "C", "param_svc__gamma": "Gamma",
        "mean_test_score": "Mean CV accuracy", "std_test_score": "CV accuracy std",
    })
    cv_columns = ["Rank", "Kernel", "C", "Gamma", "Mean CV accuracy", "CV accuracy std"]
    statistics = descriptive_stats.T.assign(Missing=missing_values).reset_index(names="Column")
    dataset_summary = pd.DataFrame([
        ("Rows", len(df)), ("Columns", len(df.columns)),
        ("Class 0", target_counts[0]), ("Class 1", target_counts[1]),
        ("Missing values", missing_values.sum()),
        ("Training rows", len(y_train)), ("Test rows", len(y_test)),
        ("Split seed", 42), ("CV folds", grid.n_splits_),
    ], columns=["Metric", "Value"])

    report_path = project_dir / "reports" / "pulsar_report.xlsx"
    save_excel_report(report_path, {
        "Model scores": scores,
        "Grid search": cv_results[cv_columns].sort_values("Rank", kind="stable"),
        "Confusion counts": pd.DataFrame(
            matrix_rows, columns=["Model", "Actual class", "Predicted 0", "Predicted 1"]
        ),
        "Statistics": statistics,
        "Dataset summary": dataset_summary,
        "Data": df,
    }, output_dir)
    print(f"\nExcel report saved to {report_path}")


if __name__ == "__main__":
    main()
