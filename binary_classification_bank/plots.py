from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from binary_classification_bank.config import FIGURES_DIR, RAW_DATA_DIR

app = typer.Typer()


def plot_feature_target_distributions(
    df: pd.DataFrame, target_col="y", bins=10, output_path=FIGURES_DIR
):
    try:
        num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        features = list(cat_cols) + list(num_cols)
        num_features = len(features)
        cols = 2
        rows = math.ceil(num_features / cols)
        plt.figure(figsize=(cols * 6, rows * 5))

        for i, col in enumerate(features):
            plt.subplot(rows, cols, i + 1)
            if col in cat_cols:
                grouped = df.groupby([col, target_col]).size().reset_index(name="count")
                grouped["percent"] = grouped.groupby(col)["count"].transform(
                    lambda x: 100 * x / x.sum()
                )
                sns.barplot(data=grouped, x=col, y="percent", hue=target_col)
            else:
                df["bin"] = pd.cut(df[col], bins=bins)
                grouped = df.groupby(["bin", target_col]).size().reset_index(name="count")
                grouped["percent"] = grouped.groupby("bin")["count"].transform(
                    lambda x: 100 * x / x.sum()
                )
                grouped[col] = grouped["bin"].astype(str)
                grouped.drop(columns=["bin"], inplace=True)
                sns.barplot(data=grouped, x=col, y="percent", hue=target_col)
            plt.title(f"Распределение {target_col} по признаку {col}")
            plt.xticks(rotation=45)
            plt.ylabel("Процент")
            plt.legend(title=target_col)

        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "feature_target_distributions.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        typer.secho(f"Графики распределений сохранены: {file_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Ошибка при построении распределений: {e}", fg=typer.colors.RED)


def plot_boxplots(df: pd.DataFrame, output_path=FIGURES_DIR):
    try:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        cols = 3
        rows = (len(numeric_cols) + cols - 1) // cols
        plt.figure(figsize=(cols * 6, rows * 5))
        for i, col in enumerate(numeric_cols):
            plt.subplot(rows, cols, i + 1)
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "boxplots.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        typer.secho(f"Boxplots сохранены: {file_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Ошибка при построении boxplots: {e}", fg=typer.colors.RED)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "data.csv",
    output_path: Path = FIGURES_DIR,
):
    df = pd.read_csv(input_path)
    plot_feature_target_distributions(df, output_path=output_path)
    plot_boxplots(df, output_path=output_path)


if __name__ == "__main__":
    app()
