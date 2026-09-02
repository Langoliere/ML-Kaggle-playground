import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import FIGURES_DIR, RAW_DATA_DIR
from loguru import logger

# === Функции для построения графиков ===


def plot_feature_target_distributions(
    df: pd.DataFrame,
    target_col: str = "y",
    bins: int = 10,
    output_path: Path = FIGURES_DIR,
) -> Path:
    """
    Строит распределения целевой переменной по всем признакам (категориальным и числовым).
    Сохраняет график в output_path / "feature_target_distributions.png".
    Возвращает путь к сохранённому файлу.
    """
    logger.info("Начинается построение распределений feature-target")
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
                df_temp = df.copy()
                df_temp["bin"] = pd.cut(df_temp[col], bins=bins)
                grouped = (
                    df_temp.groupby(["bin", target_col])
                    .size()
                    .reset_index(name="count")
                )
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
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "feature_target_distributions.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        logger.info(f"Графики распределений сохранены: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Ошибка при построении распределений: {e}")
        raise


def plot_boxplots(df: pd.DataFrame, output_path: Path = FIGURES_DIR) -> Path:
    """
    Строит boxplots для всех числовых признаков.
    Сохраняет график в output_path / "boxplots.png".
    Возвращает путь к сохранённому файлу.
    """
    logger.info("Начинается построение boxplots")
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

        logger.info(f"Boxplots сохранены: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Ошибка при построении boxplots: {e}")
        raise


def run_plots_pipeline(
    input_path: Path = RAW_DATA_DIR / "data.csv",
) -> tuple[Path, Path]:
    """
    Полный пайплайн построения графиков:
      1. Загрузка данных из input_path
      2. Построение распределений feature-target
      3. Построение boxplots
      4. Сохранение графиков в FIGURES_DIR
    Возвращает tuple (path_feature_target, path_boxplots).
    """
    logger.info("Начинается процесс построения графиков")

    df = pd.read_csv(input_path)
    logger.info(f"Данные загружены из {input_path}, размер: {df.shape}")

    path_feature_target = plot_feature_target_distributions(df)
    path_boxplots = plot_boxplots(df)

    logger.info("Построение графиков завершено успешно")
    return path_feature_target, path_boxplots
