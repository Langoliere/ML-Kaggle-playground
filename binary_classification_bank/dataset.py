from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

from binary_classification_bank.config import INTERIM_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from binary_classification_bank.data_io import load_config, load_dataset, save_dataset

app = typer.Typer()


# ------ Обработка выбросов ---------------------------------------------
@app.command()
def handle_outliers_log(path_raw: Path = RAW_DATA_DIR, path_interrim: Path = INTERIM_DATA_DIR):
    typer.secho("Начинается обработка выбросов логарифмическим методом")
    try:
        df = load_dataset(path_raw, "data.csv")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df_clean = df.copy()
        shape_before = df_clean.shape[0]
        for col in numeric_cols:
            # Пропускаем отрицательные и нулевые значения для логарифма
            if (df_clean[col] <= 0).any():
                continue

            # Логарифмируем данные
            log_data = np.log(df_clean[col])

            # Расчет квартилей и IQR для логарифмированных данных
            Q1 = log_data.quantile(0.25)
            Q3 = log_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound_log = Q1 - 1.5 * IQR
            upper_bound_log = Q3 + 1.5 * IQR

            # Экспонента границ для фильтрации изначальных данных
            lower_bound = np.exp(lower_bound_log)
            upper_bound = np.exp(upper_bound_log)

            # Фильтрация выбросов по исходным данным
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        shape_after = df_clean.shape[0]
        typer.secho(
            f"Оставлено {shape_after / shape_before * 100:.2f}% строк после удаления выбросов"
        )
        save_dataset(df_clean, path_interrim, "no_outliers.csv")
    except Exception as e:
        typer.secho(f"Ошибка при обработке выбросов: {e}", fg=typer.colors.RED)
    else:
        typer.secho("Обработка выбросов завершена успешно", fg=typer.colors.GREEN)
    # ---------------------------------------------------------------------------------


# ----------------- Оптимизация типов для числовых данных ------------------------


@app.command()
def optimize_numeric_types(path_interrim: Path = INTERIM_DATA_DIR, margin: float = 0.9):
    typer.secho("Начинается оптимизация типов числовых данных")
    try:
        df = load_dataset(path_interrim, "no_outliers.csv")
        numerics = [
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "uint16",
            "uint32",
            "uint64",
        ]
        for col in df.columns:
            col_type = df[col].dtypes

            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()

                if "int" in str(col_type):
                    int8_min, int8_max = (
                        np.iinfo(np.int8).min * margin,
                        np.iinfo(np.int8).max * margin,
                    )
                    int16_min, int16_max = (
                        np.iinfo(np.int16).min * margin,
                        np.iinfo(np.int16).max * margin,
                    )
                    int32_min, int32_max = (
                        np.iinfo(np.int32).min * margin,
                        np.iinfo(np.int32).max * margin,
                    )
                    int64_min, int64_max = (
                        np.iinfo(np.int64).min * margin,
                        np.iinfo(np.int64).max * margin,
                    )

                    if int8_min <= c_min and c_max <= int8_max:
                        df[col] = df[col].astype(np.int8)
                    elif int16_min <= c_min and c_max <= int16_max:
                        df[col] = df[col].astype(np.int16)
                    elif int32_min <= c_min and c_max <= int32_max:
                        df[col] = df[col].astype(np.int32)
                    elif int64_min <= c_min and c_max <= int64_max:
                        df[col] = df[col].astype(np.int64)
                else:
                    float16_min, float16_max = (
                        np.finfo(np.float16).min * margin,
                        np.finfo(np.float16).max * margin,
                    )
                    float32_min, float32_max = (
                        np.finfo(np.float32).min * margin,
                        np.finfo(np.float32).max * margin,
                    )

                    if float16_min <= c_min and c_max <= float16_max:
                        df[col] = df[col].astype(np.float16)
                    elif float32_min <= c_min and c_max <= float32_max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        save_dataset(df, path_interrim, "opt_types.csv")
    except Exception as e:
        typer.secho(f"Ошибка при оптимизации типов числовых данных: {e}", fg=typer.colors.RED)
    else:
        typer.secho("Оптимизация типов числовых данных выполнена успешно", fg=typer.colors.GREEN)


# -----------------------------------------

# ------------- Разделение на признаки и целевую переменную ----------------------------


@app.command()
def split_features_target(path_interrim: Path = INTERIM_DATA_DIR, target_col: str = "y"):
    typer.secho("Начинается разделение признаков и целевой переменной")
    try:
        df = load_dataset(path_interrim, "opt_types.csv")

        features = df.drop([target_col], axis=1)
        target = df[target_col]
        save_dataset(features, path_interrim, "features.csv")
        save_dataset(target, path_interrim, "target.csv")
    except Exception as e:
        typer.secho(
            f"Ошибка при разделении признаков и целевой переменной: {e}", fg=typer.colors.RED
        )
    else:
        typer.secho(
            "Разделение признаков и целевой переменной выполнено успешно", fg=typer.colors.GREEN
        )
    # -----------------------------------------


@app.command()
def split_train_val_test(
    path_interrim: Path = INTERIM_DATA_DIR,
    path_processed: Path = PROCESSED_DATA_DIR,
    test_size: float = 0.4,
    val_size: float = 0.5,
    random_state: int = 42,
    stratify_target: bool = True,
):
    typer.secho("Начинается разделение данных на train, val и test")
    try:
        features = load_dataset(path_interrim, "features.csv")
        target = load_dataset(path_interrim, "target.csv")
        stratify_param = target if stratify_target else None

        features_train, features_val_test, target_train, target_val_test = train_test_split(
            features,
            target,
            random_state=random_state,
            test_size=test_size,
            stratify=stratify_param,
        )

        stratify_val = target_val_test if stratify_target else None

        features_val, features_test, target_val, target_test = train_test_split(
            features_val_test,
            target_val_test,
            random_state=random_state,
            test_size=val_size,
            stratify=stratify_val,
        )

        save_dataset(features_train, path_interrim, "features_train.csv")
        save_dataset(features_val, path_interrim, "features_val.csv")
        save_dataset(features_test, path_interrim, "features_test.csv")
        save_dataset(target_train, path_interrim, "target_train.csv")
        save_dataset(target_val, path_interrim, "target_val.csv")
        save_dataset(target_test, path_interrim, "target_test.csv")

        save_dataset(target_train, path_processed, "target_train.csv")
        save_dataset(target_val, path_processed, "target_val.csv")
        save_dataset(target_test, path_processed, "target_test.csv")
    except Exception as e:
        typer.secho(f"Ошибка при разделении данных: {e}", fg=typer.colors.RED)
    else:
        typer.secho("Данные успешно разделены на train, val и test", fg=typer.colors.GREEN)


# ------------------------------------------------


@app.command()
def main(
    margin: float = 0.9,
    target_col: str = "y",
    test_size: float = 0.4,
    val_size: float = 0.5,
    random_state: int = 42,
    stratify_target: bool = True,
):
    path_raw: Path = RAW_DATA_DIR
    path_interrim: Path = INTERIM_DATA_DIR
    path_processed: Path = PROCESSED_DATA_DIR

    handle_outliers_log(path_raw, path_interrim)
    optimize_numeric_types(path_interrim, margin)
    split_features_target(path_interrim, target_col)
    split_train_val_test(
        path_interrim, path_processed, test_size, val_size, random_state, stratify_target
    )


if __name__ == "__main__":
    app()
