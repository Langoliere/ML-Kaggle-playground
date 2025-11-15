from itertools import combinations
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
import typer

from binary_classification_bank.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from binary_classification_bank.data_io import load_config, load_dataset, save_dataset

app = typer.Typer()


# ---- Кодирование порядковой переменной "education" -----------------------------------------------------
def encode_education_ordinal(features: pd.DataFrame) -> pd.DataFrame:
    try:
        typer.secho("Начинается кодирование признака education с помощью OrdinalEncoder")
        column = "education"
        education_categories = ["unknown", "primary", "secondary", "tertiary"]
        ordinal_encoder = OrdinalEncoder(categories=[education_categories])
        features[f"{column}_ordinal"] = ordinal_encoder.fit_transform(features[[column]])
    except Exception as e:
        typer.secho(f"Ошибка при кодировании education OrdinalEncoder: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Кодирование признака education завершено успешно", fg=typer.colors.GREEN)
    return features


# ----------- Конструирование новых признаков ---------------
def feature_constuction(features: pd.DataFrame) -> pd.DataFrame:
    try:
        features["balance_positive"] = (features["balance"] > 0).astype(int)
        features["avg_duration_per_campaign"] = features["duration"] / (features["campaign"] + 1)
        features["pdays_negative"] = (features["pdays"] == -1).astype(int)
        features["prev_to_campaign_ratio"] = features["previous"] / (features["campaign"] + 1)
        features["marital_housing"] = features["marital"] + "_" + features["housing"]
        features["balance_log"] = np.sign(features["balance"]) * np.log1p(
            np.abs(features["balance"])
        )
        features["duration_log"] = np.log1p(features["duration"])
        features["campaign_log"] = np.log1p(features["campaign"])
    except Exception as e:
        typer.secho(f"Ошибка при конструировании признаков: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Конструирование признаков завершено успешно", fg=typer.colors.GREEN)
    return features
    # -----------------------------------------


# -------------------- Добавление новых числовых признаков (комбинации признаков) -------------
def add_numeric_combinations(features: pd.DataFrame) -> pd.DataFrame:
    try:
        typer.secho("Начинается добавление новых числовых признаков (комбинации)")
        numeric_cols = features.select_dtypes(include="number").columns
        new_cols = {}
        for c1, c2 in combinations(numeric_cols, 2):
            new_cols[f"{c1}_{c2}"] = features[c1] * features[c2]
        new_features = pd.DataFrame(new_cols)
        features = pd.concat([features, new_features], axis=1)
    except Exception as e:
        typer.secho(f"Ошибка при добавлении числовых признаков: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Добавление числовых признаков завершено", fg=typer.colors.GREEN)
    return features


# ------------------------------------------------------------------------------------------------


# ------------- Добавление новых категориальных признаков (комбинации) ----------------------------
def add_categorical_combinations(features: pd.DataFrame) -> pd.DataFrame:
    try:
        typer.secho("Начинается добавление категориальных признаков (комбинации)")
        categorical_cols = features.select_dtypes(exclude="number").columns
        features[categorical_cols] = features[categorical_cols].astype("category")
        new_features = pd.DataFrame()
        for cols in combinations(categorical_cols, 2):
            new_col_name = "_".join(cols) + "_enc"
            new_features[new_col_name] = (
                features[list(cols)].astype(str).agg("_".join, axis=1).astype("category")
            )
        features = pd.concat([features, new_features], axis=1)
    except Exception as e:
        typer.secho(f"Ошибка при добавлении категориальных признаков: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Добавление категориальных признаков завершено", fg=typer.colors.GREEN)
    return features


# --------------------------------------------------------------------------------------------------------------


# ----------------- Создание различных датасетов ---------------------------------------------


def create_datasets(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
    path_interrim: Path = INTERIM_DATA_DIR,
):
    typer.secho("Начинается создание датасетов")
    try:
        datasets = {}

        base_train = features_train.copy()
        base_val = features_val.copy()
        base_test = features_test.copy()

        # Базовые признаки
        base_train = feature_constuction(base_train)
        base_val = feature_constuction(base_val)
        base_test = feature_constuction(base_test)

        datasets["features_base_train.csv"] = base_train
        datasets["features_base_val.csv"] = base_val
        datasets["features_base_test.csv"] = base_test

        # С числовыми комбинациями
        num_train = add_numeric_combinations(base_train.copy())
        num_val = add_numeric_combinations(base_val.copy())
        num_test = add_numeric_combinations(base_test.copy())

        datasets["features_with_numeric_train.csv"] = num_train
        datasets["features_with_numeric_val.csv"] = num_val
        datasets["features_with_numeric_test.csv"] = num_test

        # С категориальными комбинациями
        cat_train = add_categorical_combinations(base_train.copy())
        cat_val = add_categorical_combinations(base_val.copy())
        cat_test = add_categorical_combinations(base_test.copy())

        datasets["features_with_categorical_train.csv"] = cat_train
        datasets["features_with_categorical_val.csv"] = cat_val
        datasets["features_with_categorical_test.csv"] = cat_test

        # Со всеми комбинациями
        all_train = add_numeric_combinations(base_train.copy())
        all_train = add_categorical_combinations(all_train)
        all_val = add_numeric_combinations(base_val.copy())
        all_val = add_categorical_combinations(all_val)
        all_test = add_numeric_combinations(base_test.copy())
        all_test = add_categorical_combinations(all_test)

        datasets["features_with_all_train.csv"] = all_train
        datasets["features_with_all_val.csv"] = all_val
        datasets["features_with_all_test.csv"] = all_test

        # Сохраняем все датасеты
        for dataset_name, dataset in datasets.items():
            save_dataset(dataset, path_interrim, dataset_name)

    except Exception as e:
        typer.secho(f"Ошибка при создании датасетов: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Создание датасетов успешно завершено", fg=typer.colors.GREEN)


# ----------------------------------------------------------------------------------------------------------


def preprocess_and_pca(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
    apply_pca=False,
):
    try:
        typer.secho("Начинается препроцессинг и PCA")
        numeric_cols = features_train.select_dtypes(include="number").columns
        categorical_cols = features_train.select_dtypes(exclude="number").columns

        features_train[categorical_cols] = features_train[categorical_cols].astype("category")
        features_val[categorical_cols] = features_val[categorical_cols].astype("category")
        features_test[categorical_cols] = features_test[categorical_cols].astype("category")

        preprocessor = ColumnTransformer(
            [
                ("num", RobustScaler(), numeric_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                ),
            ]
        )

        features_train_processed = preprocessor.fit_transform(features_train)
        features_val_processed = preprocessor.transform(features_val)
        features_test_processed = preprocessor.transform(features_test)

        if apply_pca:
            pca = PCA(n_components=0.95)
            features_train_processed = pca.fit_transform(features_train_processed)
            features_val_processed = pca.transform(features_val_processed)
            features_test_processed = pca.transform(features_test_processed)

    except Exception as e:
        typer.secho(f"Ошибка в preprocess_and_pca: {e}", fg=typer.colors.RED)
        raise
    else:
        typer.secho("Преобразование и PCA выполнены успешно", fg=typer.colors.GREEN)
    return features_train_processed, features_val_processed, features_test_processed


@app.command()
def main(path_interrim: Path = INTERIM_DATA_DIR, path_processed: Path = PROCESSED_DATA_DIR):
    typer.secho("Начинается процесс препроцессинга")
    try:
        # Загружаем признаки
        features_train = load_dataset(path_interrim, "features_train.csv")
        features_val = load_dataset(path_interrim, "features_val.csv")
        features_test = load_dataset(path_interrim, "features_test.csv")

        # Кодируем ordinal features
        features_train = encode_education_ordinal(features_train)
        features_val = encode_education_ordinal(features_val)
        features_test = encode_education_ordinal(features_test)

        # Создаем различные варианты датасетов
        create_datasets(features_train, features_val, features_test, path_interrim)

        # Названия для циклической обработки
        dataset_names = [
            "features_base",
            "features_with_numeric",
            "features_with_categorical",
            "features_with_all",
        ]

        for dataset_name in dataset_names:
            train_name = dataset_name + "_train.csv"
            val_name = dataset_name + "_val.csv"
            test_name = dataset_name + "_test.csv"

            features_train = load_dataset(path_interrim, train_name)
            features_val = load_dataset(path_interrim, val_name)
            features_test = load_dataset(path_interrim, test_name)

            processed_train, processed_val, processed_test = preprocess_and_pca(
                features_train, features_val, features_test
            )

            save_dataset(
                pd.DataFrame(processed_train),
                path_processed,
                train_name.replace(".csv", "") + "_processed" + ".csv",
            )
            save_dataset(
                pd.DataFrame(processed_val),
                path_processed,
                val_name.replace(".csv", "") + "_processed" + ".csv",
            )
            save_dataset(
                pd.DataFrame(processed_test),
                path_processed,
                test_name.replace(".csv", "") + "_processed" + ".csv",
            )

    except:
        typer.secho("Ошибка препроцессинга и PCA.", fg=typer.colors.RED)
    else:
        typer.secho("Препроцессинг и PCA успешно выполнены.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
