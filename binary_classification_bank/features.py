from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from data_io import load_dataset, save_dataset
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# ---- Кодирование порядковой переменной "education" -----------------------------


def encode_education_ordinal(features: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирование признака education с помощью OrdinalEncoder.
    Возвращает DataFrame с новым столбцом education_ordinal.
    """
    logger.info("Начинается кодирование признака education с помощью OrdinalEncoder")
    try:
        column = "education"
        education_categories = ["unknown", "primary", "secondary", "tertiary"]
        ordinal_encoder = OrdinalEncoder(categories=[education_categories])
        features[f"{column}_ordinal"] = ordinal_encoder.fit_transform(
            features[[column]]
        )
        logger.info("Кодирование признака education завершено успешно")
    except Exception as e:
        logger.error(f"Ошибка при кодировании education OrdinalEncoder: {e}")
        raise
    return features


# ----------- Конструирование новых признаков ---------------


def feature_constuction(features: pd.DataFrame) -> pd.DataFrame:
    """
    Конструирование новых признаков.
    Возвращает DataFrame с новыми признаками.
    """
    logger.info("Начинается конструирование признаков")
    try:
        features = features.copy()
        features["balance_positive"] = (features["balance"] > 0).astype(int)
        features["avg_duration_per_campaign"] = features["duration"] / (
            features["campaign"] + 1
        )
        features["pdays_negative"] = (features["pdays"] == -1).astype(int)
        features["prev_to_campaign_ratio"] = features["previous"] / (
            features["campaign"] + 1
        )
        features["marital_housing"] = features["marital"] + "_" + features["housing"]
        features["balance_log"] = np.sign(features["balance"]) * np.log1p(
            np.abs(features["balance"])
        )
        features["duration_log"] = np.log1p(features["duration"])
        features["campaign_log"] = np.log1p(features["campaign"])
        logger.info("Конструирование признаков завершено успешно")
    except Exception as e:
        logger.error(f"Ошибка при конструировании признаков: {e}")
        raise
    return features


# -------------------- Добавление новых числовых признаков (комбинации признаков) -------------


def add_numeric_combinations(features: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление новых числовых признаков (произведение пар числовых колонок).
    Возвращает DataFrame с новыми признаками.
    """
    logger.info("Начинается добавление новых числовых признаков (комбинации)")
    try:
        features = features.copy()
        numeric_cols = features.select_dtypes(include="number").columns
        new_cols = {}
        for c1, c2 in combinations(numeric_cols, 2):
            new_cols[f"{c1}_{c2}"] = features[c1] * features[c2]
        new_features = pd.DataFrame(new_cols)
        features = pd.concat([features, new_features], axis=1)
        logger.info("Добавление числовых признаков завершено")
    except Exception as e:
        logger.error(f"Ошибка при добавлении числовых признаков: {e}")
        raise
    return features


# ------------- Добавление новых категориальных признаков (комбинации) ----------------------------


def add_categorical_combinations(features: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление новых категориальных признаков (комбинации категориальных колонок).
    Возвращает DataFrame с новыми признаками.
    """
    logger.info("Начинается добавление категориальных признаков (комбинации)")
    try:
        features = features.copy()
        categorical_cols = features.select_dtypes(exclude="number").columns
        features[categorical_cols] = features[categorical_cols].astype("category")
        new_features = pd.DataFrame()
        for cols in combinations(categorical_cols, 2):
            new_col_name = "_".join(cols) + "_enc"
            new_features[new_col_name] = (
                features[list(cols)]
                .astype(str)
                .agg("_".join, axis=1)
                .astype("category")
            )
        features = pd.concat([features, new_features], axis=1)
        logger.info("Добавление категориальных признаков завершено")
    except Exception as e:
        logger.error(f"Ошибка при добавлении категориальных признаков: {e}")
        raise
    return features


# ----------------- Создание различных датасетов ---------------------------------------------


def create_datasets(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
    path_interim: Path = INTERIM_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Создаёт несколько вариантов датасетов:
      - base
      - с числовыми комбинациями
      - с категориальными комбинациями
      - со всеми комбинациями
    Сохраняет их в path_interim и возвращает словарь с DataFrame.
    """
    logger.info("Начинается создание датасетов")
    try:
        datasets = {}

        base_train = feature_constuction(features_train)
        base_val = feature_constuction(features_val)
        base_test = feature_constuction(features_test)

        datasets["features_base_train.csv"] = base_train
        datasets["features_base_val.csv"] = base_val
        datasets["features_base_test.csv"] = base_test

        num_train = add_numeric_combinations(base_train)
        num_val = add_numeric_combinations(base_val)
        num_test = add_numeric_combinations(base_test)

        datasets["features_with_numeric_train.csv"] = num_train
        datasets["features_with_numeric_val.csv"] = num_val
        datasets["features_with_numeric_test.csv"] = num_test

        cat_train = add_categorical_combinations(base_train)
        cat_val = add_categorical_combinations(base_val)
        cat_test = add_categorical_combinations(base_test)

        datasets["features_with_categorical_train.csv"] = cat_train
        datasets["features_with_categorical_val.csv"] = cat_val
        datasets["features_with_categorical_test.csv"] = cat_test

        all_train = add_categorical_combinations(add_numeric_combinations(base_train))
        all_val = add_categorical_combinations(add_numeric_combinations(base_val))
        all_test = add_categorical_combinations(add_numeric_combinations(base_test))

        datasets["features_with_all_train.csv"] = all_train
        datasets["features_with_all_val.csv"] = all_val
        datasets["features_with_all_test.csv"] = all_test

        for dataset_name, dataset in datasets.items():
            save_dataset(dataset, path_interim, dataset_name)

        logger.info("Создание датасетов успешно завершено")
        return datasets

    except Exception as e:
        logger.error(f"Ошибка при создании датасетов: {e}")
        raise


# ------------------------------------------------------------------------------------------------


def preprocess_and_pca(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
    apply_pca: bool = False,
):
    """
    Препроцессинг (шкалирование числовых + OneHotEncoder для категориальных) + PCA.
    Возвращает преобразованные numpy массивы.
    """
    logger.info("Начинается препроцессинг и PCA")
    try:
        numeric_cols = features_train.select_dtypes(include="number").columns
        categorical_cols = features_train.select_dtypes(exclude="number").columns

        features_train = features_train.copy()
        features_val = features_val.copy()
        features_test = features_test.copy()

        features_train[categorical_cols] = features_train[categorical_cols].astype(
            "category"
        )
        features_val[categorical_cols] = features_val[categorical_cols].astype(
            "category"
        )
        features_test[categorical_cols] = features_test[categorical_cols].astype(
            "category"
        )

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

        logger.info("Преобразование и PCA выполнены успешно")
        return features_train_processed, features_val_processed, features_test_processed

    except Exception as e:
        logger.error(f"Ошибка в preprocess_and_pca: {e}")
        raise


# ----------------------------------------------------------------------------------------------------------


def run_features_pipeline(
    apply_pca: bool = False,
    use_numeric_combinations: bool = False,
    use_categorical_combinations: bool = False,
):
    """
    Полный пайплайн конструирования и препроцессинга признаков:
      1. Загрузка features_train/val/test
      2. Кодирование education
      3. Создание вариантов датасетов
      4. Препроцессинг + PCA для каждого варианта
      5. Сохранение обработанных данных
    """
    path_interim = INTERIM_DATA_DIR
    path_processed = PROCESSED_DATA_DIR

    logger.info("Начинается процесс препроцессинга")

    # Загружаем признаки
    features_train = load_dataset(path_interim, "features_train.csv")
    features_val = load_dataset(path_interim, "features_val.csv")
    features_test = load_dataset(path_interim, "features_test.csv")

    if features_train is None or features_val is None or features_test is None:
        raise RuntimeError("Не удалось загрузить признаки из interim")

    # Кодируем ordinal features
    features_train = encode_education_ordinal(features_train)
    features_val = encode_education_ordinal(features_val)
    features_test = encode_education_ordinal(features_test)

    # Создаем различные варианты датасетов
    create_datasets(features_train, features_val, features_test, path_interim)

    # Названия для циклической обработки
    dataset_names = ["features_base"]

    if use_numeric_combinations:
        dataset_names.append("features_with_numeric")

    if use_categorical_combinations:
        dataset_names.append("features_with_categorical")

    if use_numeric_combinations and use_categorical_combinations:
        dataset_names.append("features_with_all")

    for dataset_name in dataset_names:
        train_name = dataset_name + "_train.csv"
        val_name = dataset_name + "_val.csv"
        test_name = dataset_name + "_test.csv"

        features_train = load_dataset(path_interim, train_name)
        features_val = load_dataset(path_interim, val_name)
        features_test = load_dataset(path_interim, test_name)

        if features_train is None:
            raise RuntimeError(f"Не удалось загрузить {train_name}")

        processed_train, processed_val, processed_test = preprocess_and_pca(
            features_train, features_val, features_test, apply_pca=apply_pca
        )

        save_dataset(
            pd.DataFrame(processed_train),
            path_processed,
            train_name.replace(".csv", "") + "_processed.csv",
        )
        save_dataset(
            pd.DataFrame(processed_val),
            path_processed,
            val_name.replace(".csv", "") + "_processed.csv",
        )
        save_dataset(
            pd.DataFrame(processed_test),
            path_processed,
            test_name.replace(".csv", "") + "_processed.csv",
        )

    logger.info("Препроцессинг и PCA успешно выполнены")
