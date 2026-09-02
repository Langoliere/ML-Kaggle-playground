
import numpy as np
import pandas as pd
from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_io import load_dataset, save_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

# ------ Обработка выбросов ---------------------------------------------


def handle_outliers_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка выбросов логарифмическим методом для всех числовых колонок.
    Возвращает очищенный DataFrame.
    """
    logger.info("Начинается обработка выбросов логарифмическим методом")
    try:
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
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]

        shape_after = df_clean.shape[0]
        logger.info(
            f"Оставлено {shape_after / shape_before * 100:.2f}% строк после удаления выбросов"
        )
        return df_clean

    except Exception as e:
        logger.error(f"Ошибка при обработке выбросов: {e}")
        raise


# ----------------- Оптимизация типов для числовых данных ------------------------


def optimize_numeric_types(df: pd.DataFrame, margin: float = 0.9) -> pd.DataFrame:
    """
    Оптимизация типов числовых данных для уменьшения потребления памяти.
    Возвращает DataFrame с оптимизированными типами.
    """
    logger.info("Начинается оптимизация типов числовых данных")
    try:
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

                    if int8_min <= c_min and c_max <= int8_max:
                        df[col] = df[col].astype(np.int8)
                    elif int16_min <= c_min and c_max <= int16_max:
                        df[col] = df[col].astype(np.int16)
                    elif int32_min <= c_min and c_max <= int32_max:
                        df[col] = df[col].astype(np.int32)
                    # else оставляем int64 по умолчанию

                else:  # float
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
                    # else оставляем float64

        logger.info("Оптимизация типов числовых данных выполнена успешно")
        return df

    except Exception as e:
        logger.error(f"Ошибка при оптимизации типов числовых данных: {e}")
        raise


# -----------------------------------------

# ------------- Разделение на признаки и целевую переменную ----------------------------


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет DataFrame на признаки (X) и целевую переменную (y).
    Возвращает (features, target).
    """
    logger.info("Начинается разделение признаков и целевой переменной")
    try:
        features = df.drop([target_col], axis=1)
        target = df[target_col]
        logger.info("Разделение признаков и целевой переменной выполнено успешно")
        return features, target

    except Exception as e:
        logger.error(f"Ошибка при разделении признаков и целевой переменной: {e}")
        raise


# -----------------------------------------


def split_train_val_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.4,
    val_size: float = 0.5,
    random_state: int = 42,
    stratify_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Разделяет данные на train, val и test.
    Возвращает:
      (features_train, features_val, features_test,
       target_train, target_val, target_test)
    """
    logger.info("Начинается разделение данных на train, val и test")
    try:
        stratify_param = target if stratify_target else None

        features_train, features_val_test, target_train, target_val_test = (
            train_test_split(
                features,
                target,
                random_state=random_state,
                test_size=test_size,
                stratify=stratify_param,
            )
        )

        stratify_val = target_val_test if stratify_target else None

        features_val, features_test, target_val, target_test = train_test_split(
            features_val_test,
            target_val_test,
            random_state=random_state,
            test_size=val_size,
            stratify=stratify_val,
        )

        logger.info("Данные успешно разделены на train, val и test")
        return (
            features_train,
            features_val,
            features_test,
            target_train,
            target_val,
            target_test,
        )

    except Exception as e:
        logger.error(f"Ошибка при разделении данных: {e}")
        raise


# ------------------------------------------------


def run_dataset_pipeline(
    target_col: str = "y",
    test_size: float = 0.4,
    val_size: float = 0.5,
    random_state: int = 42,
    stratify_target: bool = True,
    margin: float = 0.9,
):
    """
    Полный пайплайн обработки данных:
      1. Загрузка raw
      2. Обработка выбросов
      3. Оптимизация типов
      4. Разделение на X и y
      5. Разделение на train/val/test
      6. Сохранение в interim и processed
    """
    path_raw = RAW_DATA_DIR
    path_interim = INTERIM_DATA_DIR
    path_processed = PROCESSED_DATA_DIR

    # 1. Загрузка
    df = load_dataset(path_raw, "data.csv")
    if df is None:
        raise RuntimeError("Не удалось загрузить данные из data.csv")

    # 2. Выбросы
    df_clean = handle_outliers_log(df)
    save_dataset(df_clean, path_interim, "no_outliers.csv")

    # 3. Оптимизация типов
    df_opt = optimize_numeric_types(df_clean, margin)
    save_dataset(df_opt, path_interim, "opt_types.csv")

    # 4. X и y
    features, target = split_features_target(df_opt, target_col)
    save_dataset(features, path_interim, "features.csv")
    save_dataset(target, path_interim, "target.csv")

    # 5. Train/val/test
    (
        features_train,
        features_val,
        features_test,
        target_train,
        target_val,
        target_test,
    ) = split_train_val_test(
        features,
        target,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify_target=stratify_target,
    )

    # 6. Сохранение
    save_dataset(features_train, path_interim, "features_train.csv")
    save_dataset(features_val, path_interim, "features_val.csv")
    save_dataset(features_test, path_interim, "features_test.csv")
    save_dataset(target_train, path_interim, "target_train.csv")
    save_dataset(target_val, path_interim, "target_val.csv")
    save_dataset(target_test, path_interim, "target_test.csv")

    save_dataset(target_train, path_processed, "target_train.csv")
    save_dataset(target_val, path_processed, "target_val.csv")
    save_dataset(target_test, path_processed, "target_test.csv")

    logger.info("Пайплайн обработки данных завершён успешно")
