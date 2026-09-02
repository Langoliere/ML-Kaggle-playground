from pathlib import Path

import pandas as pd
from loguru import logger

# === Загрузка данных ===


def load_dataset(
    directory: Path, filename: str, format: str = "csv"
) -> pd.DataFrame | None:
    """
    Загружает датасет из указанной папки и файла.

    Поддерживаемые форматы:
      - "csv" -> pd.read_csv
      - "parquet" -> pd.read_parquet

    Возвращает:
      - DataFrame, если загрузка успешна
      - None, если произошла ошибка
    """
    path = directory / filename

    logger.info(f"Загрузка данных из: {path}")

    try:
        if format == "csv":
            df = pd.read_csv(path)
        elif format == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {format}. Поддерживаются: csv, parquet"
            )

        logger.info(f"Данные успешно загружены из {path}, размер: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {path}: {e}")
        return None


# === Сохранение данных ===


def save_dataset(
    df: pd.DataFrame,
    directory: Path,
    filename: str,
    format: str = "csv",
) -> bool:
    """
    Сохраняет DataFrame в указанную папку и файл.

    Поддерживаемые форматы:
      - "csv" -> df.to_csv
      - "parquet" -> df.to_parquet

    Возвращает:
      - True, если сохранение успешно
      - False, если произошла ошибка
    """
    path = directory / filename

    logger.info(f"Сохранение данных в: {path}")

    try:
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path)
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {format}. Поддерживаются: csv, parquet"
            )

        logger.info(f"Данные успешно сохранены в {path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при сохранении данных в {path}: {e}")
        return False
