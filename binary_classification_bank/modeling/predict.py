from pathlib import Path

import pandas as pd
from config import PROCESSED_DATA_DIR
from loguru import logger
from xgboost import XGBClassifier


def predict(
    features_path: Path = PROCESSED_DATA_DIR,
    features_file: str = "features_base_test_processed.csv",
    model_path: Path = Path("models/best_model.json"),
    predictions_path: Path = Path("models/predictions.csv"),
    threshold: float = 0.5,
):
    """
    Инференс обученной модели XGBClassifier из локального файла.

    Параметры:
      - features_path: папка с features
      - features_file: имя файла с features
      - model_path: путь к сохранённой модели (best_model.json от train.py)
      - predictions_path: путь для сохранения предсказаний
      - threshold: порог для бинарного классификатора

    Возвращает:
      - dict с {predictions_proba, predictions}
    """
    logger.info("Начинается инференс модели")

    # Загрузка модели
    logger.info(f"Загрузка модели из {model_path}")
    model = XGBClassifier()
    model.load_model(str(model_path))
    logger.info("Модель загружена успешно")

    # Загрузка features
    logger.info(f"Загрузка features из {features_path / features_file}")
    features = pd.read_csv(features_path / features_file)
    if features is None or features.empty:
        raise RuntimeError(
            f"Не удалось загрузить features из {features_path / features_file}"
        )

    logger.info(
        f"Размер features: {features.shape[0]} строк, {features.shape[1]} фичей"
    )

    # Выполнение предсказаний
    logger.info("Выполнение предсказаний...")
    try:
        predictions_proba = model.predict_proba(features)[:, 1]
        predictions = (predictions_proba >= threshold).astype(int)
        logger.info(
            f"Предсказания выполнены: {predictions.sum()} позитивных, {len(predictions) - predictions.sum()} негативных"
        )
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказаний: {e}")
        raise

    # Сохранение предсказаний
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "predicted_proba": predictions_proba,
            "predicted": predictions,
        }
    )

    pred_df.to_csv(predictions_path, index=False)
    logger.info(f"Предсказания сохранены: {predictions_path}")

    logger.info("Инференс завершён успешно")

    return {
        "predictions_proba": predictions_proba,
        "predictions": predictions,
    }


def run_predict_pipeline(
    features_path: Path = PROCESSED_DATA_DIR,
    features_file: str = "features_base_test_processed.csv",
    model_path: Path = Path("models/best_model.json"),
    predictions_path: Path = Path("models/predictions.csv"),
    threshold: float = 0.5,
):
    """
    Оркестрационная функция для инференса.
    Вызывает predict() с параметрами по умолчанию.
    """
    logger.info("Начинается пайплайн инференса")

    result = predict(
        features_path=features_path,
        features_file=features_file,
        model_path=model_path,
        predictions_path=predictions_path,
        threshold=threshold,
    )

    logger.info("Пайплайн инференса завершён успешно")
    return result
