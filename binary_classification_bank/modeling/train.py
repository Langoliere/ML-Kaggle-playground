from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import yaml
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import mlflow
import mlflow.xgboost

from binary_classification_bank.config import MODELS_DIR, PROCESSED_DATA_DIR
from binary_classification_bank.data_io import load_config, load_dataset, save_dataset

app = typer.Typer()


@app.command()
def train_model():
    # Пути для признаков и таргетов
    features_train_name = "features_base_train_processed.csv"
    target_train_name = "target_train.csv"
    features_val_name = "features_base_val_processed.csv"
    target_val_name = "target_val.csv"
    features_test_name = "features_base_test_processed.csv"
    target_test_name = "target_test.csv"

    # Загрузка датасетов
    features_train = load_dataset(PROCESSED_DATA_DIR, features_train_name)
    target_train = load_dataset(PROCESSED_DATA_DIR, target_train_name)

    features_val = load_dataset(PROCESSED_DATA_DIR, features_val_name)
    target_val = load_dataset(PROCESSED_DATA_DIR, target_val_name)

    features_test = load_dataset(PROCESSED_DATA_DIR, features_test_name)
    target_test = load_dataset(PROCESSED_DATA_DIR, target_test_name)

    # Загрузка гиперпараметров
    with open("configs/best_hyperparams.yaml", "r", encoding="utf-8") as f:
        best_params = yaml.safe_load(f)

    # Создание модели
    best_xgb = XGBClassifier(**best_params)

    # Объединяем датасеты
    features_train_val = pd.concat([features_train, features_val], ignore_index=True)
    target_train_val = pd.concat([target_train, target_val], ignore_index=True)

    artifact_path = Path(
        "E:/Pet Projects/Kaggle Playground/binary_classification_bank/models/mlflow_artifacts"
    ).as_uri()
    mlflow.set_tracking_uri(artifact_path)

    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params(best_params)

        # Обучаем модель
        best_xgb.fit(features_train_val, target_train_val)

        # Предсказания на тесте
        xgb_y_pred_proba = best_xgb.predict_proba(features_test)[:, 1]

        # Вычисляем метрику
        xgb_roc_auc = roc_auc_score(target_test, xgb_y_pred_proba)

        # Логируем метрику
        mlflow.log_metric("roc_auc", xgb_roc_auc)

        # Логируем модель
        mlflow.xgboost.log_model(best_xgb, artifact_path="xgb_model")

        typer.secho(f"Модель обучена с ROC AUC: {xgb_roc_auc}")

        # Сохраняем модель локально
        model_path = Path("models/best_model.json")
        best_xgb.save_model(str(model_path))


@app.command()
def main():
    train_model()


if __name__ == "__main__":
    app()
