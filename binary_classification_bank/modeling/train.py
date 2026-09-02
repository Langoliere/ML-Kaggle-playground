from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from config import PROCESSED_DATA_DIR
from data_io import load_dataset
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier


def train_model(
    best_params_path: Path = Path("configs/best_hyperparams.yaml"),
    features_train_name: str = "features_base_train_processed.csv",
    target_train_name: str = "target_train.csv",
    features_val_name: str = "features_base_val_processed.csv",
    target_val_name: str = "target_val.csv",
    features_test_name: str = "features_base_test_processed.csv",
    target_test_name: str = "target_test.csv",
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = "xgb_model_training",
    log_model_to_mlflow: bool = True,
    save_model_local: bool = True,
    local_model_path: Path = Path("models/best_model.json"),
    threshold_overfitting: float = 0.1,
):
    """
    Обучение финальной модели XGBClassifier с лучшими гиперпараметрами.

    Параметры:
      - best_params_path: путь к best_hyperparams.yaml
      - features_train_name, target_train_name: имена файлов train
      - features_val_name, target_val_name: имена файлов val
      - features_test_name, target_test_name: имена файлов test
      - processed_data_dir: папка с обработанными данными
      - mlflow_tracking_uri: URI для MLflow (по умолчанию берётся из окружения)
      - mlflow_experiment_name: имя эксперимента в MLflow
      - log_model_to_mlflow: логировать модель в MLflow или нет
      - save_model_local: сохранять модель локально в JSON или нет
      - local_model_path: путь для локального сохранения модели
      - threshold_overfitting: порог разницы ROC AUC train-test для предупреждения о переобучении

    Возвращает:
      - dict с {run_id, roc_auc_train, roc_auc_val, roc_auc_test, pr_auc_train, pr_auc_val, pr_auc_test, model, params, feature_importance}
    """
    logger.info("Начинается обучение финальной модели")

    # Загрузка гиперпараметров
    logger.info(f"Загрузка гиперпараметров из {best_params_path}")
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = yaml.safe_load(f)

    # Загрузка датасетов
    logger.info("Загрузка обучающих данных")
    features_train = load_dataset(processed_data_dir, features_train_name)
    target_train = load_dataset(processed_data_dir, target_train_name)

    logger.info("Загрузка валидационных данных")
    features_val = load_dataset(processed_data_dir, features_val_name)
    target_val = load_dataset(processed_data_dir, target_val_name)

    logger.info("Загрузка тестовых данных")
    features_test = load_dataset(processed_data_dir, features_test_name)
    target_test = load_dataset(processed_data_dir, target_test_name)

    if features_train is None or target_train is None:
        raise RuntimeError("Не удалось загрузить обучающие данные")
    if features_val is None or target_val is None:
        raise RuntimeError("Не удалось загрузить валидационные данные")
    if features_test is None or target_test is None:
        raise RuntimeError("Не удалось загрузить тестовые данные")

    logger.info(
        f"Размер train: {features_train.shape[0]} строк, {features_train.shape[1]} фичей"
    )
    logger.info(
        f"Размер val: {features_val.shape[0]} строк, {features_val.shape[1]} фичей"
    )
    logger.info(
        f"Размер test: {features_test.shape[0]} строк, {features_test.shape[1]} фичей"
    )

    # Создание модели
    best_xgb = XGBClassifier(**best_params)

    # Объединяем train + val для финального обучения
    features_train_val = pd.concat([features_train, features_val], ignore_index=True)
    target_train_val = pd.concat([target_train, target_val], ignore_index=True)

    logger.info(
        f"Размер train+val (для обучения): {features_train_val.shape[0]} строк, {features_train_val.shape[1]} фичей"
    )

    # Настройка MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Логируем параметры
        mlflow.log_params(best_params)

        # Обучаем модель
        logger.info("Обучение модели")
        best_xgb.fit(features_train_val, target_train_val)

        # Предсказания на train
        train_pred_proba = best_xgb.predict_proba(features_train_val)[:, 1]
        train_roc_auc = roc_auc_score(target_train_val, train_pred_proba)
        train_pr_auc = average_precision_score(target_train_val, train_pred_proba)
        logger.info(
            f"ROC AUC на train+val: {train_roc_auc:.4f}, PR-AUC: {train_pr_auc:.4f}"
        )
        mlflow.log_metric("roc_auc_train", train_roc_auc)
        mlflow.log_metric("pr_auc_train", train_pr_auc)

        # Предсказания на val (отдельно)
        val_pred_proba = best_xgb.predict_proba(features_val)[:, 1]
        val_roc_auc = roc_auc_score(target_val, val_pred_proba)
        val_pr_auc = average_precision_score(target_val, val_pred_proba)
        logger.info(f"ROC AUC на val: {val_roc_auc:.4f}, PR-AUC: {val_pr_auc:.4f}")
        mlflow.log_metric("roc_auc_val", val_roc_auc)
        mlflow.log_metric("pr_auc_val", val_pr_auc)

        # Предсказания на тесте
        test_pred_proba = best_xgb.predict_proba(features_test)[:, 1]
        test_roc_auc = roc_auc_score(target_test, test_pred_proba)
        test_pr_auc = average_precision_score(target_test, test_pred_proba)
        logger.info(f"ROC AUC на тесте: {test_roc_auc:.4f}, PR-AUC: {test_pr_auc:.4f}")
        mlflow.log_metric("roc_auc_test", test_roc_auc)
        mlflow.log_metric("pr_auc_test", test_pr_auc)

        # Проверка на переобучение (по ROC AUC)
        overfitting_diff_roc = train_roc_auc - test_roc_auc
        if overfitting_diff_roc > threshold_overfitting:
            logger.warning(
                f"Высокая разница между train и test ROC AUC: {overfitting_diff_roc:.4f} > {threshold_overfitting:.4f}. "
                "Возможно переобучение!"
            )
            mlflow.log_metric("overfitting_diff_roc", overfitting_diff_roc)
        else:
            logger.info(
                f"Разница между train и test ROC AUC: {overfitting_diff_roc:.4f} — переобучения не наблюдается"
            )

        # Проверка на переобучение (по PR-AUC)
        overfitting_diff_pr = train_pr_auc - test_pr_auc
        if overfitting_diff_pr > threshold_overfitting:
            logger.warning(
                f"Высокая разница между train и test PR-AUC: {overfitting_diff_pr:.4f} > {threshold_overfitting:.4f}. "
                "Возможно переобучение!"
            )
            mlflow.log_metric("overfitting_diff_pr", overfitting_diff_pr)

        # Сохранение важности фичей
        feature_importance = pd.DataFrame(
            {
                "feature": features_train_val.columns,
                "importance": best_xgb.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        feature_importance_path = Path("models/feature_importance.csv")
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importance.to_csv(feature_importance_path, index=False)
        logger.info(f"Важность фичей сохранена: {feature_importance_path}")

        # Логируем важность фичей в MLflow как artifact
        if log_model_to_mlflow:
            mlflow.log_artifact(str(feature_importance_path))
            logger.info("Важность фичей сохранена в MLflow как artifact")

        # Логируем модель в MLflow
        if log_model_to_mlflow:
            mlflow.xgboost.log_model(best_xgb, artifact_path="xgb_model")
            logger.info("Модель сохранена в MLflow")

        # Локальное сохранение модели в формате XGBoost
        if save_model_local:
            local_model_path = Path(local_model_path)
            local_model_path.parent.mkdir(parents=True, exist_ok=True)
            best_xgb.save_model(str(local_model_path))
            logger.info(f"Модель сохранена локально: {local_model_path}")

        logger.info("Обучение модели завершено успешно")

        return {
            "run_id": run_id,
            "roc_auc_train": train_roc_auc,
            "roc_auc_val": val_roc_auc,
            "roc_auc_test": test_roc_auc,
            "pr_auc_train": train_pr_auc,
            "pr_auc_val": val_pr_auc,
            "pr_auc_test": test_pr_auc,
            "overfitting_diff_roc": overfitting_diff_roc,
            "overfitting_diff_pr": overfitting_diff_pr,
            "model": best_xgb,
            "params": best_params,
            "feature_importance": feature_importance,
        }


def run_train_pipeline(
    best_params_path: Path = Path("configs/best_hyperparams.yaml"),
    mlflow_tracking_uri: str = None,
    mlflow_experiment_name: str = "xgb_model_training",
    log_model_to_mlflow: bool = True,
    save_model_local: bool = True,
    local_model_path: Path = Path("models/best_model.json"),
    threshold_overfitting: float = 0.1,
):
    """
    Оркестрационная функция для обучения модели.
    Вызывает train_model с параметрами по умолчанию.
    """
    logger.info("Начинается пайплайн обучения модели")

    result = train_model(
        best_params_path=best_params_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        log_model_to_mlflow=log_model_to_mlflow,
        save_model_local=save_model_local,
        local_model_path=local_model_path,
        threshold_overfitting=threshold_overfitting,
    )

    logger.info(
        f"Пайплайн обучения модели завершён успешно. "
        f"ROC AUC: train={result['roc_auc_train']:.4f}, "
        f"val={result['roc_auc_val']:.4f}, "
        f"test={result['roc_auc_test']:.4f}, "
        f"overfitting_diff_roc={result['overfitting_diff_roc']:.4f}; "
        f"PR-AUC: train={result['pr_auc_train']:.4f}, "
        f"val={result['pr_auc_val']:.4f}, "
        f"test={result['pr_auc_test']:.4f}, "
        f"overfitting_diff_pr={result['overfitting_diff_pr']:.4f}"
    )
    return result
