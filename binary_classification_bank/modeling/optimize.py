from pathlib import Path

import mlflow
import optuna
import yaml
from config import PROCESSED_DATA_DIR
from data_io import load_dataset
from loguru import logger
from optuna.exceptions import TrialPruned
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def optimize_hyperparameters(
    input_path: Path,
    output_path: Path,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    features_train_name: str = "features_base_train_processed.csv",
    target_train_name: str = "target_train.csv",
    features_val_name: str = "features_base_val_processed.csv",
    target_val_name: str = "target_val.csv",
    n_trials: int = 50,
    random_seed: int = 42,
):
    """
    Оптимизация гиперпараметров XGBClassifier с помощью Optuna + MLflow.

    Параметры:
      - input_path: путь к hyperparam_ranges.yaml
      - output_path: путь, куда сохранить best_hyperparams.yaml
      - processed_data_dir: папка с обработанными данными
      - n_trials: число trials в Optuna
      - random_seed: случайное число

    Возвращает:
      - dict с лучшими параметрами
    """
    logger.info(f"Загрузка конфигурации диапазонов из {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            param_ranges = yaml.safe_load(f)

        logger.info("Загрузка обучающих данных из CSV")
        features_train = load_dataset(processed_data_dir, features_train_name)
        target_train = load_dataset(processed_data_dir, target_train_name)

        logger.info("Загрузка валидационных данных из CSV")
        features_val = load_dataset(processed_data_dir, features_val_name)
        target_val = load_dataset(processed_data_dir, target_val_name)

        if features_train is None or target_train is None:
            raise RuntimeError("Не удалось загрузить обучающие данные")
        if features_val is None or target_val is None:
            raise RuntimeError("Не удалось загрузить валидационные данные")

        # Настройка MLflow
        mlflow.set_experiment("optuna_hyperparameter_optimization")

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="roc_auc",
            # Опционально: можно добавить pruned_metric_name, best_metric_name
        )

        def objective(trial):
            # Вычисляем scale_pos_weight для несбалансированных классов
            class_ratios = target_train.value_counts(normalize=True)
            scale_pos_weight = class_ratios[0] / class_ratios[1]

            params = {}
            for param_name, bounds in param_ranges.items():
                if bounds["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        bounds["low"],
                        bounds["high"],
                        step=bounds.get("step", 1),
                    )
                elif bounds["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        bounds["low"],
                        bounds["high"],
                        log=bounds.get("log", False),
                    )
                elif bounds["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, bounds["choices"]
                    )

            params["scale_pos_weight"] = scale_pos_weight

            model = XGBClassifier(**params, random_state=random_seed)
            model.fit(features_train, target_train)

            y_pred_proba = model.predict_proba(features_val)[:, 1]
            roc_auc = roc_auc_score(target_val, y_pred_proba)

            if trial.should_prune():
                raise TrialPruned()

            return roc_auc

        sampler = TPESampler(seed=random_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        logger.info("Запуск оптимизации гиперпараметров...")
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=None,
            load_if_exists=True,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None,
            n_jobs=-1,
            callbacks=[mlflow_callback],
        )

        best_params = study.best_params
        logger.info("Оптимизация завершена. Лучшие параметры:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_params, f)
        logger.info(f"Сохранение найденных параметров в {output_path}")

        return best_params

    except Exception as e:
        logger.error(f"Ошибка при оптимизации или работе с файлах: {e}")
        raise


def run_optimize_pipeline(
    input_path: Path = Path("configs/hyperparam_ranges.yaml"),
    output_path: Path = Path("configs/best_hyperparams.yaml"),
    n_trials: int = 50,
    random_seed: int = 42,
):
    """
    Оркестрационная функция для оптимизации гиперпараметров.
    Вызывает optimize_hyperparameters с параметрами по умолчанию.
    """
    logger.info("Начинается пайплайн оптимизации гиперпараметров")
    best_params = optimize_hyperparameters(
        input_path=input_path,
        output_path=output_path,
        n_trials=n_trials,
        random_seed=random_seed,
    )
    logger.info("Пайплайн оптимизации гиперпараметров завершён успешно")
    return best_params
