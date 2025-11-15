from pathlib import Path


import mlflow
from mlflow import log_metric, log_param
from mlflow.tracking import MlflowClient
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
import typer
from xgboost import XGBClassifier
import yaml

from binary_classification_bank.config import MODELS_DIR, PROCESSED_DATA_DIR
from binary_classification_bank.data_io import load_config, load_dataset, save_dataset

app = typer.Typer()


@app.command()
def optimize(
    input_path: str = "configs/hyperparam_ranges.yaml",
    output_path: str = "configs/best_hyperparams.yaml",
):
    try:
        processed_data_dir = PROCESSED_DATA_DIR
        typer.secho(f"Загрузка конфигурации диапазонов из {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            param_ranges = yaml.safe_load(f)

        features_train_name = "features_base_train_processed.csv"
        target_train_name = "target_train.csv"
        features_val_name = "features_base_val_processed.csv"
        target_val_name = "target_val.csv"

        typer.secho("Загрузка обучающих данных из CSV")
        features_train = load_dataset(processed_data_dir, features_train_name)
        target_train = load_dataset(processed_data_dir, target_train_name)

        typer.secho("Загрузка валидационных данных из CSV ")
        features_val = load_dataset(processed_data_dir, features_val_name)
        target_val = load_dataset(processed_data_dir, target_val_name)

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="roc_auc"
        )

        mlflow.set_experiment("optuna_hyperparameter_optimization")

        def objective(trial):
            class_ratios = target_train.value_counts(normalize=True)
            scale_pos_weight = class_ratios[0] / class_ratios[1]

            params = {}
            for param_name, bounds in param_ranges.items():
                if bounds["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, bounds["low"], bounds["high"], step=bounds.get("step", 1)
                    )
                elif bounds["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, bounds["low"], bounds["high"], log=bounds.get("log", False)
                    )
                elif bounds["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, bounds["choices"])
            params["scale_pos_weight"] = scale_pos_weight

            with mlflow.start_run():
                mlflow.log_params(params)
                model = XGBClassifier(**params)
                model.fit(features_train, target_train)

                y_pred_proba = model.predict_proba(features_val)[:, 1]
                roc_auc = roc_auc_score(target_val, y_pred_proba)

                mlflow.log_metric("roc_auc", roc_auc)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return roc_auc

        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        typer.secho("Запуск оптимизации гиперпараметров...")
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, storage=None, load_if_exists=True
        )
        study.optimize(
            objective, n_trials=50, timeout=None, n_jobs=-1, callbacks=[mlflow_callback]
        )

        best_params = study.best_params
        typer.secho("Оптимизация завершена. Лучшие параметры:", fg=typer.colors.GREEN)
        for key, value in best_params.items():
            typer.secho(f"  {key}: {value}", fg=typer.colors.GREEN)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_params, f)
        typer.secho(f"Сохранение найденных параметров в {output_path}", fg=typer.colors.CYAN)

    except Exception as e:
        typer.secho(f"Ошибка при оптимизации или работе с файлами: {e}", fg=typer.colors.RED)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    optimize()


if __name__ == "__main__":
    app()
