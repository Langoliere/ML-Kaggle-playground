from pathlib import Path
import typer
import pandas as pd
import mlflow
import mlflow.pyfunc

from binary_classification_bank.config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR
from binary_classification_bank.data_io import load_config, load_dataset, save_dataset

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = Path("E:/Pet Projects/Kaggle Playground/binary_classification_bank/models/mlflow_artifacts"),
    predictions_path: Path = DATA_DIR,
):
    mlflow.set_tracking_uri(model_path.as_uri())
    typer.secho(f"Загрузка модели из {model_path.as_uri()}...")
    model = mlflow.xgboost.load_model(str(model_path))

    features = load_dataset(features_path, "features_base_test_processed.csv")
    try:
        typer.secho("Выполнение предсказаний для каждого наблюдения...")
        predictions_proba = model.predict_proba(features)[:, 1]

        pred_df = pd.DataFrame({"proba": predictions_proba})
        save_dataset(pred_df, predictions_path, "predictions/test_predictions.csv")
    except Exception as e:
        typer.secho(f"Ошибка работы модели: {e}", fg=typer.colors.RED)
    else:
        typer.secho(f"Предсказания сохранены в {predictions_path}", fg=typer.colors.GREEN)
        typer.secho("Инференс завершен.", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
