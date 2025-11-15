import pandas as pd
import typer
from pathlib import Path
import yaml


def load_dataset(directory: Path, filename: str):
    path = directory / filename
    try:
        df = pd.read_csv(path)
        typer.secho(f"Данные успешно загружены из {path}", fg=typer.colors.GREEN)
        return df
    except Exception as e:
        typer.secho(f"Ошибка при загрузке данных из {path}: {e}", fg=typer.colors.RED)
        return None


def save_dataset(df, directory: Path, filename: str):
    path = directory / filename
    try:
        df.to_csv(path, index=False)
        typer.secho(f"Данные успешно сохранены в {path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Ошибка при сохранении данных в {path}: {e}", fg=typer.colors.RED)


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
