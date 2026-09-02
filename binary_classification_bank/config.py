from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# Загружаем переменные окружения из .env файла, если он существует
load_dotenv()

# === Пути к папкам проекта ===

# Корневая папка проекта (на один уровень выше config.py)
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Папка для конфигов (YAML файлы)
CONFIG_DIR = PROJ_ROOT / "configs"

# Папка для данных
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # исходные, неизменные данные
INTERIM_DATA_DIR = (
    DATA_DIR / "interim"
)  # промежуточные данные (после предварительной обработки)
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # финальные датасеты для обучения/инференса
EXTERNAL_DATA_DIR = (
    DATA_DIR / "external"
)  # внешние данные (например, из API, открытых источников)

# Папка для сохранённых моделей
MODELS_DIR = PROJ_ROOT / "models"

# Папка для отчётов и визуализаций
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Создаём все необходимые папки, если их нет
for p in [
    CONFIG_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# === Загрузка YAML конфигураций ===


def load_yaml(path: Path) -> dict:
    """
    Загружает YAML файл и возвращает его содержимое как словарь.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Загружаем основные конфигурационные файлы
try:
    DATA_CONFIG = load_yaml(CONFIG_DIR / "data_config.yaml")
except Exception as e:
    logger.error(f"Ошибка загрузки data_config.yaml: {e}")
    DATA_CONFIG = {}

try:
    BEST_HYPERPARAMS = load_yaml(CONFIG_DIR / "best_hyperparams.yaml")
except Exception as e:
    logger.error(f"Ошибка загрузки best_hyperparams.yaml: {e}")
    BEST_HYPERPARAMS = {}

try:
    HYPERPARAM_RANGES = load_yaml(CONFIG_DIR / "hyperparam_ranges.yaml")
except Exception as e:
    logger.error(f"Ошибка загрузки hyperparam_ranges.yaml: {e}")
    HYPERPARAM_RANGES = {}

try:
    DEPENDENCIES = load_yaml(CONFIG_DIR / "dependencies.yaml")
except Exception as e:
    logger.error(f"Ошибка загрузки dependencies.yaml: {e}")
    DEPENDENCIES = {}
# === Настройка логирования ===

# Убираем дефолтный хендлер loguru, чтобы задать свою конфигурацию
logger.remove()

# Логирование в файл
logger.add(
    "logs/app.log",
    rotation="500 MB",  # новый файл каждый 500 МБ
    retention="10 days",  # хранить логи не более 10 дней
    level="INFO",  # уровень логирования
    colorize=False,  # в файле не используем цвета
)

# Логирование в консоль с tqdm
try:
    from tqdm import tqdm

    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level="INFO",
    )
except ModuleNotFoundError:
    # Если tqdm не установлен, просто логируем в консоль без tqdm
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")
