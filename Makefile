#################################################################################
# GLOBALS                                                                        #
#################################################################################

PROJECT_NAME = binary_classification_bank
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
CONDA_PREFIX = E:/Conda


#################################################################################
# ENVIRONMENT MANAGEMENT                                                         #
#################################################################################

## Создать conda окружение из environment.yml (однократная установка)
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> Conda environment created. Activate with:\nconda activate $(PROJECT_NAME)"


## Обновить зависимости в существующем окружении (пакеты, версии, удаление лишних)
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune


## Активировать окружение (Windows PowerShell / cmd)
.PHONY: activate
activate:
	@echo "Use the following command to activate the conda environment:"
	@echo "conda activate $(PROJECT_NAME)"


#################################################################################
# CLEANING AND FORMATTING                                                        #
#################################################################################

## Удалить все скомпилированные Python файлы (.pyc, .pyo) и кеши
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Проверить стиль кода с ruff (без изменений)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Исправить стиль кода автоматически с помощью ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Отдельная команда только на исправление ruff без проверки
.PHONY: ruff_fix
ruff_fix:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && ruff format ."


#################################################################################
# TESTING                                                                       #
#################################################################################

## Запустить модульные тесты с pytest
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


#################################################################################
# DATA PREPROCESSING                                                             #
#################################################################################

## Обработка выбросов с логарифмированием 
.PHONY: handle_outliers_log
handle_outliers_log:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/dataset.py handle-outliers-log"


## Оптимизация числовых типов для экономии памяти
.PHONY: optimize_numeric_types
optimize_numeric_types:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/dataset.py optimize-numeric-types"


## Разделение на признаки и таргет
.PHONY: split_features_target
split_features_target:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/dataset.py split-features-target"


## Разделение на обучающую, валидационную и тестовую выборки
.PHONY: split_train_val_test
split_train_val_test:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/dataset.py split-train-val-test"


## Полный pipeline подготовки датасета 
.PHONY: dataset
dataset:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/dataset.py main"


#################################################################################
# FEATURE ENGINEERING                                                            #
#################################################################################

## Генерация и обработка признаков
.PHONY: features
features:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/features.py"


#################################################################################
# MODEL TRAINING AND OPTIMIZATION                                               #
#################################################################################

## Оптимизация гиперпараметров с Optuna
.PHONY: hyperparam_opt
hyperparam_opt:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/modeling/hyperparam.py optimize"


## Обучение модели XGBoost с выбранными параметрами
.PHONY: train_model
train_model:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/modeling/train.py train-model"


#################################################################################
# PREDICTION AND EVALUATION                                                     #
#################################################################################

## Запуск предсказаний на новых данных
.PHONY: predict
predict:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/modeling/predict.py"


#################################################################################
# VISUALIZATION                                                                 #
#################################################################################

## Построение графиков и визуализация результатов обучения и анализа
.PHONY: plots
plots:
	cmd /c "call $(CONDA_PREFIX)\\Scripts\\activate.bat $(PROJECT_NAME) && python binary_classification_bank/plots.py"


#################################################################################
# HELP                                                                          #
#################################################################################

.DEFAULT_GOAL := help


## Автоматический вывод списка доступных команд (парсит комментарии)
define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT


.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
