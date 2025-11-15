#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = binary_classification_bank
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
CONDA_PREFIX = E:/Conda

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

.PHONY: activate
activate:
	conda activate binary_classification_bank
	
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

################################################################	
.PHONY: plots
plots:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/plots.py"

################################################################	

## Data preprocessing
.PHONY: handle_outliers_log
handle_outliers_log:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/dataset.py handle-outliers-log"


## Data preprocessing
.PHONY: optimize_numeric_types
optimize_numeric_types:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/dataset.py optimize-numeric-types" 

## Data preprocessing
.PHONY: split_features_target
split_features_target: 
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/dataset.py split-features-target"

## Data preprocessing
.PHONY: split_train_val_test
split_train_val_test: 
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/dataset.py split-train-val-test"


## Make dataset
.PHONY: dataset
dataset: 
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/dataset.py main"

################################################################

.PHONY: features
features:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/features.py"

################################################################
.PHONY: hyperparam_opt
hyperparam_opt:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/modeling/hyperparam.py optimize"

################################################################
.PHONY: train_model
train_model:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/modeling/train.py train-model"

################################################################

################################################################
.PHONY: predict
predict:
	cmd /c "call E:\\Conda\\Scripts\\activate.bat binary_classification_bank && python binary_classification_bank/modeling/predict.py"



#################################################################################
.PHONY: ruff_fix
ruff_fix:
	cmd /c "call E:\Conda\Scripts\activate.bat binary_classification_bank && ruff format ."

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
