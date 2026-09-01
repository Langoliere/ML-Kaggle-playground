import pytest
from dataset import handle_outliers_log, optimize_numeric_types, split_train_val_test


# --------------- Тесты для обработки выбросов -----------------
def test_handle_outliers_log_removes_outliers():
    """Проверка, что выбросы действительно обрабатываются."""
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})  # 1000 — выброс
    result = handle_outliers_log(data)
    
    # Выброс должен быть обработан (логарифмирован или удалён)
    assert result["a"].max() < 1000

def test_handle_outliers_log_no_outliers():
    "Проверка что не удалятся корректные данные"
    data = pd.DataFrame({"a":[1,2,3,4,5,6]})
    result = handle_outliers_log(data)
    assert result["a"].tolist() == data["a"].tolist()
    
def test_handle_outliers_log_row_count():
    """Проверка, что количество строк не меняется."""
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    result = handle_outliers_log(data)
    
    # Количество строк должно сохраниться
    assert len(result) == len(data)

def test_handle_outliers_log_negative_values():
    """Проверка обработки отрицательных значений (для log)."""
    data = pd.DataFrame({"a": [-5, -2, 1, 2, 3]})
    
    # Логарифм от отрицательных чисел не определён
    # Тест должен проверить, что функция не падает с ошибкой
    try:
        result = handle_outliers_log(data)
        assert result is not None
    except ValueError:
        # Или функция должна выбрасывать понятную ошибку
        pass

def test_handle_outliers_log_empty_dataframe():
    """Проверка обработки пустого DataFrame."""
    data = pd.DataFrame({"a": []})
    result = handle_outliers_log(data)
    
    assert len(result) == 0

def test_handle_outliers_log_nan_values():
    """Проверка обработки NaN значений."""
    data = pd.DataFrame({"a": [1, 2, None, 4, 5, 1000]})
    result = handle_outliers_log(data)
    
    # NaN должны остаться NaN
    assert result["a"].isna().sum() == 1


# ----------------- Тесты для оптимизации типов данных -------------------------------

def test_optimize_numeric_types_shape():
    """Проверка, что размерность не меняется."""
    data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        "c": [10, 20, 30, 40, 50],
    })
    result = optimize_numeric_types(data, margin=0.9)
    
    assert result.shape == data.shape
    assert len(result.columns) == len(data.columns)


def test_optimize_numeric_types_memory_reduction():
    """Проверка, что память действительно уменьшилась."""
    data = pd.DataFrame({
        "a": np.array([1, 2, 3, 4, 5] * 1000, dtype="int64"),
        "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 1000, dtype="float64"),
    })
    
    before_memory = data.memory_usage(deep=True).sum()
    result = optimize_numeric_types(data, margin=0.9)
    after_memory = result.memory_usage(deep=True).sum()
    
    assert after_memory < before_memory
    assert (before_memory - after_memory) / before_memory > 0.01  # >1% уменьшение


def test_optimize_numeric_types_dtypes_changed():
    """Проверка, что типы данных изменились."""
    data = pd.DataFrame({
        "a": np.array([1, 2, 3, 4, 5], dtype="int64"),
        "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64"),
    })
    result = optimize_numeric_types(data, margin=0.9)
    
    # Типы должны измениться на более компактные
    assert result["a"].dtype != "int64"  # стал int8/int16
    assert result["b"].dtype != "float64"  # стал float32


def test_optimize_numeric_types_values_unchanged():
    """Проверка, что значения не изменились."""
    data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = optimize_numeric_types(data, margin=0.9)
    
    assert result["a"].tolist() == data["a"].tolist()
    assert result["b"].tolist() == data["b"].tolist()


def test_optimize_numeric_types_empty_dataframe():
    """Проверка пустого DataFrame."""
    data = pd.DataFrame({"a": [], "b": []})
    result = optimize_numeric_types(data, margin=0.9)
    
    assert len(result) == 0


# ---------------- Тесты для разделения выборки на обучающую, валидационную и тестовую ----------------------------

def test_split_train_val_test_ratio():
    """Проверка соотношения разделений."""
    data = pd.DataFrame({
        "a": range(100),
        "y": [0] * 50 + [1] * 50,
    })
    train, val, test = split_train_val_test(
        data, target_col="y", test_size=0.4, val_size=0.5, random_state=42
    )
    
    # test = 40%, val = 30%, train = 30%
    assert len(test) == 40  # 40%
    assert len(val) == 30   # 30% (50% от 60%)
    assert len(train) == 30  # 30%


def test_split_train_val_test_all_rows_used():
    """Проверка, что все строки использованы."""
    data = pd.DataFrame({
        "a": range(100),
        "y": range(100),
    })
    train, val, test = split_train_val_test(
        data, target_col="y", test_size=0.3, val_size=0.3, random_state=42
    )
    
    assert len(train) + len(val) + len(test) == len(data)


def test_split_train_val_test_features_target_match():
    """Проверка, что фичи и таргет совпадают по длине."""
    data = pd.DataFrame({
        "a": range(100),
        "b": range(100),
        "y": range(100),
    })
    features_train, target_train, features_val, target_val, features_test, target_test = split_train_val_test(
        data, target_col="y", test_size=0.3, val_size=0.3, random_state=42
    )
    
    assert len(features_train) == len(target_train)
    assert len(features_val) == len(target_val)
    assert len(features_test) == len(target_test)


def test_split_train_val_test_no_overlap():
    """Проверка, что нет пересечений между split'ами."""
    data = pd.DataFrame({
        "a": range(100),
        "y": range(100),
    })
    train, val, test = split_train_val_test(
        data, target_col="y", test_size=0.3, val_size=0.3, random_state=42
    )
    
    train_ids = set(train.index)
    val_ids = set(val.index)
    test_ids = set(test.index)
    
    assert len(train_ids & val_ids) == 0  # нет пересечений
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_split_train_val_test_stratification():
    """Проверка стратификации (сохранение соотношения классов)."""
    data = pd.DataFrame({
        "a": range(100),
        "y": [0] * 70 + [1] * 30,  # 70% класса 0, 30% класса 1
    })
    train, val, test = split_train_val_test(
        data, target_col="y", test_size=0.3, val_size=0.3, random_state=42, stratify=True
    )
    
    # Соотношение классов должно сохраниться примерно
    original_ratio = data["y"].value_counts(normalize=True)[0]
    train_ratio = train["y"].value_counts(normalize=True)[0]
    
    assert abs(original_ratio - train_ratio) < 0.1  # разница <10%


def test_split_train_val_test_small_dataset():
    """Проверка для малого датасета."""
    data = pd.DataFrame({
        "a": range(10),
        "y": range(10),
    })
    train, val, test = split_train_val_test(
        data, target_col="y", test_size=0.3, val_size=0.3, random_state=42
    )
    
    # Должен работать даже на малом датасете
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
