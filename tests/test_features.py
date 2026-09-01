import pytest
from features import feature_constuction, add_numeric_combinations, add_categorical_combinations,preprocess_and_pca 

#------------------ Тесты для создания ручных фичей ---------------------------

def test_feature_construction_dimension_increased():
    """Проверка увеличения количества фичей."""
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "income": [30000, 50000, 70000],
        "age": [25, 35, 45],
    })
    original_cols = len(data.columns)
    result = feature_construction(data)
    new_cols = len(result.columns)
    
    assert new_cols > original_cols

    

#------------------ Тесты для создания численных фичей перебором --------------
def test_add_numeric_combinations_feature_count():
    """Проверка увеличения количества фичей."""
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "feature3": [7, 8, 9],
    })
    original_cols = len(data.columns)
    result = add_numeric_combinations(data)
    new_cols = len(result.columns)
    
    assert new_cols > original_cols


def test_add_numeric_combinations_all_combinations_created():
    """Проверка создания всех комбинаций."""
    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    result = add_numeric_combinations(data)
    
    # Проверка созданных комбинаций
    assert "a_b" in result.columns



def test_add_numeric_combinations_values_correct():
    """Проверка корректности значений комбинаций."""
    data = pd.DataFrame({
        "a": [2, 4, 6],
        "b": [3, 5, 7],
    })
    result = add_numeric_combinations(data)
    
    # product = a * b
    expected_product = [2*3, 4*5, 6*7]
    assert result["a_b"].tolist() == expected_product


def test_add_numeric_combinations_no_new_nan():
    """Проверка, что не создаются новые NaN."""
    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    result = add_numeric_combinations(data)
    
    # В новых фичах не должно быть NaN
    for col in result.columns:
        if col not in data.columns:
            assert result[col].isna().sum() == 0


#------------------ Тесты для создания категориальных фичей перебором --------

def test_add_categorical_combinations_feature_count():
    """Проверка увеличения количества фичей."""
    data = pd.DataFrame({
        "category1": ["A", "B", "A", "C"],
        "category2": ["X", "Y", "X", "Z"],
    })
    original_cols = len(data.columns)
    result = add_categorical_combinations(data)
    new_cols = len(result.columns)
    
    assert new_cols > original_cols


def test_add_categorical_combinations_combinations_created():
    """Проверка создания комбинаций."""
    data = pd.DataFrame({
        "category1": ["A", "B", "A"],
        "category2": ["X", "Y", "X"],
    })
    result = add_categorical_combinations(data)
    
    # Проверка созданных комбинаций
    assert "category1_category2" in result.columns


def test_add_categorical_combinations_values_valid():
    """Проверка корректности значений комбинаций."""
    data = pd.DataFrame({
        "category1": ["A", "B"],
        "category2": ["X", "Y"],
    })
    result = add_categorical_combinations(data)
    
    # Значения должны быть в формате "cat1_cat2"
    assert result["category1_category2"].iloc[0] == "A_X"
    assert result["category1_category2"].iloc[1] == "B_Y"
    

#------------------ Тесты для пайплайна предобработки и PCA -----------------

def test_preprocess_and_pca_categorical_encoded():
    """Проверка кодирования категориальных признаков."""
    features_train = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["A", "B", "A", "C", "B","A", "B", "A", "C", "B"],
    })
    features_val = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["A", "B", "A", "C", "B","A", "B", "A", "C", "B"],
    })
    features_test = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["A", "B", "A", "C", "B","A", "B", "A", "C", "B"],
    })
    
    result_train, result_val, result_test = preprocess_and_pca(features_train, features_val, features_test, apply_pca=False)
    
    # После OneHotEncoder категориальные должны стать числовыми
    assert result_train["category_A"].iloc[0] in [0, 1]
    assert result_train["category_B"].iloc[0] in [0, 1]
    assert result_train["category_C"].iloc[0] in [0, 1]
    
    assert result_val["category_A"].iloc[0] in [0, 1]
    assert result_val["category_B"].iloc[0] in [0, 1]
    assert result_val["category_C"].iloc[0] in [0, 1]
    
    assert result_test["category_A"].iloc[0] in [0, 1]
    assert result_test["category_B"].iloc[0] in [0, 1]
    assert result_test["category_C"].iloc[0] in [0, 1]


def test_preprocess_and_pca_numeric_scaled():
    """Проверка масштабирования числовых признаков."""
    features_train = pd.DataFrame({
        "numeric": [1, 100, 200, 300, 1000],
    })
     features_val = pd.DataFrame({
        "numeric": [1, 100, 200, 300, 1000],
    })
    features_test = pd.DataFrame({
        "numeric": [1, 100, 200, 300, 1000],
    })
    result_train, result_val, result_test = preprocess_and_pca(features_train, features_val, features_test, apply_pca=False)
    
    # После RobustScaler значения должны быть масштабированы
    assert result_train["numeric"].median() == 0
    assert result_val["numeric"].median() == 0
    assert result_test["numeric"].median() == 0


def test_preprocess_and_pca_dimensionality_reduced():
    """Проверка уменьшения размерности с PCA."""
     features_train = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "feature3": range(200, 300),
        "feature4": range(300, 400),
    })
   features_val = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "feature3": range(200, 300),
        "feature4": range(300, 400),
    })
   features_test = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "feature3": range(200, 300),
        "feature4": range(300, 400),
    })

    result_train_no_pca, result_val_no_pca, result_test_no_pca = preprocess_and_pca(features_train, features_val, features_test, apply_pca=False)
    result_train_pca, result_val_pca, result_test_pca = preprocess_and_pca(features_train, features_val, features_test, apply_pca=True))
    
    # PCA должен уменьшить количество фичей
    assert len(result_train_pca.columns) < len(result_train_no_pca.columns)
    assert len(result_val_pca.columns) < len(result_val_no_pca.columns)
    assert len(result_test_pca.columns) < len(result_test_no_pca.columns)


def test_preprocess_and_pca_no_nan():
    """Проверка отсутствия NaN в результате."""
    features_train = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
    })
    features_val = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
    })
    features_test = pd.DataFrame({
        "numeric": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
    })
    
    result_train, result_val, result_test = preprocess_and_pca(features_train, features_val, features_test, apply_pca=False)

    assert result_train.isna().sum().sum() == 0
    assert result_val.isna().sum().sum() == 0
    assert result_test.isna().sum().sum() == 0


def test_preprocess_and_pca_all_rows_preserved():
    """Проверка сохранения всех строк."""
    features_train = pd.DataFrame({
        "numeric": range(100),
        "category": ["A"] * 50 + ["B"] * 50,
    })
    features_val = pd.DataFrame({
        "numeric": range(100),
        "category": ["A"] * 50 + ["B"] * 50,
    })
    features_test = pd.DataFrame({
        "numeric": range(100),
        "category": ["A"] * 50 + ["B"] * 50,
    })
    result_train, result_val, result_test = preprocess_and_pca(features_train, features_val, features_test, apply_pca=False)
    
    assert len(result_train) == len(features_train)
    assert len(result_val) == len(features_val)
    assert len(result_test) == len(features_test)
