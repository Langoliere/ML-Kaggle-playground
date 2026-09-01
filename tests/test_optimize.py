import yaml, pytest

def test_hyperparam_ranges_yaml_is_valid():
    """Проверка, что файл с диапазонами корректный."""
    with open("configs/hyperparam_ranges.yaml") as f:
        param_ranges = yaml.safe_load(f)
    
    assert "max_depth" in param_ranges
    assert param_ranges["max_depth"]["type"] == "int"
    assert param_ranges["max_depth"]["low"] < param_ranges["max_depth"]["high"]
