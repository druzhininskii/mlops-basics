from src.functions import power


def test_power() -> None:
    assert power(2, 3) == 8


def test_power() -> None:
    assert power(2, -1) == 0.5
