from src.functions import add, multiplication, power


def test_add() -> None:
    assert add(3, 2) == 5

def test_multiplication() -> None:
    assert multiplication(2, 2) == 4


def test_power() -> None:
    assert power(2, 3) == 8


def test_power_negative() -> None:
    assert power(2, -1) == 0.5
