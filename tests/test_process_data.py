import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.process_data import (
    filter_noncanonical,
    length_distribution,
    length_filter,
    process_sequences,
)


@pytest.mark.parametrize(
    "sequence, expected_kept",
    [
        ("ACD", True),
        ("ACDE", True),
        ("ACDEF", False),
    ],
)
def test_length_filter(sequence, expected_kept):
    data = pd.DataFrame({"sequence": [sequence]})
    clean, removed = length_filter(data, "sequence", 4)
    expected_sequences = [sequence] if expected_kept else []

    assert clean["sequence"].tolist() == expected_sequences
    assert removed == (0 if expected_kept else 1)


@pytest.mark.parametrize(
    "sequence, expected_kept",
    [
        ("ACDEFGHIKLMNPQRSTVWY", True),
        ("AAAA", True),
        ("AX", False),
        ("ac", False),
        ("A C", False),
        ("A1", False),
        ("", False),
        (None, False),
    ],
)
def test_filter_noncanonical(sequence, expected_kept):
    data = pd.DataFrame({"sequence": [sequence]})
    clean, removed = filter_noncanonical(data, "sequence")
    expected_sequences = [sequence] if expected_kept else []

    assert clean["sequence"].tolist() == expected_sequences
    assert removed == (0 if expected_kept else 1)


def test_length_distribution():
    data = pd.DataFrame({"sequence": ["AC", "DE", "AAA", "ACDE"]})
    result = length_distribution(data, "sequence")

    assert dict(result) == {2: 2, 3: 1, 4: 1}


@pytest.fixture
def dirty_sequences():
    return pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "sequence": [
                "AC",
                "ACD",
                "AAAAA",
                "AX",
                "",
                None,
            ],
        }
    )


def test_process_sequences(dirty_sequences):
    clean, distribution, report = process_sequences(
        dirty_sequences, sequence_column="sequence", max_len=4
    )
    expected = pd.DataFrame({"protein_id": ["p1", "p2"], "sequence": ["AC", "ACD"]})
    pd.testing.assert_frame_equal(clean, expected)
    assert dict(distribution) == {2: 1, 3: 1}

    assert report == {
        "input_rows": 6,
        "removed_empty": 2,
        "removed_noncanonical": 1,
        "removed_length": 1,
        "output_rows": 2,
        "removed_total": 4,
    }


@given(
    sequences=st.lists(
        st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", max_size=30),
        max_size=30,
    ),
    max_len=st.integers(min_value=0, max_value=30),
)
def test_length_filter_hypotesis(sequences, max_len):
    expected = [sequence for sequence in sequences if len(sequence) <= max_len]
    data = pd.DataFrame({"sequence": sequences})
    clean, removed = length_filter(data, "sequence", max_len)

    assert clean["sequence"].tolist() == expected
    assert len(clean) + removed == len(data)
    assert all(len(sequence) <= max_len for sequence in clean["sequence"])


@given(
    sequences=st.lists(
        st.one_of(
            st.none(),
            st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", max_size=30),
            st.text(alphabet="AXac 1", max_size=30),
        ),
        max_size=30,
    ),
    max_len=st.integers(min_value=0, max_value=30),
)
def test_process_sequences_hypotesis(sequences, max_len):
    data = pd.DataFrame({"sequence": sequences})
    clean, distr, report = process_sequences(data, "sequence", max_len)
    clean_two, distr_two, report_two = process_sequences(clean, "sequence", max_len)

    pd.testing.assert_frame_equal(clean, clean_two)
    assert dict(distr) == dict(distr_two)

    assert report_two["removed_empty"] == 0
    assert report_two["removed_noncanonical"] == 0
    assert report_two["removed_length"] == 0
    assert report_two["removed_total"] == 0

    assert report["removed_total"] + len(clean) == len(data)

