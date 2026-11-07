import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import ProteinDataset, collate_proteins, one_hot_encode


def test_one_hot_encode():
    sequence_1 = "AFC"
    sequence_2 = "AAA"
    sequence_3 = "DDD"

    emb1 = one_hot_encode(sequence_1)
    emb2 = one_hot_encode(sequence_2)
    emb3 = one_hot_encode(sequence_3)

    assert emb1 != emb2
    assert emb1 != emb3
    assert emb2 != emb3


def test_one_hot_encode_uses_correct_positions():
    result = one_hot_encode("AC")

    assert result == [
        [1.0, 0.0] + [0.0] * 18,
        [0.0, 1.0] + [0.0] * 18,
    ]


def test_one_hot_encode_empty_sequence():
    assert one_hot_encode("") == []


@pytest.mark.parametrize("sequence", ["AX", "ac", "A C", "AB"])
def test_one_hot_encode_rejects_invalid_symbols(sequence):
    with pytest.raises(ValueError):
        one_hot_encode(sequence)


def test_dataset_returns_encoded_sequence_and_label():
    dataset = ProteinDataset(["AC", "DDD"], [0, 1])

    features, label = dataset[0]

    expected = torch.tensor([
        [1.0, 0.0] + [0.0] * 18,
        [0.0, 1.0] + [0.0] * 18,
    ])

    assert len(dataset) == 2
    torch.testing.assert_close(features, expected)
    assert label.item() == 0
    assert label.dtype == torch.long


def test_dataloader():
    dataset = ProteinDataset(["AC", "DDD", "A"], [0, 1, 0])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_proteins
    )

    batches = list(loader)
    assert len(batches) == 2

    features, labels, lengths = batches[0]

    assert features.shape == (2, 3, 20)
    assert labels.tolist() == [0, 1]
    assert lengths.tolist() == [2, 3]
    assert torch.equal(features[0, 2], torch.zeros(20))

    last_features, last_labels, last_lengths = batches[1]

    assert last_features.shape == (1, 1, 20)
    assert last_labels.tolist() == [0]
    assert last_lengths.tolist() == [1]   