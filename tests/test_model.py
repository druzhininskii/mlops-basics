import pytest
import torch
from torch.nn import functional as F

from src.model import ConvBlock, ProteinClassifier


@pytest.mark.parametrize(
    "batch_size, sequence_length",
    [(1, 1), (2, 5), (3, 8)],
)
def test_block_and_model_output_shapes(batch_size, sequence_length):
    tokens = torch.randint(0, 20, (batch_size, sequence_length))
    features = F.one_hot(tokens, num_classes=20).float()
    lengths = torch.full(
        (batch_size,), sequence_length, dtype=torch.long
    )

    block = ConvBlock(hidden_size=16)
    model = ProteinClassifier(num_classes=3, hidden_size=16)

    assert block(features).shape == (
        batch_size, sequence_length, 16
    )
    assert model(features, lengths).shape == (batch_size, 3)


def test_model_supports_backward():
    torch.manual_seed(0)

    model = ProteinClassifier(num_classes=3)

    tokens = torch.randint(0, 20, (2, 5))
    features = F.one_hot(tokens, num_classes=20).float()
    lengths = torch.tensor([5, 3])
    features[1, 3:] = 0.0

    targets = torch.tensor([0, 2])

    logits = model(features, lengths)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    assert torch.isfinite(loss).item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all().item(), name