import math

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import ProteinDataset, collate_proteins
from src.model import ProteinClassifier
from src.process_data import process_sequences
from src.training import evaluate, train_epoch, train_step


def test_train_step_updates_weights():
    torch.manual_seed(0)

    model = ProteinClassifier(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    batch = collate_proteins([
        ProteinDataset(["ACD"], [0])[0],
    ])

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss = train_step(model, optimizer, batch)

    assert math.isfinite(loss)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    assert all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )

@pytest.mark.integration
def test_training_pipeline():
    torch.manual_seed(0)

    data = pd.DataFrame({
        "sequence": ["AC", "DDD", "AAAA", "AX", None],
        "label": [0, 1, 0, 1, 0],
    })

    clean, _, _ = process_sequences(
        data, "sequence", max_len=4
    )
    assert clean["sequence"].tolist() == ["AC", "DDD", "AAAA"]

    dataset = ProteinDataset(
        clean["sequence"].tolist(),
        clean["label"].tolist(),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_proteins,
    )

    model = ProteinClassifier(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_loss = train_epoch(model, optimizer, loader)
    metrics = evaluate(model, loader)

    assert math.isfinite(train_loss)
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
