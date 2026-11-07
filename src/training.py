import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import ProteinDataset, collate_proteins
from src.model import ProteinClassifier
from src.process_data import process_sequences


def train_step(model, optimizer, batch):
    model.train()
    features, labels, lengths = batch

    optimizer.zero_grad()
    logits = model(features, lengths)
    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model, optimizer, loader):
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        loss = train_step(model, optimizer, batch)
        batch_size = len(batch[1])

        total_loss += loss * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for features, labels, lengths in loader:
        logits = model(features, lengths)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
    }



def train_model():
    torch.manual_seed(42)

    data = pd.read_csv("data/proteins.csv")
    clean, _, _ = process_sequences(
        data, "sequence", max_len=800
    )

    classes = sorted(clean["protein_family"].unique())
    class_to_index = {
        name: index for index, name in enumerate(classes)
    }
    clean["label"] = clean["protein_family"].map(class_to_index)

    validation = pd.concat([
        group.sample(frac=0.2, random_state=42)
        for _, group in clean.groupby("protein_family")
    ])
    training = clean.drop(index=validation.index)

    def make_loader(frame, shuffle):
        dataset = ProteinDataset(
            frame["sequence"].tolist(),
            frame["label"].tolist(),
        )
        return DataLoader(
            dataset,
            batch_size=16,
            shuffle=shuffle,
            collate_fn=collate_proteins,
        )

    train_loader = make_loader(training, shuffle=True)
    val_loader = make_loader(validation, shuffle=False)

    model = ProteinClassifier(
        num_classes=len(classes),
        hidden_size=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    baseline = (
        validation["label"]
        .eq(training["label"].mode().iloc[0])
        .mean()
    )
    print(f"Train: {len(training)}, validation: {len(validation)}")
    print(f"Accuracy при предсказании частого класса: {baseline:.3f}")

    for epoch in range(1, 101):
        train_loss = train_epoch(model, optimizer, train_loader)
        metrics = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_accuracy={metrics['accuracy']:.3f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "hidden_size": 32,
        },
        "protein_classifier.pt",
    )


if __name__ == "__main__":
    train_model()