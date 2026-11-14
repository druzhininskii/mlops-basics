import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import ProteinDataset, collate_proteins
from src.model import ProteinClassifier
from src.process_data import process_sequences

import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime


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



def train_model(
        batch_size: int = 16,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        seed: int = 42,
        max_sequence_length: int = 800,
):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("protein-classification")

    torch.manual_seed(seed)

    data = pd.read_csv("data/proteins.csv")
    clean, _, _ = process_sequences(
        data, "sequence", max_len=max_sequence_length
    )

    classes = sorted(clean["protein_family"].unique())
    class_to_index = {
        name: index for index, name in enumerate(classes)
    }
    clean["label"] = clean["protein_family"].map(class_to_index)

    validation = pd.concat([
        group.sample(frac=0.2, random_state=seed)
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
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_proteins,
        )

    train_loader = make_loader(training, shuffle=True)
    val_loader = make_loader(validation, shuffle=False)

    mlflow.pytorch.autolog(log_every_n_step=1)
    run_name = f"lr_{learning_rate}_batch_{batch_size}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "seed": seed,
            "max_sequence_length": max_sequence_length,
        })

        model = ProteinClassifier(
            num_classes=len(classes),
            hidden_size=hidden_size,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        baseline = (
            validation["label"]
            .eq(training["label"].mode().iloc[0])
            .mean()
        )
        print(f"Train: {len(training)}, validation: {len(validation)}")
        print(f"Accuracy for most frequent class: {baseline:.3f}")

        writer = SummaryWriter()
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, optimizer, train_loader)
            metrics = evaluate(model, val_loader)

            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("val_loss", metrics["loss"], epoch)
            writer.add_scalar("val_accuracy", metrics["accuracy"], epoch)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"val_accuracy={metrics['accuracy']:.3f}"
            )

        writer.close()

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_path = results_dir / f"protein_classifier_{timestamp}.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "hidden_size": hidden_size,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(checkpoint_path)

if __name__ == "__main__":
    train_model()