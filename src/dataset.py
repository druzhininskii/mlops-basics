import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def one_hot_encode(sequence: str) -> list[list[float]]:
    encoded = []

    for amino_acid in sequence:
        if amino_acid not in AMINO_ACIDS:
            raise ValueError(f"Unknown amino acid: {amino_acid}")
        vector = []
        for candidate in AMINO_ACIDS:
            if amino_acid == candidate:
                vector.append(1.0)
            else:
                vector.append(0.0)
        encoded.append(vector)

    return encoded


class ProteinDataset(Dataset):

    def __init__(self, sequences: list[str], labels: list[int]):
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have equal lengths")
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        features = torch.tensor(
            one_hot_encode(self.sequences[index]),
            dtype=torch.float32).reshape(-1, 20)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long)

        return features, label


def collate_proteins(batch):
    features, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in features],
        dtype=torch.long
    )
    padded = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )

    return padded, torch.stack(labels), lengths

