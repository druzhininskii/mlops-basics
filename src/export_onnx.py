import argparse
from pathlib import Path

import torch

from src.model import ProteinClassifier


def export_onnx(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ProteinClassifier(len(checkpoint["classes"]), checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = torch.zeros(2, 3, 20)
    lengths = torch.tensor([3, 2])
    batch = torch.export.Dim("batch", min=1)
    sequence = torch.export.Dim("sequence", min=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (features, lengths),
        output_path,
        input_names=["features", "lengths"],
        output_names=["logits"],
        dynamic_shapes=({0: batch, 1: sequence}, {0: batch}),
        external_data=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
