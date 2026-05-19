"""Training loop for the graph-transformer diarization model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .data import GraphDiarizationDataset
from .graph_transformer import GraphTransformerConfig, GraphTransformerNet


def _labels_from_batch(batch) -> torch.Tensor:
    """Extract speaker labels from ``batch.y``.

    Expected labels are rows of ``[speaker_id, is_augmented, file_id, segment_id]``.
    """
    if batch.y.ndim == 1:
        return batch.y.long()
    return batch.y[:, 0].long()


def _laplacian_encoding(batch, device: torch.device):
    """Return the available positional encoding field from a graph batch."""
    lap_pos_enc = getattr(batch, "lap_cos", None)
    if lap_pos_enc is None:
        lap_pos_enc = getattr(batch, "lap_bin", None)
    if lap_pos_enc is None:
        return None
    return lap_pos_enc.to(device)


def train_epoch(
    model: GraphTransformerNet,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        if batch.x.size(0) <= 2:
            continue
        batch = batch.to(device)
        lap_pos_enc = _laplacian_encoding(batch, device)
        logits = model(batch.x, batch.edge_index, lap_pos_enc)
        labels = _labels_from_batch(batch).to(device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_accuracy(model: GraphTransformerNet, loader: Iterable, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        if batch.x.size(0) <= 2:
            continue
        batch = batch.to(device)
        lap_pos_enc = _laplacian_encoding(batch, device)
        logits = model(batch.x, batch.edge_index, lap_pos_enc)
        predictions = logits.argmax(dim=1)
        labels = _labels_from_batch(batch).to(device)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return correct / total if total else 0.0


def train_model(
    train_dir: str | Path,
    output_dir: str | Path,
    config: GraphTransformerConfig,
    eval_dir: str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: str | torch.device | None = None,
    checkpoint_every: int = 50,
) -> GraphTransformerNet:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GraphTransformerNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(GraphDiarizationDataset(train_dir), batch_size=batch_size, shuffle=True)
    eval_loader = None if eval_dir is None else DataLoader(GraphDiarizationDataset(eval_dir), batch_size=1)

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        message = f"epoch={epoch:04d} train_loss={loss:.4f}"
        if eval_loader is not None:
            accuracy = evaluate_accuracy(model, eval_loader, device)
            message += f" eval_accuracy={accuracy:.4f}"
        print(message)

        if checkpoint_every and epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), output_dir / f"graph_transformer_epoch_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "graph_transformer_final.pt")
    return model
