"""Dataset utilities for graph-based diarization experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Dataset


class GraphDiarizationDataset(Dataset):
    """Load PyTorch Geometric graph files named ``data_0.pt``, ``data_1.pt``, ..."""

    def __init__(self, root: str | Path, transform=None, pre_transform=None):
        self.root_path = Path(root)
        super().__init__(str(self.root_path), transform, pre_transform)

    def len(self) -> int:
        return len(list(self.root_path.glob("data_*.pt")))

    def get(self, idx: int):
        graph_path = self.root_path / f"data_{idx}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        return torch.load(graph_path)
