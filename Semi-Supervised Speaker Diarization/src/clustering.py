"""Clustering helpers for graph-transformer diarization outputs."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def agglomerative_cluster(
    embeddings: torch.Tensor | np.ndarray,
    distance_threshold: float = 0.65,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Cluster segment embeddings with average-linkage agglomerative clustering."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold if n_clusters is None else None,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)
