#!/usr/bin/env python
"""Cluster saved segment embeddings with agglomerative clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apollo_diarization_gt.clustering import agglomerative_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help="Path to .npy embedding matrix")
    parser.add_argument("--output", default="cluster_labels.npy", help="Output .npy labels file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine distance threshold")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    labels = agglomerative_cluster(embeddings, distance_threshold=args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, labels)
    print(f"saved {len(labels)} cluster labels to {output}")


if __name__ == "__main__":
    main()
