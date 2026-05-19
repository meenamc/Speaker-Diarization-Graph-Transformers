# Semi-Supervised Speaker Diarization with Graph Transformers and LLM-Based Segmentation

This repository contains an implementation of a speaker diarization pipeline for naturalistic Apollo mission audio. The method combines word-level alignment, LLM-assisted speaker segmentation, multi-segmentation refinement, graph construction over speech segments, and a Graph Transformer network for speaker clustering.

## Overview

The pipeline is designed for long-duration, multi-speaker communication audio with noisy channels, short turn-taking windows, speaker imbalance, and changing speaker roles across mission shifts.

Main components:

- Word-level transcript alignment
- LLM-assisted speaker-change labeling
- Multi-segmentation refinement around word boundaries
- Graph construction from segmented speech embeddings
- Laplacian positional encodings for graph structure
- Graph Transformer model for speaker representation and clustering
- Agglomerative clustering over learned segment representations

## Repository layout

```text
src/apollo_diarization_gt/
├── graph_transformer.py   # Graph Transformer architecture
├── data.py                # PyTorch Geometric dataset loader
├── train.py               # Training and evaluation loop
├── clustering.py          # Agglomerative clustering helper
└── segmentation.py        # Multi-segmentation refinement utilities

scripts/
├── train.py               # Command-line training entry point
└── cluster_embeddings.py  # Cluster learned embeddings

configs/
└── default.yaml           # Example configuration
```

## Installation

```bash
pip install -e .
```

Install PyTorch and PyTorch Geometric versions compatible with your CUDA environment before training on GPU.

## Training

```bash
python scripts/train.py \
  --train-dir /path/to/graphs/Train \
  --eval-dir /path/to/graphs/Eval \
  --output-dir checkpoints \
  --out-dim 276 \
  --epochs 100
```

Graph files are expected to be PyTorch Geometric objects saved as:

```text
data_0.pt
data_1.pt
data_2.pt
...
```

Each graph should contain node features, graph edges, speaker labels, and optional Laplacian positional encodings.

## Clustering embeddings

```bash
python scripts/cluster_embeddings.py \
  --embeddings embeddings.npy \
  --output cluster_labels.npy \
  --threshold 0.65
```

## Citation

If you use this work, please cite the associated ICASSP paper:

```bibtex
@inproceedings{meena2025semi,
  title={Semi-Supervised Speaker Diarization using Graph Networks and LLMs on Naturalistic Apollo 11 Data},
  author={M. C. Shekar, Meena and Hansen, John H. L.},
  booktitle={ICASSP},
  year={2025}
}
```
