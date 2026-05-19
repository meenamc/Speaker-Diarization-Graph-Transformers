#!/usr/bin/env python
"""Train the Apollo graph-transformer diarization model."""

from __future__ import annotations

import argparse

from apollo_diarization_gt.graph_transformer import GraphTransformerConfig
from apollo_diarization_gt.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Directory containing training graph files data_*.pt")
    parser.add_argument("--eval-dir", default=None, help="Optional evaluation graph directory")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--in-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, required=True, help="Number of speaker classes")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pos-enc-dim", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GraphTransformerConfig(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        pos_enc_dim=args.pos_enc_dim,
    )
    train_model(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
