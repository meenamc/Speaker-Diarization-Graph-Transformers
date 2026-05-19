"""Graph Transformer network for speaker-cluster embedding prediction.

The model consumes a graph of speech-segment embeddings. Edges encode relations
between segments and optional Laplacian eigenvector positional encodings provide
structural information to the transformer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


@dataclass
class GraphTransformerConfig:
    """Configuration for :class:`GraphTransformerNet`."""

    in_dim: int = 256
    hidden_dim: int = 256
    out_dim: int = 276
    n_heads: int = 8
    n_layers: int = 2
    in_feat_dropout: float = 0.1
    dropout: float = 0.4
    layer_norm: bool = False
    batch_norm: bool = True
    residual: bool = True
    lap_pos_enc: bool = True
    pos_enc_dim: int = 10


class MultiHeadAttentionLayer(MessagePassing):
    """Sparse graph multi-head attention layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, use_bias: bool = False):
        super().__init__(aggr="add", node_dim=0)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.query = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.key = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.value = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        query = self.query(x).view(-1, self.num_heads, self.out_dim)
        key = self.key(x).view(-1, self.num_heads, self.out_dim)
        value = self.value(x).view(-1, self.num_heads, self.out_dim)
        return self.propagate(edge_index, query=query, key=key, value=value, size=None)

    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        attention_score = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_dim)
        attention_score = softmax(attention_score, index, ptr, size_i)
        return value_j * attention_score.unsqueeze(-1)


class GraphTransformerLayer(nn.Module):
    """A graph-transformer block with residual attention and feed-forward layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.4,
        layer_norm: bool = False,
        batch_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual
        self.use_layer_norm = layer_norm
        self.use_batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads)
        self.output_projection = nn.Linear(out_dim, out_dim)

        self.feed_forward_1 = nn.Linear(out_dim, out_dim * 2)
        self.feed_forward_2 = nn.Linear(out_dim * 2, out_dim)

        if layer_norm:
            self.layer_norm_1 = nn.LayerNorm(out_dim)
            self.layer_norm_2 = nn.LayerNorm(out_dim)
        if batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(out_dim)
            self.batch_norm_2 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual_input = x
        h = self.attention(x, edge_index).view(-1, self.out_dim)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.output_projection(h)
        if self.residual:
            h = residual_input + h
        if self.use_layer_norm:
            h = self.layer_norm_1(h)
        if self.use_batch_norm:
            h = self.batch_norm_1(h)

        residual_input = h
        h = self.feed_forward_1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.feed_forward_2(h)
        if self.residual:
            h = residual_input + h
        if self.use_layer_norm:
            h = self.layer_norm_2(h)
        if self.use_batch_norm:
            h = self.batch_norm_2(h)
        return h


class GraphTransformerNet(nn.Module):


    def __init__(self, config: GraphTransformerConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = GraphTransformerConfig(
                in_dim=config.get("in_dim", 256),
                hidden_dim=config.get("hidden_dim", 256),
                out_dim=config.get("out_dim", 276),
                n_heads=config.get("n_heads", 8),
                n_layers=config.get("L", config.get("n_layers", 2)),
                in_feat_dropout=config.get("in_feat_dropout", 0.1),
                dropout=config.get("dropout", 0.4),
                layer_norm=config.get("layer_norm", False),
                batch_norm=config.get("batch_norm", True),
                residual=config.get("residual", True),
                lap_pos_enc=config.get("lap_pos_enc", True),
                pos_enc_dim=config.get("pos_enc_dim", 10),
            )
        self.config = config

        self.node_embedding = nn.Linear(config.in_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.in_feat_dropout)

        if config.lap_pos_enc:
            self.position_embedding = nn.Linear(config.pos_enc_dim, config.hidden_dim)
        else:
            self.position_embedding = None

        self.layers = nn.ModuleList(
            GraphTransformerLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.n_heads,
                dropout=config.dropout,
                layer_norm=config.layer_norm,
                batch_norm=config.batch_norm,
                residual=config.residual,
            )
            for _ in range(config.n_layers)
        )
        self.classifier = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        lap_pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_embedding(x)
        if self.position_embedding is not None:
            if lap_pos_enc is None:
                raise ValueError("lap_pos_enc is required when lap_pos_enc=True")
            h = h + self.position_embedding(lap_pos_enc.float())
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.classifier(h)
