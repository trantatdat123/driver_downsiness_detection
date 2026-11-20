"""Stage 3 causal temporal heads: CausalGRU, CausalLSTM, and CausalTransformer."""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
import torch.nn as nn

from configs.temporal import TemporalConfig

GRUState: TypeAlias = torch.Tensor
LSTMState: TypeAlias = tuple[torch.Tensor, torch.Tensor]
TemporalState: TypeAlias = GRUState | LSTMState | torch.Tensor


class _BaseCausalRNN(nn.Module):
    """Shared unidirectional RNN temporal head returning per-frame logits (B, T, 1)."""

    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm: nn.Module = (
            nn.LayerNorm(config.input_dim) if config.use_layer_norm else nn.Identity()
        )
        self.rnn = self._build_rnn(config)
        self.head_norm: nn.Module = (
            nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        )
        self.head_dropout: nn.Module = (
            nn.Dropout(p=config.dropout) if config.dropout > 0.0 else nn.Identity()
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def _build_rnn(self, config: TemporalConfig) -> nn.Module:
        raise NotImplementedError

    def _encode_sequence(
        self,
        frame_feats: torch.Tensor,
        hidden_state: TemporalState | None = None,
    ) -> tuple[torch.Tensor, TemporalState]:
        if frame_feats.ndim != 3 or frame_feats.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected frame_feats of shape (B, T, {self.config.input_dim}), "
                f"got {tuple(frame_feats.shape)}"
            )

        # (B, T, 2*D + 2) -> (B, T, 2*D + 2)
        normed = self.input_norm(frame_feats)
        # (B, T, 2*D + 2) -> (B, T, hidden_size)
        seq_out, next_state = self.rnn(normed, hidden_state)
        # (B, T, hidden_size) -> (B, T, hidden_size)
        seq_out = self.head_dropout(self.head_norm(seq_out))
        # (B, T, hidden_size) -> (B, T, 1)
        logits = self.classifier(seq_out)
        return logits, next_state

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """Predict per-frame drowsiness logits without sigmoid: (B, T, D_in) -> (B, T, 1)."""
        logits, _ = self._encode_sequence(frame_feats, hidden_state=None)
        return logits

    def forward_with_state(
        self,
        frame_feats: torch.Tensor,
        hidden_state: TemporalState | None = None,
    ) -> tuple[torch.Tensor, TemporalState]:
        """Stateful streaming forward returning (logits, next_hidden_state)."""
        return self._encode_sequence(frame_feats, hidden_state=hidden_state)


class CausalGRU(_BaseCausalRNN):
    """Unidirectional GRU temporal head (bidirectional=False enforced)."""

    def _build_rnn(self, config: TemporalConfig) -> nn.GRU:
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        return nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=False,
        )

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GRUState:
        """Create zero initial hidden state (num_layers, B, hidden_size)."""
        return torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_size,
            device=device,
            dtype=dtype,
        )


class CausalLSTM(_BaseCausalRNN):
    """Unidirectional LSTM temporal head (bidirectional=False enforced)."""

    def _build_rnn(self, config: TemporalConfig) -> nn.LSTM:
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        return nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=False,
        )

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> LSTMState:
        """Create zero initial (h_0, c_0) states of shape (num_layers, B, hidden_size)."""
        shape = (self.config.num_layers, batch_size, self.config.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding over sequence time steps."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe = torch.zeros(1, max_seq_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            cos_cols = pe[0, :, 1::2].shape[1]
            pe[0, :, 1::2] = torch.cos(position * div_term[:cos_cols])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding: (B, T, d_model) -> (B, T, d_model)."""
        t = x.shape[1]
        if t > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len={self.pe.shape[1]}"
            )
        # (B, T, d_model) + (1, T, d_model) -> (B, T, d_model)
        return x + self.pe[:, :t, :].to(dtype=x.dtype)


class CausalTransformer(nn.Module):
    """Causal TransformerEncoder temporal head using upper-triangular -inf attention mask."""

    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_norm: nn.Module = (
            nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        )
        self.pos_encoder = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        self.input_dropout: nn.Module = (
            nn.Dropout(p=config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        self.head_norm: nn.Module = (
            nn.LayerNorm(config.d_model) if config.use_layer_norm else nn.Identity()
        )
        self.classifier = nn.Linear(config.d_model, 1)

        # Precompute strict upper-triangular causal mask (-inf for j > i)
        causal_mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """Predict causal per-frame logits without sigmoid: (B, T, D_in) -> (B, T, 1)."""
        if frame_feats.ndim != 3 or frame_feats.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected frame_feats of shape (B, T, {self.config.input_dim}), "
                f"got {tuple(frame_feats.shape)}"
            )

        t = frame_feats.shape[1]
        if t > self.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}"
            )

        # (B, T, 2*D + 2) -> (B, T, d_model)
        x = self.input_norm(self.input_proj(frame_feats))
        # (B, T, d_model) -> (B, T, d_model)
        x = self.input_dropout(self.pos_encoder(x))

        # Slice precomputed causal mask: (max_seq_len, max_seq_len) -> (T, T)
        attn_mask = self.causal_mask[:t, :t].to(dtype=x.dtype)
        # (B, T, d_model) -> (B, T, d_model)
        encoded = self.encoder(x, mask=attn_mask, is_causal=True)
        # (B, T, d_model) -> (B, T, 1)
        logits = self.classifier(self.head_norm(encoded))
        return logits

    def forward_with_state(
        self,
        frame_feats: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward using a rolling feature history buffer (B, T_hist, D_in)."""
        t_cur = frame_feats.shape[1]
        if hidden_state is not None:
            # (B, T_hist, D_in) + (B, T_cur, D_in) -> (B, T_hist + T_cur, D_in)
            window = torch.cat([hidden_state, frame_feats], dim=1)
        else:
            window = frame_feats

        if window.shape[1] > self.max_seq_len:
            window = window[:, -self.max_seq_len :, :]

        all_logits = self.forward(window)
        # (B, T_win, 1) -> (B, T_cur, 1)
        cur_logits = all_logits[:, -t_cur:, :]
        return cur_logits, window


_TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {
    "gru": CausalGRU,
    "lstm": CausalLSTM,
    "transformer": CausalTransformer,
}


def build_temporal_head(config: TemporalConfig) -> CausalGRU | CausalLSTM | CausalTransformer:
    """Instantiate a causal temporal head from TemporalConfig.arch."""
    arch = config.arch.lower()
    if arch not in _TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unsupported temporal arch '{config.arch}'. Expected one of {tuple(_TEMPORAL_REGISTRY)}."
        )
    return _TEMPORAL_REGISTRY[arch](config)  # type: ignore[return-value]


class TemporalHead(nn.Module):
    """Unified Stage 3 wrapper switching between CausalGRU, CausalLSTM, and CausalTransformer."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else TemporalConfig()
        self.arch = self.config.arch.lower()
        self.head = build_temporal_head(self.config)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """Predict causal per-frame logits: (B, T, D_in) -> (B, T, 1)."""
        return self.head(frame_feats)

    def forward_with_state(
        self,
        frame_feats: torch.Tensor,
        hidden_state: TemporalState | None = None,
    ) -> tuple[torch.Tensor, TemporalState]:
        """Stateful streaming forward returning (logits, next_state)."""
        return self.head.forward_with_state(frame_feats, hidden_state=hidden_state)
