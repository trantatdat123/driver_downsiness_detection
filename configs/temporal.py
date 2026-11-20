"""Configuration dataclasses and YAML loader for Stage 3 temporal heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TEMPORAL_ARCHS = ("gru", "lstm", "transformer")


@dataclass(frozen=True)
class TemporalConfig:
    """Hyperparameters for CausalGRU, CausalLSTM, and CausalTransformer."""

    arch: str = "gru"
    input_dim: int = 130
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.35
    use_layer_norm: bool = True

    # Transformer-specific parameters
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 128
    max_seq_len: int = 256

    # Sliding window metadata used by training/streaming
    window_size: int = 30

    def __post_init__(self) -> None:
        arch_norm = self.arch.lower()
        if arch_norm not in SUPPORTED_TEMPORAL_ARCHS:
            raise ValueError(
                f"Unsupported temporal arch '{self.arch}'. Choose from {SUPPORTED_TEMPORAL_ARCHS}."
            )
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be > 0 and divisible by nhead ({self.nhead})"
            )
        if self.dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be > 0, got {self.dim_feedforward}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")


def load_temporal_config(config_path: str | Path) -> TemporalConfig:
    """Load TemporalConfig from a YAML file (reads top-level or `temporal:` section)."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    sec = raw.get("temporal", raw)
    return TemporalConfig(
        arch=str(sec.get("arch", "gru")).lower(),
        input_dim=int(sec.get("input_dim", 130)),
        hidden_size=int(sec.get("hidden_size", 64)),
        num_layers=int(sec.get("num_layers", 1)),
        dropout=float(sec.get("dropout", 0.35)),
        use_layer_norm=bool(sec.get("use_layer_norm", True)),
        d_model=int(sec.get("d_model", 64)),
        nhead=int(sec.get("nhead", 4)),
        dim_feedforward=int(sec.get("dim_feedforward", 128)),
        max_seq_len=int(sec.get("max_seq_len", 256)),
        window_size=int(sec.get("window_size", 30)),
    )
