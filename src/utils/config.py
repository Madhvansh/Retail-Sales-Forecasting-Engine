"""Lightweight YAML configuration with attribute access and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested overrides.

    Example
    -------
    >>> cfg = Config({"data": {"horizon": 28}})
    >>> cfg.data.horizon
    28
    >>> cfg.override("data.horizon", 14)
    >>> cfg.data.horizon
    14
    """

    def __init__(self, mapping: dict | None = None) -> None:
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def override(self, dotted_key: str, value: Any) -> None:
        """Set a nested value using a dotted path, creating intermediates."""
        keys = dotted_key.split(".")
        node: Config = self
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], Config):
                node[key] = Config()
            node = node[key]
        node[keys[-1]] = value

    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def copy(self) -> Config:
        return Config(copy.deepcopy(self.to_dict()))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and apply optional dotted-key overrides."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    for key, value in (overrides or {}).items():
        cfg.override(key, value)
    return cfg
