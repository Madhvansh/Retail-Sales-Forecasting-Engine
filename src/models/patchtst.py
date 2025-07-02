"""PatchTST — a channel-independent patched Transformer for forecasting.

Reference: Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023).

This is a compact, self-contained implementation tuned to run on CPU for the
benchmark. It is a *global* supervised model: a single network is trained on
windows drawn from every series in the panel and emits a multi-quantile head
so it is directly comparable to the probabilistic foundation model.

Demand is non-negative and heavy-tailed, so we train in ``log1p`` space with
per-window instance normalisation (RevIN-style) and a multi-quantile pinball
loss.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import SeriesPanel, TimeSeries
from src.models.base import Forecast, Forecaster
from src.utils import get_logger

log = get_logger("models.patchtst")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


if _TORCH:

    class _WindowDataset(Dataset):
        """Sliding (context, target) windows in log1p space."""

        def __init__(self, panel: SeriesPanel, context: int, horizon: int):
            self.samples: list[tuple[np.ndarray, np.ndarray]] = []
            win = context + horizon
            for ts in panel:
                y = np.log1p(np.clip(ts.values, 0, None))
                if len(y) < win:
                    continue
                # stride to keep training tractable
                stride = max(1, horizon // 2)
                for start in range(0, len(y) - win + 1, stride):
                    ctx = y[start : start + context]
                    tgt = y[start + context : start + win]
                    self.samples.append((ctx, tgt))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            ctx, tgt = self.samples[idx]
            return torch.from_numpy(ctx).float(), torch.from_numpy(tgt).float()

    class _PatchTSTNet(nn.Module):
        def __init__(
            self,
            context: int,
            horizon: int,
            n_quantiles: int,
            patch_len: int,
            stride: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.context = context
            self.horizon = horizon
            self.n_quantiles = n_quantiles
            self.patch_len = patch_len
            self.stride = stride

            self.n_patches = (context - patch_len) // stride + 1
            self.patch_embed = nn.Linear(patch_len, d_model)
            self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model * self.n_patches, horizon * n_quantiles)

        def _patchify(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, context) -> (B, n_patches, patch_len)
            return x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            patches = self._patchify(x)                 # (B, P, patch_len)
            h = self.patch_embed(patches) + self.pos    # (B, P, d_model)
            h = self.encoder(h)                         # (B, P, d_model)
            h = h.flatten(1)                            # (B, P*d_model)
            out = self.head(h)                          # (B, H*Q)
            return out.view(-1, self.horizon, self.n_quantiles)


class PatchTST(Forecaster):
    name = "patchtst"
    is_global = True

    def __init__(
        self,
        context_length: int = 168,
        horizon: int = 28,
        quantile_levels: np.ndarray | None = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for PatchTST")
        self.context_length = context_length
        self.horizon = horizon
        self.quantile_levels = np.asarray(
            quantile_levels if quantile_levels is not None else [0.1, 0.5, 0.9],
            dtype=np.float64,
        )
        self.params = dict(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = self._resolve_device(device)
        self.net: _PatchTSTNet | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if _TORCH and torch.cuda.is_available() else "cpu"

    def _pinball(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, H, Q), target: (B, H)
        levels = torch.tensor(self.quantile_levels, device=pred.device).view(1, 1, -1)
        err = target.unsqueeze(-1) - pred
        loss = torch.maximum(levels * err, (levels - 1.0) * err)
        return loss.mean()

    def fit(self, panel: SeriesPanel) -> PatchTST:
        torch.manual_seed(self.seed)
        ds = _WindowDataset(panel, self.context_length, self.horizon)
        if len(ds) == 0:
            raise ValueError("No training windows; series too short for context+horizon")
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net = _PatchTSTNet(
            context=self.context_length,
            horizon=self.horizon,
            n_quantiles=len(self.quantile_levels),
            **self.params,
        ).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        log.info(
            "Training PatchTST on %d windows (%d epochs, device=%s)",
            len(ds), self.epochs, self.device,
        )
        self.net.train()
        for epoch in range(self.epochs):
            running = 0.0
            for ctx, tgt in loader:
                ctx, tgt = ctx.to(self.device), tgt.to(self.device)
                # RevIN-style per-window instance normalisation.
                mu = ctx.mean(dim=1, keepdim=True)
                sigma = ctx.std(dim=1, keepdim=True) + 1e-5
                ctx_n = (ctx - mu) / sigma
                pred = self.net(ctx_n)               # (B, H, Q) normalised
                pred = pred * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
                loss = self._pinball(pred, tgt)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                running += loss.item() * ctx.size(0)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                log.info("  epoch %d/%d  pinball=%.4f", epoch + 1, self.epochs, running / len(ds))
        return self

    @torch.no_grad() if _TORCH else (lambda f: f)
    def predict(
        self,
        history: TimeSeries,
        horizon: int,
        quantile_levels: np.ndarray,
    ) -> Forecast:
        if self.net is None:
            raise RuntimeError("PatchTST.fit must be called before predict")
        self.net.eval()

        y = np.log1p(np.clip(history.values, 0, None))
        ctx = y[-self.context_length :]
        if len(ctx) < self.context_length:
            ctx = np.pad(ctx, (self.context_length - len(ctx), 0), mode="edge")

        x = torch.from_numpy(ctx).float().view(1, -1).to(self.device)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-5
        pred = self.net((x - mu) / sigma)            # (1, H, Q)
        pred = pred * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
        pred = pred.squeeze(0).cpu().numpy()         # (H, Q) in log space
        pred = np.expm1(pred)
        pred = np.clip(pred, 0.0, None)

        # Enforce monotonicity across quantiles then map to requested levels.
        pred = np.sort(pred, axis=1)                 # (H, Q_train)
        q_train = self.quantile_levels
        quantiles = np.vstack([
            np.interp(quantile_levels, q_train, pred[h]) for h in range(pred.shape[0])
        ]).T                                         # (Q_req, H)

        median_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
        point = quantiles[median_idx]
        return Forecast(history.item_id, point, quantiles, np.asarray(quantile_levels))
