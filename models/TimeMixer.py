"""Channel-independent TimeMixer baseline (ICLR 2024).

This is the forecasting path from the TSLib implementation, adapted only at
the boundary from TSLib's (B, L, V) API to DUCT's (B, V, L) API. Calendar
features are intentionally omitted because the DUCT protocol does not expose
them to any compared model.
"""
import torch
import torch.nn as nn


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = _MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_average(x)
        return x - trend, trend


class _ScaleNorm(nn.Module):
    def __init__(self, n_vars: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_vars))
        self.bias = nn.Parameter(torch.zeros(n_vars))
        self.mean = None
        self.stdev = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        return (x - self.mean) / self.stdev * self.weight + self.bias

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class _TokenEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            1, d_model, kernel_size=3, padding=1,
            padding_mode="circular", bias=False,
        )
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))


class _SeasonMixing(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i], lengths[i + 1]),
                nn.GELU(),
                nn.Linear(lengths[i + 1], lengths[i + 1]),
            )
            for i in range(len(lengths) - 1)
        ])

    def forward(self, values):
        high = values[0]
        low = values[1]
        outputs = [high.transpose(1, 2)]
        for i in range(len(values) - 1):
            low = low + self.layers[i](high)
            high = low
            if i + 2 < len(values):
                low = values[i + 2]
            outputs.append(high.transpose(1, 2))
        return outputs


class _TrendMixing(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i + 1], lengths[i]),
                nn.GELU(),
                nn.Linear(lengths[i], lengths[i]),
            )
            for i in reversed(range(len(lengths) - 1))
        ])

    def forward(self, values):
        reversed_values = list(reversed(values))
        low = reversed_values[0]
        high = reversed_values[1]
        outputs = [low.transpose(1, 2)]
        for i in range(len(reversed_values) - 1):
            high = high + self.layers[i](low)
            low = high
            if i + 2 < len(reversed_values):
                high = reversed_values[i + 2]
            outputs.append(low.transpose(1, 2))
        outputs.reverse()
        return outputs


class _PastDecomposableMixing(nn.Module):
    def __init__(
        self, lengths, d_model: int, d_ff: int, moving_avg: int,
    ):
        super().__init__()
        self.decomposition = _SeriesDecomp(moving_avg)
        self.season_mixing = _SeasonMixing(lengths)
        self.trend_mixing = _TrendMixing(lengths)
        self.out_cross = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, values):
        seasons = []
        trends = []
        lengths = []
        for value in values:
            lengths.append(value.size(1))
            season, trend = self.decomposition(value)
            seasons.append(season.transpose(1, 2))
            trends.append(trend.transpose(1, 2))

        mixed_seasons = self.season_mixing(seasons)
        mixed_trends = self.trend_mixing(trends)
        return [
            original + self.out_cross(season + trend)[:, :length, :]
            for original, season, trend, length in zip(
                values, mixed_seasons, mixed_trends, lengths
            )
        ]


class TimeMixer(nn.Module):
    def __init__(
        self,
        n_vars: int,
        lookback: int,
        pred_len: int,
        d_model: int = 32,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
        moving_avg: int = 25,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.lengths = [
            lookback // (down_sampling_window ** i)
            for i in range(down_sampling_layers + 1)
        ]

        self.pool = nn.AvgPool1d(down_sampling_window)
        self.norms = nn.ModuleList([
            _ScaleNorm(n_vars) for _ in range(down_sampling_layers + 1)
        ])
        self.embedding = _TokenEmbedding(d_model, dropout)
        self.blocks = nn.ModuleList([
            _PastDecomposableMixing(self.lengths, d_model, d_ff, moving_avg)
            for _ in range(n_layers)
        ])
        self.predictors = nn.ModuleList([
            nn.Linear(length, pred_len) for length in self.lengths
        ])
        self.projection = nn.Linear(d_model, 1)

    def _multi_scale_inputs(self, x: torch.Tensor):
        values = [x]
        current = x.transpose(1, 2)
        for _ in range(self.down_sampling_layers):
            current = self.pool(current)
            values.append(current.transpose(1, 2))
        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, L) -> (B, V, pred_len)."""
        batch, n_vars, _ = x.shape
        values = self._multi_scale_inputs(x.transpose(1, 2))

        embedded = []
        for norm, value in zip(self.norms, values):
            value = norm.norm(value)
            length = value.size(1)
            value = value.transpose(1, 2).reshape(batch * n_vars, length, 1)
            embedded.append(self.embedding(value))

        for block in self.blocks:
            embedded = block(embedded)

        forecasts = []
        for predictor, value in zip(self.predictors, embedded):
            value = predictor(value.transpose(1, 2)).transpose(1, 2)
            value = self.projection(value)
            value = value.reshape(batch, n_vars, self.pred_len).transpose(1, 2)
            forecasts.append(value)

        forecast = torch.stack(forecasts, dim=-1).sum(dim=-1)
        forecast = self.norms[0].denorm(forecast)
        return forecast.transpose(1, 2).contiguous()


__all__ = ["TimeMixer"]
