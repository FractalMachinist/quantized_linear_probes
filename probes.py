"""Difference-of-means and logistic-regression linear probes, copied from
chapter1_transformer_interp/exercises/part31_linear_probes/solutions.py (MMProbe, LRProbe).
"""

import torch as t
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor


class MMProbe(t.nn.Module):
    """Difference-of-means probe: direction = mean(positive) - mean(negative)."""

    def __init__(
        self,
        direction: Float[Tensor, " d_model"],
        covariance: Float[Tensor, "d_model d_model"] | None = None,
        atol: float = 1e-3,
    ):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if covariance is not None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = None

    def forward(self, x: Float[Tensor, "n d_model"], iid: bool = False) -> Float[Tensor, " n"]:
        if iid and self.inv is not None:
            return t.sigmoid(x @ self.inv @ self.direction)
        else:
            return t.sigmoid(x @ self.direction)

    def pred(self, x: Float[Tensor, "n d_model"], iid: bool = False) -> Float[Tensor, " n"]:
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        device: str = "cpu",
    ) -> "MMProbe":
        acts, labels = acts.to(device), labels.to(device)
        pos_acts = acts[labels == 1]
        neg_acts = acts[labels == 0]
        pos_mean = pos_acts.mean(0)
        neg_mean = neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], dim=0)
        covariance = centered.t() @ centered / acts.shape[0]

        return MMProbe(direction, covariance=covariance).to(device)


class LRProbe(t.nn.Module):
    """Logistic-regression probe, fit with sklearn on standardized activations."""

    def __init__(self, d_in: int, scaler_mean: Tensor | None = None, scaler_scale: Tensor | None = None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self(x).round()

    @property
    def direction(self) -> Float[Tensor, " d_model"]:
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        C: float = 0.1,
        device: str = "cpu",
    ) -> "LRProbe":
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000)
        lr_model.fit(X_scaled, y)

        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)
        probe = LRProbe(acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr_model.coef_[0], dtype=t.float32).to(device)

        return probe


PROBE_CLASSES: dict[str, type] = {"diff_of_means": MMProbe, "logistic": LRProbe}
