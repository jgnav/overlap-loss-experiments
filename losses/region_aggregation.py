"""Weighted region statistics and distribution matching (population moments)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = (
    "mean", "mean_scalar_variance", "mean_variance",
    "mean_covariance", "mean_projected_variance", "mean_projected_covariance",
    "swd", "mean_centered_swd", "mean_normalized_swd",
)


class RegionAggregation(nn.Module):
    def __init__(self, method="mean"):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"region_aggregation must be one of {METHODS}")
        self.method = method
        # Fixed ablation recipe: neutral unit coefficients, no tuned scaling.
        self.beta = self.scale_weight = self.shape_weight = 1.
        self.projections, self.quantiles = 64, 128
        self.eps, self.seed = 1e-8, 0
        # Reconstructed deterministically from config on resume; no checkpoint
        # migration or dependence on the training RNG / rank is needed.
        self.register_buffer("directions", torch.empty(0), persistent=False)

    def project(self, x):
        if self.directions.shape != (x.shape[-1], self.projections):
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            directions = torch.randn(x.shape[-1], self.projections, generator=generator)
            self.directions = F.normalize(directions, dim=0).to(x.device)
        return x @ self.directions.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def moments(x, weights):
        weights = weights / weights.sum(-1, keepdim=True)
        x = x.float().masked_fill(weights[..., None] == 0, 0)
        mean = (x * weights[..., None]).sum(1)
        residual = (x - mean[:, None]).masked_fill(weights[..., None] == 0, 0)
        return mean, residual, weights

    @staticmethod
    def covariance_distance(s, t, sw, tw):
        # ||S^T S - T^T T||_F^2 via patch Gram matrices. Exact full-D
        # covariance matching without allocating a prototypes x prototypes array.
        s = s * sw.sqrt()[..., None]
        t = t * tw.sqrt()[..., None]
        ss, tt, st = s @ s.transpose(1, 2), t @ t.transpose(1, 2), s @ t.transpose(1, 2)
        return (ss.square().sum((1, 2)) + tt.square().sum((1, 2))
                - 2 * st.square().sum((1, 2))).clamp_min(0)

    def empirical_quantiles(self, x, weights):
        # Inverse empirical CDF at shared midpoint quantiles; area weights are
        # probability masses, and zero-weight patches never enter the CDF.
        output = []
        levels = (torch.arange(self.quantiles, device=x.device, dtype=x.dtype) + .5) / self.quantiles
        for values, mass in zip(x, weights):
            selected = mass > 0
            values, mass = values[selected], mass[selected]
            sorted_values, order = values.sort(dim=0)
            cdf = mass[order].cumsum(0).transpose(0, 1).contiguous()
            cdf = cdf / cdf[:, -1:]
            index = torch.searchsorted(cdf, levels.expand(x.shape[-1], -1).contiguous())
            output.append(sorted_values.transpose(0, 1).gather(1, index.clamp_max(len(values) - 1)))
        return torch.stack(output)

    def swd(self, s, t, sw, tw):
        return (self.empirical_quantiles(s, sw) - self.empirical_quantiles(t, tw)).square().mean((1, 2))

    def forward(self, s, t, sw, tw, mean_loss):
        """Return one loss per region; caller supplies detached teacher values."""
        t = t.detach()
        _, sr, sw = self.moments(s, sw)
        _, tr, tw = self.moments(t, tw)
        method = self.method
        if method == "mean_scalar_variance":
            sv = (sr.square().sum(-1) * sw).sum(-1)
            tv = (tr.square().sum(-1) * tw).sum(-1)
            extra = ((sv + self.eps).sqrt() - (tv + self.eps).sqrt()).square()
        elif method == "mean_variance":
            sv = (sr.square() * sw[..., None]).sum(1)
            tv = (tr.square() * tw[..., None]).sum(1)
            extra = ((sv + self.eps).sqrt() - (tv + self.eps).sqrt()).square().mean(-1)
        elif method == "mean_covariance":
            extra = self.covariance_distance(sr, tr, sw, tw)
        elif method == "mean_projected_covariance":
            sproj, tproj = self.project(sr), self.project(tr)
            sc = sproj.transpose(1, 2) @ (sproj * sw[..., None])
            tc = tproj.transpose(1, 2) @ (tproj * tw[..., None])
            extra = (sc - tc).square().sum((1, 2))
        elif method == "swd":
            return self.swd(self.project(s), self.project(t), sw, tw)
        elif method in ("mean_projected_variance", "mean_centered_swd", "mean_normalized_swd"):
            sproj, tproj = self.project(sr), self.project(tr)
            if method == "mean_centered_swd":
                extra = self.swd(sproj, tproj, sw, tw)
            else:
                ss = ((sproj.square() * sw[..., None]).sum(1) + self.eps).sqrt()
                ts = ((tproj.square() * tw[..., None]).sum(1) + self.eps).sqrt()
                extra = (ss - ts).square().mean(-1)
                if method == "mean_normalized_swd":
                    shape = self.swd(sproj / (ss.detach()[:, None] + self.eps),
                                     tproj / (ts.detach()[:, None] + self.eps), sw, tw)
                    return mean_loss + self.scale_weight * extra + self.shape_weight * shape
        else:
            return mean_loss
        return mean_loss + self.beta * extra
