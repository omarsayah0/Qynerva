"""Gaussian Diffusion (DDPM) with conditional sampling.

Implements:
  - Linear and cosine noise schedules
  - Forward (q) and reverse (p) processes
  - Training loss: MSE on predicted noise
  - Optional TGAP (topology-aware) loss placeholder
  - DDPM and DDIM sampling
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def linear_beta_schedule(steps: int, beta_start: float, beta_end: float) -> Tensor:
    return torch.linspace(beta_start, beta_end, steps)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> Tensor:
    t = torch.linspace(0, steps, steps + 1) / steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """DDPM wrapper around the UNet and TSA modules."""

    def __init__(
        self,
        unet: nn.Module,
        tsa: nn.Module,
        num_steps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lambda_tgap: float = 0.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.tsa = tsa
        self.num_steps = num_steps
        self.lambda_tgap = lambda_tgap

        # Build noise schedule
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            betas = linear_beta_schedule(num_steps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Derived quantities for forward / reverse
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # Posterior variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_start = batch["image"]   # (B, 1, H, W)
        cond = batch["cond"]       # (B, 5, H, W)
        modality = batch["modality"]  # (B,)

        B = x_start.size(0)
        device = x_start.device

        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # TSA conditioning
        cond_feat = self.tsa(cond)

        # UNet predicts noise
        pred_noise = self.unet(x_noisy, t, cond_feat, modality)

        # MSE loss on predicted noise
        mse_loss = F.mse_loss(pred_noise, noise)

        # Optional TGAP loss (topology-aware placeholder)
        tgap_loss = torch.tensor(0.0, device=device)
        if self.lambda_tgap > 0.0:
            tgap_loss = self._tgap_loss(pred_noise, noise, cond)

        total_loss = mse_loss + self.lambda_tgap * tgap_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss.detach(),
            "tgap_loss": tgap_loss.detach(),
        }

    @staticmethod
    def _tgap_loss(pred_noise: Tensor, true_noise: Tensor, cond: Tensor) -> Tensor:
        """
        Topology-aware gap penalty (TGAP) placeholder.

        Penalizes noise prediction errors in anatomically significant
        regions (tumor + ventricles) more than background.
        """
        tumor_mask = cond[:, 0:1]   # tumor_mask channel
        lv_mask = cond[:, 4:5]      # lv channel
        topology_weight = 1.0 + 2.0 * tumor_mask + 1.0 * lv_mask
        weighted_diff = topology_weight * (pred_noise - true_noise) ** 2
        return weighted_diff.mean()

    # ------------------------------------------------------------------
    # Reverse process: p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_mean_variance(
        self, x: Tensor, t: Tensor, cond_feat: Tensor, modality: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pred_noise = self.unet(x, t, cond_feat, modality)

        x0_pred = (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        ).clamp(-1, 1)

        mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(
        self, x: Tensor, t: Tensor, cond_feat: Tensor, modality: Tensor
    ) -> Tensor:
        mean, log_var = self.p_mean_variance(x, t, cond_feat, modality)
        noise = torch.randn_like(x)
        nonzero = (t > 0).float().view(-1, *([1] * (x.ndim - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    # ------------------------------------------------------------------
    # Full sampling (DDPM)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        modality: Tensor,
        shape: Optional[Tuple] = None,
        progress: bool = True,
    ) -> Tensor:
        B = cond.size(0)
        device = cond.device

        if shape is None:
            H = W = self.unet.input_conv.weight.shape[-1]  # fallback
            shape = (B, 1, 128, 128)

        cond_feat = self.tsa(cond)
        x = torch.randn(shape, device=device)

        iterator = range(self.num_steps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Sampling", leave=False, total=self.num_steps)

        for i in iterator:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond_feat, modality)

        return x.clamp(-1, 1)

    # ------------------------------------------------------------------
    # DDIM sampling (faster, deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        cond: Tensor,
        modality: Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        B = cond.size(0)
        device = cond.device

        cond_feat = self.tsa(cond)
        x = torch.randn(B, 1, 128, 128, device=device)

        step_size = self.num_steps // num_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        from tqdm import tqdm
        for i in tqdm(timesteps, desc="DDIM sampling", leave=False):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t, cond_feat, modality)

            alpha_t = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_prev = self._extract(
                self.alphas_cumprod,
                (t - step_size).clamp(min=0),
                x.shape,
            )

            x0_pred = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = (
                eta
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            )
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = (
                alpha_prev.sqrt() * x0_pred
                + (1 - alpha_prev - sigma**2).sqrt() * pred_noise
                + sigma * noise
            )

        return x.clamp(-1, 1)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: Tensor, t: Tensor, shape: Tuple) -> Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting."""
        B = t.shape[0]
        out = a.gather(0, t)
        return out.view(B, *([1] * (len(shape) - 1)))
