import torch
import torch.nn.functional as F
import math

def mdn_loss(mu, sigma, pi, future):
    """
    mu:     [B, K, T, 2]      (means per Gaussian)
    sigma:  [B, K, T, 2]      (std per Gaussian)
    pi:     [B, K]            (mixture weights)
    future: [B, T, 2]         (ground truth trajectory)
    """
    # Add Gaussian axis to future: [B, 1, T, 2]
    future = future.unsqueeze(1)

    # Ensure strictly positive variance
    var = sigma**2 + 1e-6  # [B, K, T, 2]

    # Log-likelihood of each Gaussian component
    # log N(y | mu, sigma) for each (B, K)
    log_normal = -0.5 * torch.sum(
        ((future - mu) ** 2) / var + torch.log(2 * math.pi * var),
        dim=(2, 3)  # sum over T and (x,y)
    )  # -> [B, K]

    # Add log mixture weights
    log_pi = torch.log(pi + 1e-8)       # [B, K]
    log_mix = torch.logsumexp(log_pi + log_normal, dim=1)  # [B]

    # Negative log-likelihood
    loss = -torch.mean(log_mix)
    return loss


def ADE(pred, gt):
    """
    pred: [B, T, 2]
    gt:   [B, T, 2]
    """
    return torch.mean(torch.linalg.norm(pred - gt, dim=-1))


def FDE(pred, gt):
    """
    pred: [B, T, 2]
    gt:   [B, T, 2]
    """
    return torch.mean(torch.linalg.norm(pred[:, -1] - gt[:, -1], dim=-1))
