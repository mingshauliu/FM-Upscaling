"""
Conditional Normalizing Flow for cosmological parameter inference.

Uses Neural Spline Flows (NSF) or Masked Autoregressive Flows (MAF)
for learning conditional distributions p(cosmo_params | summary_vector).
"""

import torch
import torch.nn as nn
import zuko


class ConditionalFlow(nn.Module):
    """
    Conditional normalizing flow with target normalization.

    Wraps a zuko NSF or MAF flow, handling automatic target
    standardization and providing sampling/log_prob interfaces.

    Args:
        num_params: Number of target parameters (e.g. 2 for Omega_m, sigma_8)
        context_dim: Dimension of conditioning vector from encoder
        hidden_dim: Hidden layer size in flow transforms
        num_transforms: Number of flow transformation layers
        flow_type: "nsf" (Neural Spline Flow) or "maf" (Masked Autoregressive)
        target_mean: Optional pre-computed target mean for normalization
        target_std: Optional pre-computed target std for normalization
    """
    def __init__(
        self,
        num_params=2,
        context_dim=128,
        hidden_dim=128,
        num_transforms=8,
        flow_type="nsf",
        target_mean=None,
        target_std=None,
    ):
        super().__init__()

        self.num_params = num_params
        self.context_dim = context_dim

        if flow_type == "nsf":
            self.flow = zuko.flows.NSF(
                features=num_params,
                context=context_dim,
                transforms=num_transforms,
                bins=8,
                hidden_features=[hidden_dim, hidden_dim],
                randperm=True,
            )
        elif flow_type == "maf":
            self.flow = zuko.flows.MAF(
                features=num_params,
                context=context_dim,
                transforms=num_transforms,
                hidden_features=[hidden_dim, hidden_dim, hidden_dim],
                randperm=True,
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        if target_mean is not None:
            self.register_buffer('target_mean', target_mean)
            self.register_buffer('target_std', target_std)
        else:
            self.register_buffer('target_mean', torch.zeros(num_params))
            self.register_buffer('target_std', torch.ones(num_params))

    def normalize_targets(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_targets(self, y_norm):
        return y_norm * self.target_std + self.target_mean

    def log_prob(self, context, y):
        """
        Compute log probability of targets given context.

        Args:
            context: (B, context_dim)
            y: (B, num_params)
        Returns:
            log_prob: (B,)
        """
        y_norm = self.normalize_targets(y)
        dist = self.flow(context)
        log_prob = dist.log_prob(y_norm)
        log_prob = log_prob - self.target_std.log().sum()
        return log_prob

    def sample(self, context, num_samples=1000):
        """
        Sample from the conditional distribution.

        Args:
            context: (B, context_dim)
            num_samples: samples per context
        Returns:
            samples: (num_samples, B, num_params) in original scale
        """
        dist = self.flow(context)
        samples_norm = dist.sample((num_samples,))
        return self.denormalize_targets(samples_norm)

    def get_posterior_stats(self, context, num_samples=2000):
        """
        Get posterior mean and std from samples.

        Returns:
            mean: (B, num_params)
            std: (B, num_params)
        """
        samples = self.sample(context, num_samples)
        return samples.mean(dim=0), samples.std(dim=0)
