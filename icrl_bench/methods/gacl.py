"""GACL, "GAIL-Constraint" (baseline in Malik et al. 2021 and Liu et al.
2023): a GAIL discriminator D(features) in (0, 1), expert = 1, nominal = 0,
whose log(D + eps) is added to the agent's reward. Port of Malik's
GailDiscriminator (github.com/shehryar-malik/icrl, commit f28214e,
icrl/gail_utils.py) as called by GailCallback (one train(1, ...) per
rollout). It is updated after every rollout of one training run, through
PPO.train's reward_hook (docs/decisions.md, 2026-11-01). The network part is
a copy of binary_classifier.py; deviations in docs/methods.md."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class GACLSettings:
    """Defaults: Malik's CLI defaults (icrl/gail.py argparse). Per-env values
    live in configs."""
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4  # constant: his learning rate never anneals
    eps: float = 1e-5  # the extra reward is log(D + eps)
    # Feature columns the network sees, in order; None = all. Malik's
    # discriminator has the same column bug as his constraint network
    # (docs/methods.md, "Bugs in the official code" (2)); set it only to
    # match his code.
    input_columns: tuple[int, ...] | None = None
    seed: int = 0
    device: str = "cpu"


class GACL:
    """Learns D(features) in (0, 1) by labelling expert steps 1 and nominal
    steps 0. The agent's extra reward is log(D + eps); the cost is 1 - D."""

    def __init__(self, n_features, settings=None):
        self.settings = settings or GACLSettings()
        s = self.settings
        # Malik's discriminator starts from the same weights in every run
        # (bug (3), as in his ICRL code); we seed per run instead.
        torch.manual_seed(s.seed)
        # GailDiscriminator._build: SB3 create_mlp (Linear + ReLU per hidden
        # layer, PyTorch default init) then a sigmoid.
        in_dim = n_features if s.input_columns is None else len(s.input_columns)
        layers = []
        for size in s.hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers).to(s.device)
        # eps=1e-5 avoids NaNs in Adam (GailDiscriminator.__init__).
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=s.learning_rate, eps=1e-5)

    def _inputs(self, features):
        """Network input: a (steps, features) array as a float tensor,
        keeping only input_columns if set (select_appropriate_dims)."""
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.settings.device)
        if self.settings.input_columns is not None:
            x = x[:, list(self.settings.input_columns)]
        return x

    def _d(self, features):
        """D for a (steps, features) array, as a float32 numpy array."""
        with torch.no_grad():
            return self.net(self._inputs(features)).squeeze(-1).cpu().numpy()

    def update(self, expert, nominal):
        """One full-batch Adam step on BCE(D_nominal, 0) + BCE(D_expert, 1)
        (GailDiscriminator.train with iterations=1, batch_size=None).
        expert, nominal: one (steps, features) array per trajectory (or per
        rollout), joined in order. Returns Malik's discriminator metrics."""
        expert_x = self._inputs(np.concatenate(expert))
        nominal_x = self._inputs(np.concatenate(nominal))
        # Quirk A (docs/methods.md §3): his batch indices run over
        # min(n_nominal, n_expert), so the bigger pile only contributes its
        # first rows. His shuffle is skipped: one full batch, same mean.
        n = min(len(expert_x), len(nominal_x))
        expert_x, nominal_x = expert_x[:n], nominal_x[:n]

        d_nominal = self.net(nominal_x).squeeze(-1)
        d_expert = self.net(expert_x).squeeze(-1)
        # nn.BCELoss: mean over steps; each log is clamped at -100.
        nominal_loss = nn.functional.binary_cross_entropy(d_nominal, torch.zeros_like(d_nominal))
        expert_loss = nn.functional.binary_cross_entropy(d_expert, torch.ones_like(d_expert))
        loss = nominal_loss + expert_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Values from before the step, as in his train.
        return {
            "disc_loss": loss.item(),
            "expert_loss": expert_loss.item(),
            "nominal_loss": nominal_loss.item(),
            "mean_nominal_preds": d_nominal.mean().item(),
            "mean_expert_preds": d_expert.mean().item(),
        }

    def reward(self, features):
        """Per-step extra reward log(D + eps) for a (steps, features) array
        (GailDiscriminator.reward_function; numpy float32, as his)."""
        return np.log(self._d(features) + np.float32(self.settings.eps))

    def cost(self, features):
        """Per-step cost 1 - D (Liu's GailDiscriminator.cost_function)."""
        return 1 - self._d(features)
