"""Minimal adversarial-evaluation adapters (PROTOTYPE — under discussion in #1486).

These are thin gym.Wrapper implementations that perturb observations or actions
with a seeded perturbation function, so that an existing evaluation loop (e.g.
``examples/baselines/*/evaluate.py``) can be reused unchanged for robustness /
adversarial evaluation. Both wrappers are strict no-ops when no perturbation
function is provided.
"""

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.structs.types import Array


class ObservationPerturbationWrapper(gym.ObservationWrapper):
    """Applies a seeded perturbation to observations before they reach the policy.

    Args:
        env: the environment to wrap.
        fn: callable ``(obs, rng) -> obs``. ``None`` means a no-op.
        seed: seed for the perturbation RNG.
    """

    def __init__(
        self,
        env,
        fn: Optional[Callable[[Array, np.random.Generator], Array]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.fn = fn
        self.rng = np.random.default_rng(seed)

    def observation(self, observation):
        if self.fn is None:
            return observation
        return self.fn(observation, self.rng)


class ActionPerturbationWrapper(gym.ActionWrapper):
    """Applies a seeded perturbation to actions before they reach the environment.

    The perturbed action is clipped to the environment action space, so a noisy
    policy can never produce out-of-bounds commands.

    Args:
        env: the environment to wrap.
        fn: callable ``(action, rng) -> action``. ``None`` means a no-op.
        seed: seed for the perturbation RNG.
    """

    def __init__(
        self,
        env,
        fn: Optional[Callable[[Array, np.random.Generator], Array]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.fn = fn
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        if self.fn is None:
            return action
        perturbed = self.fn(action, self.rng)
        low = torch.as_tensor(self.env.action_space.low, dtype=perturbed.dtype)
        high = torch.as_tensor(self.env.action_space.high, dtype=perturbed.dtype)
        return torch.clip(perturbed, low, high)


def gaussian_obs_noise(std: float) -> Callable[[Array, np.random.Generator], Array]:
    """Returns a perturbation function that adds Gaussian noise to state observations."""

    def fn(obs, rng):
        obs = obs.copy()
        for key in ("agent", "extra"):
            if key in obs:
                for sub in obs[key]:
                    obs[key][sub] = obs[key][sub] + rng.normal(
                        0, std, size=obs[key][sub].shape
                    )
        return obs

    return fn


def gaussian_action_noise(std: float) -> Callable[[Array, np.random.Generator], Array]:
    """Returns a perturbation function that adds Gaussian noise to actions."""

    def fn(action, rng):
        return action + rng.normal(0, std, size=action.shape)

    return fn
