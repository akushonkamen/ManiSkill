"""PROTOTYPE for #1486: adversarial-evaluation adapter smoke test on CPU.

Evaluates a simple scripted PD policy on PushCube-v1 (pd_ee_delta_pos,
state observations) under zero / action-noise / observation-noise perturbations
and reports success rates.
"""

import numpy as np
import torch
import gymnasium as gym

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.wrappers.adversarial import (
    ActionPerturbationWrapper,
    ObservationPerturbationWrapper,
    gaussian_action_noise,
    gaussian_obs_noise,
)


def make_env(fn=None, seed=None):
    env = PushCubeEnv(obs_mode="state_dict", control_mode="pd_ee_delta_pos", render_mode=None)
    env = gym.wrappers.TimeLimit(env, 400)
    if fn is not None:
        if fn[0] == "action":
            env = ActionPerturbationWrapper(env, gaussian_action_noise(fn[1]), seed=seed)
        else:
            env = ObservationPerturbationWrapper(env, gaussian_obs_noise(fn[1]), seed=seed)
    return env


class ScriptedPolicy:
    """Phase-based push policy with hysteresis so phases never oscillate."""

    def __init__(self):
        self.phase = 0

    def __call__(self, obs):
        tcp = obs["extra"]["tcp_pose"][0][:3].numpy()
        cube = obs["extra"]["obj_pose"][0][:3].numpy()
        goal = obs["extra"]["goal_pos"][0].numpy()
        dir = (cube - goal) / np.linalg.norm(cube - goal)
        behind_hi = cube + 0.09 * dir + np.array([0, 0, 0.10])
        behind = cube + 0.05 * dir + np.array([0, 0, 0.01])
        push = np.array([goal[0], goal[1], behind[2]])
        targets = [behind_hi, behind, push]
        # advance phase only when the current waypoint is reached; never regress
        while (
            self.phase < len(targets) - 1
            and np.linalg.norm(tcp - targets[self.phase]) < 0.02
        ):
            self.phase += 1
        target = targets[self.phase]
        gripper = 1.0 if self.phase < 2 else -1.0  # close fingers before pushing
        delta = np.clip(4.0 * (target - tcp), -0.04, 0.04)
        return torch.tensor([delta[0], delta[1], delta[2], gripper]).unsqueeze(0)


scripted_policy = ScriptedPolicy()


def run(label, fn, episodes=20, seed=0):
    env = make_env(fn, seed=seed)
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy = ScriptedPolicy()
        for _ in range(400):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated:
                break
        successes += int(info["success"])
    env.close()
    print(f"{label}: success {successes}/{episodes} = {successes / episodes:.2%}")
    return successes / episodes


if __name__ == "__main__":
    run("clean              ", None)
    run("action noise 0.05  ", ("action", 0.05))
    run("action noise 0.10  ", ("action", 0.10))
    run("obs noise 0.01     ", ("obs", 0.01))
    run("obs noise 0.03     ", ("obs", 0.03))
