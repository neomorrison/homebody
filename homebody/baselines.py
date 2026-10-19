"""Hand-written reference brains. Not the goal; they prove the body works.

Each takes an observation vector (the same one a learned brain gets) and
returns 20 duty cycles.
"""
from __future__ import annotations

import numpy as np

from . import spec as S
from .env import DEFAULT_POSE

N = S.N_JOINTS
CROUCH = np.array([DEFAULT_POSE.get(n, 0.0) for n in S.JOINT_NAMES])


def limp(obs):
    return np.zeros(N)


def make_hold(target=CROUCH, kp=6.0, kd=0.15):
    """A spinal reflex: push each joint back toward a posture using only its encoder."""
    def brain(obs):
        q, dq = obs[:N], obs[N:2 * N] * 10.0
        return np.clip(kp * (target - q) - kd * dq, -1.0, 1.0)
    return brain


def make_random(rng=None, scale=0.4, smooth=0.9):
    """Correlated random muscle twitches (motor babbling)."""
    rng = rng or np.random.default_rng()
    state = np.zeros(N)

    def brain(obs):
        nonlocal state
        state = smooth * state + (1 - smooth) * rng.normal(0, scale * 3, N)
        return np.clip(state, -1, 1)
    return brain
