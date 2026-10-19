"""Hand-written reflexes. Not the goal; they prove the body can do the chores.

They act only through the same 10 motor commands a learned brain uses. Some
read ground truth from the simulator (``env.data``) where a real robot would
use its cameras; those exist for tests and demos only.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import spec as S
from .env import STOW, STOW_LIFT

A = {n: i for i, n in enumerate(S.ACTION_NAMES)}
SERVO_ACT = np.array([A[n] for n in S.SERVO_NAMES])
STOW_Q = np.array([STOW[n] for n in S.SERVO_NAMES])


_LO = np.array([math.radians(j.range[0]) if not j.slide else j.range[0] for j in S.SERVOS])
_HI = np.array([math.radians(j.range[1]) if not j.slide else j.range[1] for j in S.SERVOS])


def servo_duty(env, target, kp=2.0, kd=0.08, grip=None):
    """Servo commands that move the 6 bus servos to ``target`` (rad, gripper in m).

    Position mode: the command is the target itself, scaled to -1..1 over the joint range.
    PWM mode: a joint-space PD on encoder readings. ``grip`` squeezes the gripper shut (position
    mode: command fully closed and let the servo's torque limit set the grip force)."""
    target = np.array(target, float)
    g = S.SERVO_NAMES.index("gripper")
    if env.cfg.servo_mode == "position":
        u = (target - (_LO + _HI) / 2) / ((_HI - _LO) / 2)
        if grip is not None:
            u[g] = -1.0
        return np.clip(u, -1, 1)
    d = env.data
    q, dq = d.qpos[env.servo_q], d.qvel[env.servo_v]
    u = np.clip(kp * (target - q) - kd * dq, -1, 1)
    if grip is not None:
        u[g] = -abs(grip)
    return u


def servo_ramp(env, start, goal, seconds, grip=None):
    """A function of env that eases the servo targets from ``start`` to ``goal`` over ``seconds``
    (fast servo moves fling whatever the gripper holds)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    t0 = env.t

    def f(e):
        x = min(max((e.t - t0) / seconds, 0.0), 1.0)
        x = x * x * (3 - 2 * x)
        return servo_duty(e, start + (goal - start) * x, grip=grip)
    return f


def lift_cmd(env, travel, gain=10.0):
    """Speed command toward a lift travel (m from the bottom)."""
    return float(np.clip(gain * (travel - env.data.qpos[env.lift_q]), -1, 1))


def arm_cmd(env, ext, gain=10.0):
    """Speed command toward a telescope extension (m)."""
    return float(np.clip(gain * (ext - env.data.qpos[env.arm_q].sum()), -1, 1))


def wheel_duty(env, v, w, kv=0.3):
    """Feed-forward + P on hub speeds for body speed v (m/s) and turn rate w (rad/s)."""
    g = S.GEOM
    target = np.array([v - w * g.wheel_track / 2, v + w * g.wheel_track / 2]) / g.wheel_radius
    vb = max(env.battery.v, 1.0)
    ff = target / S.HUB_MOTOR.no_load_speed * (S.HUB_MOTOR.v_rated / vb)
    return np.clip(ff + kv * (target - env.data.qvel[env.wheel_v]) / 10.0, -1, 1)


def command(env, wheels=(0.0, 0.0), lift=None, arm=None, servos=None):
    """Full action. Unspecified parts hold still: lift and arm hold their position, servos hold stow."""
    a = np.zeros(S.N_ACTIONS)
    a[[A["wheel_l"], A["wheel_r"]]] = wheels
    a[A["lift"]] = 0.0 if lift is None else lift
    a[A["arm"]] = 0.0 if arm is None else arm
    a[SERVO_ACT] = servo_duty(env, STOW_Q) if servos is None else servos
    return a


def grip_point(env):
    s = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "grip_center")
    return env.data.site_xpos[s].copy()


def to_base(env, p):
    """World point -> robot base frame."""
    d = env.data
    R = d.xmat[env.robot].reshape(3, 3)
    return R.T @ (np.asarray(p) - d.xpos[env.robot])


def cartesian_goal(env, target_world):
    """Lift travel and telescope extension that put the grip centre over/at ``target_world`` with the
    gripper pointing down (wrist at stow), plus the forward offset the base still has to drive (m).
    The design is Cartesian: lift = z, telescope = -y in the base frame, driving = x."""
    d = env.data
    here = to_base(env, grip_point(env))
    want = to_base(env, target_world)
    travel = d.qpos[env.lift_q] + (want[2] - here[2])
    ext = d.qpos[env.arm_q].sum() - (want[1] - here[1])
    return float(travel), float(ext), float(want[0] - here[0])


def run(env, seconds, fn):
    """Step ``fn(env) -> action`` for ``seconds``; returns the last info."""
    info = env.info
    for _ in range(int(seconds * S.CONTROL_HZ)):
        _, _, term, _, info = env.step(fn(env))
        if term:
            break
    return info


# ---------------------------------------------------------------------------
# scripted chores (sim-assisted: targets come from ground truth, not cameras)
# ---------------------------------------------------------------------------

def move_grip(env, target, servos_fn, seconds, lift_gain=10.0, rise_first=False):
    """Lift + telescope so the grip centre reaches ``target`` (world). ``rise_first`` changes height
    before reaching out, so the arm doesn't sweep under things on the way."""
    tr, ext, _ = cartesian_goal(env, target)
    if rise_first:
        run(env, seconds, lambda e: command(e, lift=lift_cmd(e, tr), servos=servos_fn(e)))
    return run(env, seconds, lambda e: command(e, lift=lift_cmd(e, tr, lift_gain), arm=arm_cmd(e, ext), servos=servos_fn(e)))


DROP_IN_TOTE = STOW_Q.copy()
DROP_IN_TOTE[0], DROP_IN_TOTE[1] = math.radians(120), math.radians(90)
DROP_IN_TOTE[2] = math.radians(90)          # roll so the jaws open sideways: otherwise the item rests on the lower jaw


def pick_into_tote(env, body):
    """Pick the item ``body`` (beside the robot, within the telescope's reach) and drop it in the deck tote."""
    q_open = STOW_Q.copy()
    q_open[3] = 0.055
    p = env.data.xpos[body].copy()
    move_grip(env, p + [0, 0, 0.12], lambda e: servo_duty(e, q_open), 4)
    move_grip(env, np.array([p[0], p[1], max(p[2], 0.024)]), lambda e: servo_duty(e, q_open), 4, lift_gain=6.0)
    run(env, 0.7, lambda e: command(e, servos=servo_duty(e, q_open)))
    run(env, 1.0, lambda e: command(e, servos=servo_duty(e, q_open, grip=1)))
    run(env, 4, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, STOW_Q, grip=1)))
    ramp = servo_ramp(env, STOW_Q, DROP_IN_TOTE, 2.5, grip=1)
    run(env, 4, lambda e: command(e, lift=lift_cmd(e, 0.65), arm=arm_cmd(e, 0.0), servos=ramp(e)))
    q_rel = DROP_IN_TOTE.copy()
    q_rel[3] = 0.055
    return run(env, 2, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q_rel)))


def open_lever_door(env, lever_geom="bedroom_door_lever_b"):
    """Press a lever handle down with the fingertips, then push the door with the palm.
    Needs the robot parked beside the door with the lever within reach. Returns the last info."""
    m, d = env.model, env.data
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lever_geom)
    lever = d.geom_xpos[g].copy()
    toward_end = -d.geom_xmat[g].reshape(3, 3)[:, 0]          # the lever's free end, away from its pivot
    q_fist = STOW_Q.copy()
    q_fist[0] = math.pi / 2                                   # palm parallel to the door
    above = lever + 0.04 * toward_end + np.array([0, 0, 0.035])   # fingertips just over the lever's end
    sv = lambda e: servo_duty(e, q_fist)
    move_grip(env, above + [0, 0, 0.12], sv, 9, rise_first=True)
    move_grip(env, above, sv, 3, lift_gain=4.0)
    move_grip(env, above - [0, 0, 0.11], sv, 2, lift_gain=6.0)  # press: unlatch
    hold = grip_point(env)
    hold[2] += 0.01
    to_base_y = d.xmat[env.robot].reshape(3, 3)[:, 1]
    return move_grip(env, hold - 0.40 * to_base_y, sv, 5, lift_gain=6.0)   # push with the palm


def pull_knob_door(env, knob_geom="laundry_door_knob_a", pull=0.2):
    """Grip a door knob from above and pull it toward the robot (draws the telescope in), breaking
    the magnetic catch. Needs the robot parked beside the door, clear of its swing."""
    m, d = env.model, env.data
    knob = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, knob_geom)].copy()
    q_open = STOW_Q.copy()
    q_open[0] = math.pi / 2                  # jaws open along the knob's sides
    q_open[3] = 0.055
    move_grip(env, knob + [0, 0, 0.12], lambda e: servo_duty(e, q_open), 9, rise_first=True)
    move_grip(env, knob - [0, 0, 0.02], lambda e: servo_duty(e, q_open), 4, lift_gain=6.0)   # fingers wrap the knob
    run(env, 1.0, lambda e: command(e, servos=servo_duty(e, q_open, grip=1)))
    _, ext, _ = cartesian_goal(env, knob)
    return run(env, 4, lambda e: command(e, arm=arm_cmd(e, ext - pull), servos=servo_duty(e, q_open, grip=1)))
