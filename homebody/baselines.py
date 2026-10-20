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


def command(env, wheels=None, lift=None, arm=None, servos=None):
    """Full action. Unspecified parts hold still: wheels brake to zero speed (as the hub firmware does),
    lift and arm hold their position, servos hold stow."""
    a = np.zeros(S.N_ACTIONS)
    a[[A["wheel_l"], A["wheel_r"]]] = wheel_duty(env, 0.0, 0.0, kv=1.5) if wheels is None else wheels
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

def move_grip(env, target, servos_fn, seconds, lift_gain=10.0, rise_first=False, drive=False):
    """Lift + telescope so the grip centre reaches ``target`` (world). ``rise_first`` changes height
    before reaching out, so the arm doesn't sweep under things on the way. ``drive`` also creeps the
    base forward or back to close the along-the-robot error, which the telescope can't."""
    tr, ext, _ = cartesian_goal(env, target)
    if rise_first:
        run(env, seconds, lambda e: command(e, lift=lift_cmd(e, tr), servos=servos_fn(e)))
    if not drive:
        return run(env, seconds, lambda e: command(e, lift=lift_cmd(e, tr, lift_gain), arm=arm_cmd(e, ext), servos=servos_fn(e)))

    def f(e):
        fwd = to_base(e, target)[0] - to_base(e, grip_point(e))[0]
        return command(e, wheels=wheel_duty(e, float(np.clip(1.5 * fwd, -0.06, 0.06)), 0.0, kv=1.5),
                       lift=lift_cmd(e, tr, lift_gain), arm=arm_cmd(e, ext), servos=servos_fn(e))
    return run(env, seconds, f)


DROP_IN_TOTE = STOW_Q.copy()
DROP_IN_TOTE[0], DROP_IN_TOTE[1] = math.radians(120), math.radians(90)
DROP_IN_TOTE[2] = math.radians(90)          # roll so the jaws open sideways: otherwise the item rests on the lower jaw


FINGERTIP_BELOW_GRIP = 0.0225     # fingertips sit this far below the grip-centre site


def jaw_yaw(env, axis_world):
    """Wrist yaw (gripper pointing down) that makes the jaws close along ``axis_world``.
    The jaws are symmetric, so the two solutions 180 deg apart are equivalent: take the one nearer 0."""
    u = to_base(env, env.data.xpos[env.robot] + np.asarray(axis_world, float))
    yaw = math.atan2(-u[0], u[1])
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def grasp_plan(env, body):
    """Where to put the grip centre and which way to close the jaws for the item ``body``.
    Returns (grip target in world, jaw axis in world, rise first?)."""
    from . import items as I
    d = env.data
    k = env.items.body.index(body)
    it = I.ITEMS[k]
    p = d.xpos[body].copy()
    R = d.xmat[body].reshape(3, 3)
    if it.shape == "cloth":
        rows, cols = it.grid
        axis = R[:, 0] if rows <= cols else R[:, 1]        # close across one patch, where it has no neighbour
        return np.array([p[0], p[1], FINGERTIP_BELOW_GRIP + 0.001]), axis, False
    if it.shape == "hollow" and R[2, 2] > 0.8:              # upright and open-topped: pinch a long wall's rim
        hx, hy, hz = it.size
        walls = [p + R @ np.array([0, s * (hy - 0.003), 0]) for s in (1, -1)]
        wall = min(walls, key=lambda w: abs(to_base(env, w)[1]))   # the wall nearer the mast: less reach
        rim = p[2] + hz
        return np.array([wall[0], wall[1], rim - 0.04 + FINGERTIP_BELOW_GRIP]), R[:, 1], True
    axis = None
    if it.geom != "cylinder":                               # close across the narrowest side
        hx, hy = it.size[0], it.size[1]
        axis = R[:, 0] if hx < hy else R[:, 1]
    return np.array([p[0], p[1], max(p[2], 0.024)]), axis, False


SAFE_Z = 0.30          # grip-centre height that clears the chassis, wheels and tote when reaching out


def pick_up(env, body):
    """Grasp the item ``body`` (beside the robot, within the telescope's reach) and lift it clear.
    Rises before reaching (tucked in, the low gripper scrapes the chassis and the right wheel), turns
    the wrist out there, creeps the base to centre the jaws, then descends and closes.
    Returns the servo targets that keep holding it."""
    target, axis, tall = grasp_plan(env, body)
    q_reach = STOW_Q.copy()
    q_reach[3] = 0.055
    q_open = q_reach.copy()
    if axis is not None:
        q_open[0] = jaw_yaw(env, axis)
    above = target + [0, 0, 0.12]
    if tall:                                                # clear the item's top before reaching over it
        above[2] = max(above[2], env.data.xpos[body][2] + 0.25)
    high = above.copy()
    high[2] = max(high[2], SAFE_Z)
    move_grip(env, high, lambda e: servo_duty(e, q_reach), 4, rise_first=True)
    run(env, 1.0, lambda e: command(e, servos=servo_duty(e, q_open)))
    move_grip(env, above, lambda e: servo_duty(e, q_open), 2.5, drive=True)   # turning shifts the grip: re-aim
    move_grip(env, target, lambda e: servo_duty(e, q_open), 4, lift_gain=6.0)
    tr, _, _ = cartesian_goal(env, target)                   # settle the last centimetre, then hold there
    hold = lambda e: lift_cmd(e, tr, 25.0)
    run(env, 1.0, lambda e: command(e, lift=hold(e), servos=servo_duty(e, q_open)))
    run(env, 1.0, lambda e: command(e, lift=hold(e), servos=servo_duty(e, q_open, grip=1)))
    run(env, 4, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q_open, grip=1)))
    return q_open


def weigh(env, q, tilt_deg=25.0):
    """Tip the wrist a little and read the wrist-pitch servo's load (what the servo reports over the
    bus, as a fraction of stall). Returns kg held, from the change against the empty-hand estimate."""
    i = S.SERVO_NAMES.index("wrist_pitch")
    j = S.DUTY_NAMES.index("wrist_pitch")
    load = lambda e: float(e.data.actuator_force[e.duty_act[j]] / e.stall[j])      # = obs "load.wrist_pitch"
    q_t = q.copy()
    q_t[1] = math.radians(tilt_deg)
    vals = []
    run(env, 1.2, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q_t, grip=1)))
    for _ in range(10):
        run(env, 0.05, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q_t, grip=1)))
        vals.append(load(env))
    run(env, 1.0, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q, grip=1)))
    torque = abs(np.mean(vals)) * S.SERVOS[i].motor.stall_torque
    lever = 0.035 + S.GEOM.finger_length * 0.75 + 0.06    # pitch axis to a held can's centre, roughly
    raw = torque / (9.81 * lever * math.sin(math.radians(tilt_deg)))
    return max((raw - WEIGH_ZERO_KG) / WEIGH_SCALE, 0.0)


# Calibration (sim, 2026-10-19): empty hand reads 0.122 "raw kg", a 15 g can 0.137, a full 370 g can
# 0.462. On hardware, redo this with a known weight: the same two numbers.
WEIGH_ZERO_KG, WEIGH_SCALE = 0.122, 0.915
LIQUID_KG = 0.05                   # an empty can is 15 g: anything well above that still has drink in it


def _is_can(env, body):
    from . import items as I
    return I.ITEMS[env.items.body.index(body)].liquid > 0


def pick_into_tote(env, body, check_liquid=True):
    """Pick a small item and drop it in the deck tote. Big ones (paper bags, the box) don't fit the
    tote: use ``pick_and_carry``. A can that weighs more than an empty one still has drink in it: it
    stays upright in the hand instead (returns early with ``env.info``; the caller decides where it goes)."""
    q = pick_up(env, body)
    if check_liquid and _is_can(env, body):
        env.last_weight = weigh(env, q)
        if env.last_weight > LIQUID_KG:
            env.holding = q
            return run(env, 0.5, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q, grip=1)))
    ramp = servo_ramp(env, q, DROP_IN_TOTE, 2.5, grip=1)
    run(env, 4, lambda e: command(e, lift=lift_cmd(e, 0.65), arm=arm_cmd(e, 0.0), servos=ramp(e)))
    q_rel = DROP_IN_TOTE.copy()
    q_rel[3] = 0.055
    return run(env, 2, lambda e: command(e, lift=lift_cmd(e, 0.65), servos=servo_duty(e, q_rel)))


def pick_and_carry(env, body, v=0.25, seconds=4.0):
    """Pick a big item, tuck the arm in and drive off holding it, as you would carry a paper bag to
    recycling. Returns the last info; check where the item is afterwards."""
    return carry(env, pick_up(env, body), v, seconds)


def carry(env, q, v=0.25, seconds=4.0):
    """Holding something with servo targets ``q``: tuck the arm in, keep the wrist as it is (a drink
    stays upright), and drive off."""
    run(env, 3, lambda e: command(e, lift=lift_cmd(e, 0.65), arm=arm_cmd(e, 0.08), servos=servo_duty(e, q, grip=1)))
    return run(env, seconds, lambda e: command(e, wheels=wheel_duty(e, v, 0.0), lift=lift_cmd(e, 0.65),
                                               arm=arm_cmd(e, 0.08), servos=servo_duty(e, q, grip=1)))


def _lowest_z(env, body):
    """Lowest point of the item's geoms (axis-aligned bounding approximation)."""
    m, d = env.model, env.data
    root = m.body_rootid[body]
    z = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] == root:
            R = d.geom_xmat[g].reshape(3, 3)
            z.append(d.geom_xpos[g][2] - np.abs(R[2]) @ m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                     else d.geom_xpos[g][2] - m.geom_rbound[g])
    return min(z)


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
