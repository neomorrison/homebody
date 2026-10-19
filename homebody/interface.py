"""One brain, two bodies.

A trained brain only ever talks to a ``Body``: it reads the observation vector
(layout in ``HomebodyEnv.obs_names``, plus camera images) and writes the 10
commands in ``spec.ACTION_NAMES``. ``SimBody`` is the simulator; ``RobotBody``
is the physical robot. Swapping one for the other is the "implant the brain"
step. On the real robot the Pi runs this loop and forwards observations to the
brain on the desktop GPU over Wi-Fi.

``RobotBody`` is a skeleton: the drivers below are the ones the parts list
implies, and each TODO names what to measure or verify on the bench first.
"""
from __future__ import annotations

import struct
import time
from abc import ABC, abstractmethod

import numpy as np

from . import spec as S


class Body(ABC):
    control_hz = S.CONTROL_HZ

    @abstractmethod
    def observe(self):
        """Observation in ``HomebodyEnv`` format."""

    @abstractmethod
    def act(self, command: np.ndarray) -> None:
        """10 numbers in [-1, 1], ``spec.ACTION_NAMES`` order."""

    def run(self, brain, seconds: float = float("inf")):
        period = 1.0 / self.control_hz
        t_end = time.monotonic() + seconds
        nxt = time.monotonic()
        while time.monotonic() < t_end:
            self.act(np.clip(brain(self.observe()), -1.0, 1.0))
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))


class SimBody(Body):
    def __init__(self, env):
        self.env = env
        self.obs, _ = env.reset()

    def observe(self):
        return self.obs

    def act(self, command):
        self.obs, *_ = self.env.step(command)


# ----------------------------------------------------------------------------
# hardware
# ----------------------------------------------------------------------------

# Feetech STS bus (half duplex TTL, 1 Mbaud). VERIFY against the servos' memory table.
STS_REG = {"mode": 33, "torque_enable": 40, "goal_position": 42, "torque_limit": 48,
           "present_position": 56, "present_speed": 58, "present_load": 60, "present_voltage": 62,
           "present_temp": 63}
INST_WRITE, INST_SYNC_WRITE = 0x03, 0x83


def _packet(sid: int, inst: int, params: bytes) -> bytes:
    body = bytes([sid, len(params) + 2, inst]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


class RobotBody(Body):
    """The physical robot.

    * hub motors: hoverboard mainboard on hoverboard-firmware-hack-FOC, UART "steer/speed" or voltage
      commands at 50+ Hz; TODO: pick the firmware's voltage mode so a command means the same as the
      sim's duty cycle, and read back wheel speeds.
    * lift and telescope: STEP/DIR to the stepper drivers from the Pi's GPIO (or a small MCU); a speed
      command becomes a step rate; position = step count; home on the limit switches at boot.
    * wrist, gripper, head: Feetech bus, position mode, sync-write goal positions; gripper torque limit
      set to ``spec.WRIST[3].torque_limit``.
    * sensors: RPLIDAR C1 (rplidar SDK), USB cameras (OpenCV), BNO085 (I2C), INA226 (I2C), cliff
      sensors and bumper (GPIO), beacon photodiodes (ADC), charge flag (GPIO from the charge board).
    """

    def __init__(self, servo_port: str, hub_port: str, servo_ids=(1, 2, 3, 4, 5, 6)):
        import serial  # pyserial
        self.servo = serial.Serial(servo_port, 1_000_000, timeout=0.005)
        self.hub = serial.Serial(hub_port, 115200, timeout=0.005)
        self.servo_ids = list(servo_ids)

    def _servo_targets(self, commands):
        """-1..1 per servo -> 12-bit goal positions. TODO: per-servo zero offsets and directions
        from a calibration run."""
        out = []
        for j, u in zip(S.SERVOS, commands):
            lo, hi = j.range
            x = (lo + hi) / 2 + u * (hi - lo) / 2
            turns = (x / 360.0) if not j.slide else (x * j.gear / (2 * np.pi))
            out.append(int(round(2048 + turns * 4096)) & 0x0FFF)
        return out

    def act(self, command):
        a = dict(zip(S.ACTION_NAMES, command))
        params = bytes([STS_REG["goal_position"], 2])
        for sid, pos in zip(self.servo_ids, self._servo_targets([a[n] for n in S.SERVO_NAMES])):
            params += bytes([sid]) + struct.pack("<H", pos)
        self.servo.write(_packet(0xFE, INST_SYNC_WRITE, params))
        # TODO: hub motor frame for the FOC firmware (wheel_l, wheel_r), and step rates for lift/arm.
        raise NotImplementedError("hub motor and stepper output are not wired yet")

    def observe(self):
        raise NotImplementedError(
            "TODO once hardware exists: read every sensor listed in the class docstring and pack them in "
            "HomebodyEnv.obs_names order with the same scaling as HomebodyEnv._obs.")
