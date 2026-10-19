"""One brain, two bodies.

A trained brain only ever talks to a ``Body``: it reads the observation vector
(layout in ``HomebodyEnv.obs_names``) and writes 20 duty cycles. ``SimBody``
is the simulator; ``FeetechBody`` is the physical robot over the servo bus.
Swapping one for the other is the whole "implant the brain" step.

The hardware side is a skeleton. Register addresses below follow the
community-documented Feetech STS memory table and must be checked against the
datasheet of the servos you buy before the robot is powered with a brain
attached (a wrong mode byte turns PWM commands into position commands).
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
    def observe(self) -> np.ndarray:
        """Observation vector in ``HomebodyEnv.obs_names`` order."""

    @abstractmethod
    def act(self, duty: np.ndarray) -> None:
        """20 PWM duty cycles in [-1, 1], ``spec.JOINT_NAMES`` order."""

    def run(self, brain, seconds: float = float("inf")):
        """Fixed-rate loop: observe -> brain -> act."""
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

    def act(self, duty):
        self.obs, *_ = self.env.step(duty)


# ----------------------------------------------------------------------------
# Feetech STS bus (half-duplex TTL, 1 Mbaud) - skeleton
# ----------------------------------------------------------------------------

STS_REG = {                 # VERIFY against your servo's memory table
    "mode": 33,             # 0 position, 1 speed, 2 PWM open-loop (STS); HLS numbering differs
    "torque_enable": 40,
    "goal_pwm": 44,         # 2 bytes, 0..1000 (0.1 %), bit 10 = direction, in PWM mode
    "present_position": 56, # 2 bytes, 0..4095
    "present_speed": 58,    # 2 bytes, bit 15 = sign, steps/s
    "present_load": 60,     # 2 bytes, bit 10 = sign, 0.1 %
    "present_voltage": 62,  # 1 byte, 0.1 V
    "present_temp": 63,     # 1 byte, C
}
INST_READ, INST_WRITE, INST_SYNC_WRITE, INST_SYNC_READ = 0x02, 0x03, 0x83, 0x82


def _packet(sid: int, inst: int, params: bytes) -> bytes:
    body = bytes([sid, len(params) + 2, inst]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


class FeetechBody(Body):
    """Physical robot. Needs pyserial and the on-board sensor drivers (IMU, FSR ADCs,
    INA226, IR receivers), which are left as TODOs until the hardware exists."""

    def __init__(self, port: str, servo_ids: list[int] | None = None, offsets=None, signs=None):
        import serial  # pyserial
        self.ser = serial.Serial(port, 1_000_000, timeout=0.005)
        self.ids = servo_ids or list(range(1, S.N_JOINTS + 1))
        self.offsets = np.zeros(S.N_JOINTS) if offsets is None else np.asarray(offsets)   # calibration
        self.signs = np.ones(S.N_JOINTS) if signs is None else np.asarray(signs)
        self.prev = np.zeros(S.N_JOINTS)

    def enable_pwm_mode(self):
        for sid in self.ids:
            self._write(sid, STS_REG["torque_enable"], b"\x00")
            self._write(sid, STS_REG["mode"], b"\x02")
            self._write(sid, STS_REG["torque_enable"], b"\x01")

    def _write(self, sid, addr, data: bytes):
        self.ser.write(_packet(sid, INST_WRITE, bytes([addr]) + data))

    def act(self, duty):
        duty = np.clip(duty * self.signs, -1, 1)
        params = bytes([STS_REG["goal_pwm"], 2])
        for sid, u in zip(self.ids, duty):
            mag = int(round(abs(u) * 1000))
            params += bytes([sid]) + struct.pack("<H", mag | (0x400 if u < 0 else 0))
        self.ser.write(_packet(0xFE, INST_SYNC_WRITE, params))
        self.prev = duty

    def observe(self):
        raise NotImplementedError(
            "TODO once hardware exists: sync-read position/speed/load/temp from the servos, "
            "read BNO085 (gravity, gyro, accel), 2x ADS1115 (FSRs), INA226 (V, I), the charge "
            "flag GPIO and the 4 IR receivers, then pack them in HomebodyEnv.obs_names order "
            "with the same scaling as HomebodyEnv._obs."
        )
