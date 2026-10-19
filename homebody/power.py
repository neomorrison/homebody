"""Battery, servo electrical/thermal model and the dock's charger.

The motors pull current from the pack, the pack's voltage sags under load and
as it empties, and lower voltage means weaker muscles. Below the BMS cutoff the
whole robot goes limp. The only way to get energy back is to stand on the dock.
"""
from __future__ import annotations

import numpy as np

from . import spec as S


class Battery:
    def __init__(self, b: S.BatterySpec = S.BATTERY, capacity_scale: float = 1.0):
        self.b = b
        self.capacity_c = b.capacity_ah * 3600.0 * capacity_scale   # coulombs
        self.soc = 1.0
        self.v = self.ocv()
        self.i = 0.0            # + discharge, - charge (A)
        self.dead = False

    def ocv(self, soc=None) -> float:
        soc = self.soc if soc is None else soc
        return float(np.interp(soc, self.b.ocv_soc, self.b.ocv_v)) * self.b.cells_series

    def reset(self, soc: float):
        self.soc = float(np.clip(soc, 0.0, 1.0))
        self.i = 0.0
        self.v = self.ocv()
        self.dead = False

    def step(self, load_current: float, charge_current: float, dt: float) -> float:
        """Advance dt seconds. Returns energy drawn from the pack (J, negative when charging)."""
        if self.dead and charge_current <= 0:
            self.i, self.v = 0.0, self.ocv()
            return 0.0
        i = load_current - charge_current
        self.i = i
        self.v = self.ocv() - i * self.b.r_internal
        self.soc = float(np.clip(self.soc - i * dt / self.capacity_c, 0.0, 1.0))
        cutoff = self.b.cell_cutoff_v * self.b.cells_series
        if self.v < cutoff or self.soc <= 0.0:
            self.dead = True          # BMS opens; only the dock can wake it
        elif self.dead and charge_current > 0 and self.soc > 0.03:
            self.dead = False
        return self.v * i * dt


class Charger:
    """CC/CV charger in the dock. Power flows only after the feet have settled."""

    def __init__(self, d: S.DockSpec = S.DOCK, b: S.BatterySpec = S.BATTERY):
        self.d, self.b = d, b
        self.contact_time = 0.0

    def reset(self):
        self.contact_time = 0.0

    def current(self, docked: bool, battery: Battery, dt: float) -> float:
        if not docked:
            self.contact_time = 0.0
            return 0.0
        self.contact_time += dt
        if self.contact_time < self.d.contact_settle_s:
            return 0.0
        # CV phase: current limited by the voltage headroom over the internal resistance
        headroom = (self.d.charge_voltage - battery.ocv()) / self.b.r_internal
        return float(np.clip(headroom, 0.0, self.d.charge_current)) * self.d.charge_efficiency


class ServoBank:
    """Electrical and thermal state of all servos (vectorised over joints)."""

    def __init__(self, joints=S.JOINTS, ambient: float = 25.0):
        self.kt = np.array([j.servo.torque_constant for j in joints])
        self.rw = np.array([j.servo.winding_resistance for j in joints])
        self.i0 = np.array([j.servo.no_load_current for j in joints])
        self.wnl = np.array([j.servo.no_load_speed for j in joints])
        self.rth = np.array([j.servo.thermal_resistance for j in joints])
        self.cth = np.array([j.servo.thermal_capacity for j in joints])
        self.cut = np.array([j.servo.cutout_temp for j in joints])
        self.ambient = ambient
        self.temp = np.full(len(joints), ambient)
        self.tripped = np.zeros(len(joints), bool)

    def reset(self, ambient: float | None = None):
        if ambient is not None:
            self.ambient = ambient
        self.temp[:] = self.ambient
        self.tripped[:] = False

    def step(self, torque, duty, omega, dt):
        """Returns battery current drawn by the motors (A)."""
        i_motor = torque / self.kt                                   # winding current, signed
        i_spin = self.i0 * np.minimum(np.abs(omega) / self.wnl, 1.0) * (np.abs(duty) > 0.01)
        # H-bridge: battery current ~= duty * motor current; braking regen is not recovered
        i_batt = np.maximum(duty * i_motor, 0.0) + i_spin
        heat = i_motor ** 2 * self.rw
        self.temp += dt * (heat - (self.temp - self.ambient) / self.rth) / self.cth
        self.tripped = np.where(self.tripped, self.temp > self.cut - 10.0, self.temp > self.cut)
        return float(i_batt.sum())
