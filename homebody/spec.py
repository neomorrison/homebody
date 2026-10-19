"""The robot, as parts you can buy.

Everything physical lives here: link lengths, masses, which servo drives which
joint, the battery, the electronics and the charging dock. The simulator
(``robot_xml``, ``world_xml``, ``env``) and the hardware runtime
(``interface``) both read this file, so the body you train is the body you
build. Change a number here and both sides change together.

Servo figures are reseller-listed values for Feetech bus servos (no official
datasheet was available when this was written). Check them against the parts
you receive: ``scripts/identify_servo.py`` (planned) will fit the motor model
to a real servo on a bench.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


# ----------------------------------------------------------------------------
# actuators ("muscles")
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ServoSpec:
    """A geared DC servo, referenced to the output shaft at the rated voltage.

    The simulator drives it in open-loop PWM mode: the brain sets a duty cycle
    in [-1, 1] and the motor makes torque

        tau = stall_torque * (duty * V_battery / v_rated) - (stall_torque / no_load_speed) * omega

    clipped to +-stall_torque (the current limit). This is a muscle, not a
    position command: nothing holds a pose unless the brain keeps pushing.
    """
    name: str
    stall_torque: float      # N m at v_rated
    no_load_speed: float     # rad/s at v_rated
    stall_current: float     # A at v_rated
    no_load_current: float   # A
    v_rated: float           # V
    mass: float              # kg
    gear_ratio: float
    armature: float          # kg m^2, reflected rotor inertia (gear_ratio^2 * rotor)
    friction: float          # N m, gearbox Coulomb friction
    damping: float           # N m s/rad, gearbox viscous loss
    price_usd: float
    # first-order winding thermal model (overheat cutout like the real firmware)
    thermal_resistance: float = 8.0    # K/W, winding -> air
    thermal_capacity: float = 6.0      # J/K
    cutout_temp: float = 70.0          # C, firmware disables torque above this

    @property
    def winding_resistance(self) -> float:
        return self.v_rated / self.stall_current

    @property
    def torque_constant(self) -> float:
        """N m / A at the output shaft."""
        return self.stall_torque / self.stall_current


KGCM = 9.80665 / 100.0   # kg.cm -> N m

# Feetech STS3250-C002: 12 V, 50 kg.cm peak (16 kg.cm rated), 0.133 s/60deg, 4.2 A stall, 74.5 g, 1:345.
# Buy the C002 SKU: the C001 is listed with different specs at $97-125.
STS3250 = ServoSpec(
    name="Feetech STS3250", stall_torque=50 * KGCM, no_load_speed=(math.pi / 3) / 0.133,
    stall_current=4.2, no_load_current=0.28, v_rated=12.0, mass=0.0745, gear_ratio=345,
    armature=0.012, friction=0.06, damping=0.02, price_usd=74.0,
)

# Feetech STS3215 (12 V version): 30 kg.cm, ~0.222 s/60deg, ~2.7 A stall, 55 g, 1:345
STS3215_12V = ServoSpec(
    name="Feetech STS3215 (12V)", stall_torque=30 * KGCM, no_load_speed=(math.pi / 3) / 0.222,
    stall_current=2.7, no_load_current=0.18, v_rated=12.0, mass=0.055, gear_ratio=345,
    armature=0.008, friction=0.04, damping=0.015, price_usd=22.0,
)


@dataclass(frozen=True)
class Joint:
    name: str
    axis: tuple            # in the parent link frame
    range_deg: tuple       # mechanical limits
    servo: ServoSpec


# Joint order == action order == encoder order. 20 servos.
JOINTS: list[Joint] = []
for side in ("l", "r"):
    s = 1 if side == "l" else -1
    JOINTS += [
        Joint(f"hip_yaw_{side}",     (0, 0, 1), (-40, 40),                                 STS3215_12V),
        Joint(f"hip_roll_{side}",    (1, 0, 0), (-25, 60) if s > 0 else (-60, 25),        STS3250),
        Joint(f"hip_pitch_{side}",   (0, 1, 0), (-120, 45),                               STS3250),
        Joint(f"knee_{side}",        (0, 1, 0), (0, 140),                                 STS3250),
        Joint(f"ankle_pitch_{side}", (0, 1, 0), (-60, 60),                                STS3250),
        Joint(f"ankle_roll_{side}",  (1, 0, 0), (-30, 30),                                STS3215_12V),
    ]
for side in ("l", "r"):
    s = 1 if side == "l" else -1
    JOINTS += [
        Joint(f"shoulder_pitch_{side}", (0, 1, 0), (-180, 60),                            STS3215_12V),
        Joint(f"shoulder_roll_{side}",  (1, 0, 0), (-10, 120) if s > 0 else (-120, 10),  STS3215_12V),
        Joint(f"elbow_{side}",          (0, 1, 0), (-140, 0),                             STS3215_12V),
    ]
JOINTS += [
    Joint("neck_yaw",   (0, 0, 1), (-90, 90), STS3215_12V),
    Joint("neck_pitch", (0, 1, 0), (-40, 60), STS3215_12V),
]
JOINT_NAMES = [j.name for j in JOINTS]
N_JOINTS = len(JOINTS)


# ----------------------------------------------------------------------------
# body geometry (m) and link masses (kg, excluding the servos, which are added
# where they sit). About 0.62 m tall, about 3.3 kg.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    hip_half_width: float = 0.050      # pelvis centre -> hip yaw axis (y)
    hip_drop: float = 0.030            # pelvis centre -> hip pitch axis (z)
    thigh: float = 0.130
    shank: float = 0.130
    ankle_to_sole: float = 0.035
    foot_length: float = 0.120
    foot_width: float = 0.055
    foot_heel: float = 0.040           # ankle axis -> heel
    torso_height: float = 0.170        # pelvis centre -> shoulder axis
    torso_depth: float = 0.080
    torso_width: float = 0.120
    shoulder_half_width: float = 0.085
    upper_arm: float = 0.100
    forearm: float = 0.110
    neck: float = 0.040
    head_radius: float = 0.050

    @property
    def standing_pelvis_height(self) -> float:
        return self.hip_drop + self.thigh + self.shank + self.ankle_to_sole


GEOM = Geometry()

# 3D-printed PETG/PLA brackets + fasteners per link (servo masses added separately)
FRAME_MASS = {
    "pelvis": 0.120, "torso": 0.260, "head": 0.090,
    "hip_yaw_link": 0.020, "hip_roll_link": 0.030, "thigh": 0.060, "shank": 0.055,
    "ankle_link": 0.015, "foot": 0.070,
    "upper_arm_a": 0.015, "upper_arm": 0.035, "forearm": 0.045,
}


# ----------------------------------------------------------------------------
# power: battery, electronics, dock
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BatterySpec:
    """3S1P Li-ion pack (3 x 18650, e.g. Molicel P42A / Samsung 35E) with a BMS."""
    cells_series: int = 3
    capacity_ah: float = 3.5
    r_internal: float = 0.15           # ohm, whole pack incl. BMS and wiring
    cell_cutoff_v: float = 3.0         # BMS low-voltage disconnect (under load)
    cell_full_v: float = 4.2
    mass: float = 0.165
    price_usd: float = 35.0
    # open-circuit voltage per cell vs state of charge (typical NMC 18650)
    ocv_soc: tuple = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    ocv_v: tuple = (3.00, 3.30, 3.45, 3.56, 3.63, 3.70, 3.76, 3.84, 3.93, 4.02, 4.10, 4.20)

    @property
    def nominal_wh(self) -> float:
        return self.capacity_ah * 3.65 * self.cells_series


@dataclass(frozen=True)
class ElectronicsSpec:
    """Always-on load: brain computer, IMU, servo logic boards, IR receivers."""
    computer: str = "Raspberry Pi 5 (8 GB)"
    computer_w: float = 5.0
    servo_logic_w: float = 0.10        # per servo, idle
    sensors_w: float = 0.4
    buck_efficiency: float = 0.90      # 12 V -> 5 V for the computer

    def idle_power(self, n_servos: int) -> float:
        return (self.computer_w + self.sensors_w) / self.buck_efficiency + n_servos * self.servo_logic_w


@dataclass(frozen=True)
class DockSpec:
    """Floor charging pad: two copper plates, one per foot, and an IR beacon.

    The robot charges by standing with one foot on each plate. The soles carry
    copper pads wired through an ideal-diode bridge, so either foot can stand on
    either plate. The dock only energises the plates after it measures the
    robot's sense resistor across them (no live metal otherwise), then runs a
    CC/CV charge.
    """
    base_size: tuple = (0.34, 0.36, 0.008)   # x (depth), y (width), z (thickness), m
    plate_size: tuple = (0.24, 0.14)         # each plate, x, y
    plate_gap: float = 0.030                 # insulating strip between plates
    beacon_height: float = 0.30              # IR LED post at the back of the dock
    charge_current: float = 2.0              # A, CC phase
    charge_voltage: float = 12.6             # V, CV phase
    charge_efficiency: float = 0.93
    contact_settle_s: float = 0.5            # feet must rest this long before power flows
    ir_wavelength_nm: int = 940
    ir_beacon_power: float = 1.0             # arbitrary units at 1 m on axis


BATTERY = BatterySpec()
ELECTRONICS = ElectronicsSpec()
DOCK = DockSpec()


# ----------------------------------------------------------------------------
# sensors (what the brain gets; each one exists on the physical robot)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorSpec:
    imu: str = "BNO085 (accel + gyro + fused orientation), in the torso"
    joint: str = "servo 12-bit magnetic encoder, speed, load, temperature (bus readback)"
    feet: str = "4 FSR402 per sole (heel/toe x inner/outer)"
    power: str = "INA226 on the battery line: pack voltage and current"
    charge: str = "dock-present / charging flag from the charge controller"
    ir: str = "4 x 940 nm photodiodes on the head (front, left, back, right), amplitude of the 38 kHz beacon"
    camera: str = "Pi Camera Module 3 Wide on the head (optional in sim; costs render time)"

    imu_gyro_noise: float = 0.01       # rad/s
    imu_acc_noise: float = 0.05        # m/s^2
    imu_grav_noise: float = 0.01
    enc_noise: float = 0.0015          # rad (12-bit = 0.0015 rad/count)
    vel_noise: float = 0.05            # rad/s
    fsr_noise: float = 0.3             # N
    v_noise: float = 0.02              # V
    i_noise: float = 0.02              # A


SENSORS = SensorSpec()

CONTROL_HZ = 50        # brain rate (bus read + write of 20 servos at 1 Mbaud fits in < 10 ms)
PHYSICS_HZ = 500


def bill_of_materials() -> list[tuple[str, int, float]]:
    """(item, qty, unit price USD), approximate 2026 hobby prices."""
    n50 = sum(j.servo is STS3250 for j in JOINTS)
    n30 = sum(j.servo is STS3215_12V for j in JOINTS)
    return [
        (STS3250.name + " (hips pitch/roll, knees, ankle pitch)", n50, STS3250.price_usd),
        (STS3215_12V.name + " (hip yaw, ankle roll, arms, neck)", n30, STS3215_12V.price_usd),
        ("Feetech URT-1 / Waveshare bus servo adapter (USB-TTL)", 1, 12.0),
        (ELECTRONICS.computer, 1, 80.0),
        ("Pi Camera Module 3 Wide", 1, 35.0),
        ("BNO085 IMU breakout", 1, 25.0),
        ("FSR402 force sensors", 8, 6.0),
        ("ADS1115 ADC (FSRs)", 2, 6.0),
        ("INA226 current/voltage monitor", 1, 6.0),
        ("940 nm photodiodes (BPW34-class) + op-amp, beacon strength per side", 4, 4.0),
        ("3S 18650 pack with BMS (3.5 Ah)", 1, BATTERY.price_usd),
        ("12 V -> 5 V 5 A buck converter", 1, 10.0),
        ("Ideal-diode bridge (LT4320 board) for foot contacts", 1, 12.0),
        ("Copper tape / sole contact pads, wiring, connectors", 1, 25.0),
        ("PETG filament for brackets (~1 kg)", 1, 22.0),
        ("Fasteners, bearings, servo horns", 1, 30.0),
        ("Dock: 3S CC/CV charger board + 12.6 V supply", 1, 25.0),
        ("Dock: copper plates, IR LED + 38 kHz driver, MCU (presence check)", 1, 20.0),
    ]
