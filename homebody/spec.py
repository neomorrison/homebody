"""The robot, as parts you can buy.

A Stretch-style house robot built from cheap parts:

* base: two hoverboard hub motors on a plywood/aluminium chassis, steel
  ballast low down, casters front and rear, anti-tip outrigger wheels at the
  corners, and a tote on the deck for carrying laundry and trash;
* a 1.6 m mast with a belt-driven lift, counterbalanced by a constant-force
  spring so the stepper only moves the payload;
* a telescoping arm (three drawer-slide stages, one stepper and belt) that
  slides out sideways, so heavy loads push along slides instead of twisting a
  joint, and reaches straight into a washer or dryer drum;
* a three-axis wrist (yaw, pitch, roll) and a soft two-finger gripper on
  Feetech bus servos, with a camera in the palm;
* a pan/tilt head camera at the top of the mast, a lidar on the front of the
  base, IMU, bumper, cliff sensors, battery monitor and a dock beacon.

It charges by backing onto a wall dock. Everything physical lives here; the
simulator and the hardware runtime both read it. Prices are 2026 listings
where checked, otherwise marked "est". Motor figures marked "approx" must be
measured on a bench before trusting a trained brain on hardware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G_ACC = 9.81


# ----------------------------------------------------------------------------
# actuators
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorSpec:
    """A DC motor referenced to its output shaft at the rated voltage, driven by a duty cycle:

        tau = stall_torque * (duty * V_supply / v_rated) - (stall_torque / no_load_speed) * omega

    clipped to +-stall_torque (the driver's current limit). Nothing holds still
    unless the brain keeps pushing.
    """
    name: str
    stall_torque: float      # N m at v_rated (or the driver's current limit)
    no_load_speed: float     # rad/s at v_rated
    stall_current: float     # A
    no_load_current: float   # A
    v_rated: float           # V
    supply: str              # "battery" (straight off the pack) or "12v" (regulated bus)
    mass: float              # kg
    armature: float          # kg m^2, reflected rotor inertia
    friction: float          # N m
    damping: float           # N m s/rad
    price_usd: float
    thermal_resistance: float = 8.0
    thermal_capacity: float = 6.0
    cutout_temp: float = 70.0

    @property
    def winding_resistance(self) -> float:
        return self.v_rated / self.stall_current

    @property
    def torque_constant(self) -> float:
        return self.stall_torque / self.stall_current


KGCM = G_ACC / 100.0

# Hoverboard hub motor (6.5 in wheel) on hoverboard-firmware-hack-FOC, 36 V, current-limited (approx)
HUB_MOTOR = MotorSpec(
    name="Hoverboard hub motor 6.5in (approx)", stall_torque=12.0, no_load_speed=50.0,
    stall_current=15.0, no_load_current=0.5, v_rated=36.0, supply="battery", mass=2.8,
    armature=0.02, friction=0.15, damping=0.05, price_usd=0.0,     # comes with the used hoverboard
    thermal_resistance=1.5, thermal_capacity=400.0, cutout_temp=90.0,
)
# Feetech STS3250-C002: 50 kg.cm peak, 0.133 s/60deg at 12 V, 4.2 A
STS3250 = MotorSpec(
    name="Feetech STS3250-C002", stall_torque=50 * KGCM, no_load_speed=(math.pi / 3) / 0.133,
    stall_current=4.2, no_load_current=0.28, v_rated=12.0, supply="12v", mass=0.0745,
    armature=0.012, friction=0.06, damping=0.02, price_usd=74.0,
)
# Feetech STS3215 12 V (Waveshare "ST3215 12V"): 30 kg.cm
STS3215 = MotorSpec(
    name="Feetech STS3215 12V", stall_torque=30 * KGCM, no_load_speed=(math.pi / 3) / 0.222,
    stall_current=2.7, no_load_current=0.18, v_rated=12.0, supply="12v", mass=0.055,
    armature=0.008, friction=0.04, damping=0.015, price_usd=22.0,
)


@dataclass(frozen=True)
class AxisSpec:
    """A stepper-driven linear axis. The brain commands speed (a step rate), as on the real driver.
    Steppers hold position while powered (holding current) and go slack when the pack dies."""
    name: str
    stroke: float            # m of travel
    max_speed: float         # m/s
    max_force: float         # N at the carriage
    moving_power: float      # W
    holding_power: float     # W
    friction: float          # N, rails + belt
    price_usd: float


# NEMA17 + 2:1 belt reduction + GT2 20T pulley on a 1.6 m 2040 rail, constant-force spring counterbalance
LIFT = AxisSpec("lift", stroke=1.50, max_speed=0.15, max_force=120.0, moving_power=10.0, holding_power=4.0,
                friction=8.0, price_usd=150.0)
# three-stage telescope on heavy-duty drawer slides, NEMA17 + GT2 belt
ARM = AxisSpec("arm", stroke=0.54, max_speed=0.25, max_force=60.0, moving_power=8.0, holding_power=3.0,
               friction=6.0, price_usd=60.0)


@dataclass(frozen=True)
class Joint:
    name: str
    axis: tuple
    range: tuple             # degrees for hinges, metres for slides
    motor: MotorSpec
    slide: bool = False
    gear: float = 1.0        # joint force = motor torque x gear (a rack and pinion of radius 1/gear)
    torque_limit: float = 1.0   # the servo's torque-limit register, as a fraction of stall
    stiffness_err: float = 0.15 # position mode: error (rad or m) at which the servo's loop hits full torque


WRIST = [
    Joint("wrist_yaw", (0, 0, 1), (-100, 280), STS3215),     # 0 = gripper heading out along the arm
    Joint("wrist_pitch", (-1, 0, 0), (-60, 95), STS3250),    # 0 = gripper pointing down, 90 = pointing out
    Joint("wrist_roll", (0, 0, 1), (-150, 150), STS3215),
    # parallel jaws on a rack and pinion (20 mm pinion): each finger slides 0..55 mm, 0 = closed
    Joint("gripper", (0, 1, 0), (0.0, 0.055), STS3250, slide=True, gear=50.0, torque_limit=0.25, stiffness_err=0.01),
]
HEAD = [
    Joint("head_pan", (0, 0, 1), (-200, 100), STS3215),
    Joint("head_tilt", (0, 1, 0), (-30, 85), STS3215),       # positive looks down
]

# Action order. 10 numbers in [-1, 1]: wheel duty x2, lift speed, arm speed, then the six bus servos.
# Servos run in their own position mode by default (the servo's internal ~1 kHz loop; the brain sends a
# target, -1..1 across the joint range). SERVO_MODE = "pwm" makes them raw duty cycles instead: purer,
# but a 20 Hz brain driving PWM makes servos chatter and overheat, on hardware as in sim.
WHEELS = ["wheel_l", "wheel_r"]
AXES = ["lift", "arm"]
SERVOS = WRIST + HEAD
SERVO_NAMES = [j.name for j in SERVOS]
ACTION_NAMES = WHEELS + AXES + SERVO_NAMES
N_ACTIONS = len(ACTION_NAMES)
DUTY_NAMES = WHEELS + SERVO_NAMES
SERVO_MODE = "position"
DUTY_MOTORS = [HUB_MOTOR, HUB_MOTOR] + [j.motor for j in SERVOS]


# ----------------------------------------------------------------------------
# geometry (m, base frame: +x forward, +y left, z up) and masses (kg)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    base_length: float = 0.50
    base_width: float = 0.30           # chassis between the wheels
    base_bottom: float = 0.050
    base_top: float = 0.130
    wheel_radius: float = 0.0825       # 6.5 in hub wheels
    wheel_width: float = 0.055
    wheel_track: float = 0.38          # centre to centre
    caster_radius: float = 0.025
    caster_x: float = 0.20
    caster_travel: float = 0.012       # sprung casters: +-12 mm
    caster_spring: float = 4000.0      # N/m per caster: light, so the drive wheels carry the weight
    caster_preload: float = 0.008      # m: sprung slightly down, so the casters still touch on a bump
    outrigger_radius: float = 0.020
    outrigger_clearance: float = 0.015 # anti-tip wheels ride this far above the floor
    outrigger_xy: tuple = (0.32, 0.32)  # wide enough that an 80 N shove at the mast top doesn't tip it (sim trials)
    mast_x: float = -0.17
    mast_y: float = 0.08
    mast_top: float = 1.82             # tall enough to reach into the dryer (door centre 1.55 m, measured)
    lift_low: float = 0.10             # telescope height above the floor at the bottom of travel
    arm_x: float = -0.12               # telescope centre line (x)
    arm_root_y: float = 0.18           # inner end of the telescope (left of the mast, so the retracted
                                       # gripper sits beside the deck tote)
    stage_length: float = 0.30
    n_stages: int = 3
    wrist_drop: float = 0.06           # wrist yaw axis -> wrist pitch axis
    finger_length: float = 0.09
    lidar_x: float = 0.23
    lidar_z: float = 0.165
    tote_x: tuple = (-0.06, 0.18)      # deck tote, inner extent
    tote_y: tuple = (-0.13, 0.13)
    tote_height: float = 0.18
    dock_pad_z: float = 0.090          # charge pads on the rear bumper
    dock_pad_y: float = 0.060


GEOM = Geometry()

MASS = {   # ballast: a steel plate under the deck, the cheapest anti-tip there is
    "chassis": 2.5, "ballast": 12.0, "electronics": 0.40, "lidar": 0.11, "mast": 1.8, "carriage": 0.40,
    "stage": 0.25, "wrist_bracket": 0.08, "palm": 0.10, "finger": 0.03, "head": 0.25, "tote": 0.30,
}


# ----------------------------------------------------------------------------
# power and dock
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BatterySpec:
    """10S2P Li-ion pack (36 V, hoverboard format, new not salvaged) with BMS."""
    cells_series: int = 10
    capacity_ah: float = 5.0
    r_internal: float = 0.25
    cell_cutoff_v: float = 3.0
    cell_full_v: float = 4.2
    mass: float = 1.10
    price_usd: float = 70.0     # est
    ocv_soc: tuple = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    ocv_v: tuple = (3.00, 3.30, 3.45, 3.56, 3.63, 3.70, 3.76, 3.84, 3.93, 4.02, 4.10, 4.20)

    @property
    def nominal_v(self) -> float:
        return 3.65 * self.cells_series

    @property
    def nominal_wh(self) -> float:
        return self.capacity_ah * self.nominal_v


@dataclass(frozen=True)
class ElectronicsSpec:
    """Always-on load. The Pi is the robot's spinal cord; the brain runs on a desktop GPU over Wi-Fi."""
    computer: str = "Raspberry Pi 5 (8 GB)"
    computer_w: float = 6.0
    hub_board_w: float = 1.5           # hoverboard mainboard idle
    lidar_w: float = 1.2
    cameras_w: float = 1.4             # head + palm USB cameras
    sensors_w: float = 0.4
    servo_logic_w: float = 0.10        # per bus servo
    converter_efficiency: float = 0.90 # 36 V -> 12 V / 5 V bucks

    def idle_power(self) -> float:
        five = self.computer_w + self.lidar_w + self.cameras_w + self.sensors_w
        return five / self.converter_efficiency + self.hub_board_w + len(SERVOS) * self.servo_logic_w


@dataclass(frozen=True)
class DockSpec:
    """Wall dock: backboard with two spring-loaded contacts and an IR beacon. The robot reverses onto it."""
    board_size: tuple = (0.10, 0.30, 0.24)   # a 10 cm deep stand: the rear outriggers pass either side of it
    contact_size: tuple = (0.04, 0.03)
    contact_y: float = 0.065
    spring_travel: float = 0.005
    beacon_z: float = 0.20
    charge_current: float = 2.0        # A, a standard 42 V 2 A e-bike/hoverboard charger
    charge_voltage: float = 42.0
    charge_efficiency: float = 0.93
    contact_settle_s: float = 0.5
    ir_beacon_power: float = 1.0
    price_usd: float = 50.0


BATTERY = BatterySpec()
ELECTRONICS = ElectronicsSpec()
DOCK = DockSpec()


# ----------------------------------------------------------------------------
# sensors
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LidarSpec:
    model: str = "Slamtec RPLIDAR C1"
    range: tuple = (0.05, 12.0)
    rays: int = 180
    fov_deg: float = 240.0          # the deck tote and mast shade the rear
    noise: float = 0.01
    dropout: float = 0.02
    price_usd: float = 80.0


@dataclass(frozen=True)
class CameraSpec:
    model: str
    fovy: float
    depth_range: tuple | None = None
    depth_noise: float = 0.0
    price_usd: float = 0.0


HEAD_CAM = CameraSpec("USB RGB camera, wide (head)", fovy=70.0, price_usd=20.0)
PALM_CAM = CameraSpec("USB RGB camera, wide (palm)", fovy=80.0, price_usd=20.0)
DEPTH_UPGRADE = CameraSpec("Orbbec Gemini 2 (optional, replaces the head camera)", fovy=66.0,
                           depth_range=(0.15, 10.0), depth_noise=0.01, price_usd=267.0)
LIDAR = LidarSpec()


@dataclass(frozen=True)
class SensorNoise:
    enc: float = 0.0015
    vel: float = 0.05
    axis_pos: float = 0.0005    # stepper step counting (m)
    gyro: float = 0.01
    acc: float = 0.05
    grav: float = 0.01
    odom_drift: float = 0.02
    v: float = 0.05
    i: float = 0.05


NOISE = SensorNoise()

CONTROL_HZ = 20
PHYSICS_HZ = 250        # 4 ms steps: as stable as 500 Hz here (item drift tested), twice as fast


# ----------------------------------------------------------------------------
# durability: parts that break, and what breaks them
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PartSpec:
    """A component that fails after enough hard knocks. ``impulse_limit`` is the
    accumulated excess impulse (N s above ``force_threshold``) it survives."""
    name: str
    force_threshold: float
    impulse_limit: float
    effect: str


PARTS = [
    PartSpec("wrist", 60.0, 6.0, "wrist and gripper servos stop responding (stripped gears)"),
    PartSpec("palm_cam", 40.0, 2.0, "palm camera goes black"),
    PartSpec("head", 40.0, 3.0, "head servos and head camera stop"),
    PartSpec("lidar", 50.0, 3.0, "lidar returns nothing"),
    PartSpec("mast", 250.0, 40.0, "mast bent: lift jams"),
]


def bill_of_materials() -> list[tuple[str, int, float]]:
    """(item, qty, unit price USD). About $1,190 robot + dock, plus the house kit."""
    return [
        ("Used hoverboard (2 hub motors, mainboard; reflash with hoverboard-firmware-hack-FOC)", 1, 50.0),
        ("ST-Link programmer for the reflash", 1, 10.0),
        ("New 36 V 10S2P battery with BMS (don't reuse the hoverboard's)", 1, BATTERY.price_usd),
        ("Base: plywood/aluminium chassis, 2 spring-loaded casters, 4 outrigger arms + wheels, foam bumper", 1, 95.0),
        ("Steel ballast plate, 12 kg", 1, 35.0),
        ("Deck tote (plastic bin) + mounts", 1, 10.0),
        ("Lift: 1.6 m 2040 rail, V-wheel carriage, NEMA17, belt, 2:1 pulleys, constant-force spring", 1, LIFT.price_usd),
        ("Telescoping arm: 3 heavy-duty drawer slides, NEMA17, belt", 1, ARM.price_usd),
        ("Stepper drivers (36 V capable, e.g. DM542-class)", 2, 12.0),
        (STS3250.name + " (wrist pitch, gripper)", 2, STS3250.price_usd),
        (STS3215.name + " (wrist yaw, wrist roll, head pan, head tilt)", 4, STS3215.price_usd),
        ("Bus servo driver board", 1, 12.0),
        ("Printed parallel-jaw gripper (rack and pinion) with TPU finger pads", 1, 10.0),
        ("Raspberry Pi 5 8 GB + cooler + microSD", 1, 95.0),
        (LIDAR.model, 1, LIDAR.price_usd),
        (HEAD_CAM.model, 1, HEAD_CAM.price_usd),
        (PALM_CAM.model, 1, PALM_CAM.price_usd),
        ("BNO085 IMU", 1, 25.0),
        ("IR cliff sensors x4 + bumper switches", 1, 17.0),
        ("INA226 battery monitor", 1, 6.0),
        ("940 nm photodiodes + op-amp (dock beacon)", 4, 4.0),
        ("36 V -> 12 V 10 A and 36 V -> 5 V 5 A converters", 1, 25.0),
        ("E-stop, fuses, wiring, connectors", 1, 40.0),
        ("PETG filament + fasteners", 1, 40.0),
        ("Dock: 42 V 2 A charger, spring contacts, IR LED, MCU, board", 1, DOCK.price_usd),
    ]


HOUSE_KIT = [
    ("SwitchBot-style button pushers (washer + dryer start)", 2, 25.0),
    ("Printed AprilTag markers (dock, door, washer, dryer, basket, trash can, pod bucket)", 1, 0.0),
    ("Vacuum: trigger lock clip", 1, 5.0),
    ("Vacuum: Wi-Fi smart plug (robot switches it on and off)", 1, 10.0),
    ("Vacuum: retractable cord reel (stays plugged in, no cord to wind)", 1, 25.0),
]
