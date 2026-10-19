# Design

## Principles

1. **Designed around the owner's chores.** The chores: trash and clothes off the floor, laundry to
   the laundry room, the lever-handle door, the stacked washer and dryer, and self-charging. The design
   changes whenever a chore needs it. The 0.6 m walking humanoid (tag `humanoid-v1`) was dropped because
   it could not reach counters, carry laundry or open doors.
2. **One spec, two bodies.** `homebody/spec.py` is the only place a physical number lives.
3. **Cheap strength through mechanics.** A vertical lift plus a horizontal telescope, as on Hello
   Robot's Stretch: loads push along slides, not against a joint motor. Forceful moves use the big
   axes. A door lever is pressed down by lowering the lift, and a door is pushed by extending the
   telescope or driving.
4. **Commands are what the drivers take.** Duty cycles for the hub motors, step rates for the
   steppers, targets for the bus servos (their own 1 kHz loop).
5. **The brain only gets on-board senses.** Everything else is in `info`.
6. **Durability is designed in and tested.** See below.

## Parts (about $1,190, plus a $50 house kit)

| Part | Qty | Each | Total |
|---|---|---|---|
| Used hoverboard (2 hub motors, mainboard; reflash with hoverboard-firmware-hack-FOC) | 1 | $50 (est) | $50 |
| ST-Link programmer for the reflash | 1 | $10 | $10 |
| New 36 V 10S2P battery with BMS (don't reuse the hoverboard's) | 1 | $70 (est) | $70 |
| Base: plywood/aluminium chassis, 2 casters, 4 outrigger arms + wheels, foam bumper | 1 | $85 | $85 |
| Steel ballast plate, 12 kg | 1 | $35 | $35 |
| Deck tote + mounts | 1 | $10 | $10 |
| Lift: 1.6 m 2040 rail, V-wheel carriage, NEMA17, belt, 2:1 pulleys, constant-force spring | 1 | $150 | $150 |
| Telescoping arm: 3 heavy-duty drawer slides, NEMA17, belt | 1 | $60 | $60 |
| Stepper drivers (36 V capable) | 2 | $12 | $24 |
| Feetech STS3250-C002 (wrist pitch, gripper) | 2 | $74 | $148 |
| Feetech STS3215 12 V (wrist yaw, wrist roll, head pan, head tilt) | 4 | $22 | $88 |
| Bus servo driver board | 1 | $12 | $12 |
| Printed parallel-jaw gripper with TPU pads | 1 | $10 | $10 |
| Raspberry Pi 5 8 GB + cooler + microSD | 1 | $95 | $95 |
| Slamtec RPLIDAR C1 | 1 | $80 | $80 |
| USB RGB cameras (head, palm) | 2 | $20 | $40 |
| BNO085 IMU | 1 | $25 | $25 |
| Cliff sensors x4, bumper switches | 1 | $17 | $17 |
| INA226 battery monitor | 1 | $6 | $6 |
| Dock beacon photodiodes + op-amp | 4 | $4 | $16 |
| 36 V -> 12 V and 36 V -> 5 V converters | 1 | $25 | $25 |
| E-stop, fuses, wiring, connectors | 1 | $40 | $40 |
| PETG filament, fasteners | 1 | $40 | $40 |
| Dock: 42 V 2 A charger, spring contacts, IR LED, MCU, stand | 1 | $50 | $50 |
| House kit: 2 button pushers for the washer/dryer start, printed AprilTags | 1 | $50 | $50 |

Optional: Orbbec Gemini 2 depth camera ($267) in place of the head camera (`EnvConfig(depth=True)`).
Servo, hub-motor and battery figures are reseller listings or estimates; bench-test before training for real.

## Geometry

```
  head camera (pan/tilt) ── 1.68 m
          │
        mast (x -0.17, y +0.08)
          │  carriage + telescope (arm height 0.22 .. 1.52 m) ──> slides out to the right, 0..0.54 m
          │                                              └ wrist yaw ─ pitch ─ roll ─ parallel jaws (+ palm camera)
  ┌──────────────────────────────┐
  │ deck tote      lidar ▶       │  base 0.50 x 0.30 m, hub wheels at ±0.19 m, casters front/back
  └──────────────────────────────┘  outriggers at (±0.32, ±0.32), 15 mm off the floor; dock pads on the rear
```

The robot reaches sideways, so for the washer it parks alongside the machine and slides the arm
straight into the drum. The retracted gripper sits beside the tote, so a 120° wrist swing drops what
it holds into the tote.

## Durability and falls

Measured in sim with an 80 N, 0.2 s shove at the top of the mast, lift up, arm out, 10 directions:

| Design | Tipped |
|---|---|
| 4 kg ballast, outriggers ±0.24/0.20 m | 10/10 |
| 12 kg ballast | 4/10 |
| outriggers ±0.32 m | 4/10 |
| **12 kg + ±0.32 m (adopted)** | **0/10** |
| same, 150-200 N shove | 8-10/10 |

A firm deliberate push at the top still tips it, as with any tall, narrow robot (Stretch tips at about
30 N by the same arithmetic). The layers:

1. **Hard to tip:** ballast and battery low, wide outriggers, the lift lowered and the arm tucked in
   to drive.
2. **Catches itself:** the IMU sees a lean, the wheels drive under it, and the lift drops. These are
   learnable reflexes; `push_rate` trains them.
3. **Survives:** foam on the mast cap and arm, breakaway camera and lidar mounts, a protected
   battery. In sim, impacts break individual parts (`spec.PARTS`).
4. **Calls for help:** a tipped 26 kg robot can't right itself, so it stops its motors and notifies
   the owner.

Other wear limits: bus servos overheat when stalled (thermal model, 70 °C cutout), and the gripper's
torque limit caps its grip at about 60 N so it can't crush things.

## Known sim-to-real gaps

- Clothes are patch grids, not true cloth. Pinch-grasping them needs work (and a real cloth model
  later).
- The servo position loop is idealised (stiffness + damping + torque limit), with no backlash or
  deadband.
- Hub motors use a first-order DC model at the firmware's current limit; the real FOC firmware differs.
- Rugs are fixed in place, and the floor heightfield has 4 cm cells.
- The house is a stand-in floor plan until the real rooms are scanned.

## What next

1. **Teleop and demonstrations.** `scripts/view.py` drives the sim robot by keyboard. Next comes a
   gamepad or phone teleop that records episodes in LeRobot's dataset format, then 50 to 100
   demonstrations per chore, then train ACT or SmolVLA on the RTX 3080.
2. **RL for the body skills:** driving, docking, not tipping, recovering from shoves (`push_rate`).
   Use cheap sensors and many parallel environments; MJX on the GPU would speed this up.
3. **Digital twin:** a phone lidar scan of the real rooms replaces `home.py`.
4. **Hardware:** bench-test one hub motor, the lift and a servo, and fit the motor models in `spec.py`.
