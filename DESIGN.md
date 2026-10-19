# Design

## Principles

1. **One spec, two bodies.** `homebody/spec.py` is the only place a physical number lives. The MJCF is
   generated from it and the hardware runtime reads it.
2. **The brain gets only on-board senses.** Anything the physical robot cannot measure (position in
   the room, velocity, distance to the dock) stays in `info`.
3. **Muscles, not commands.** The action is a PWM duty cycle per servo. Nothing holds a pose unless the
   brain keeps pushing, exactly like the real servos in PWM mode.
4. **Energy is life.** Every watt comes out of a simulated pack; the only refill is the physical dock.

## Kinematics (20 DOF)

```
                 neck_yaw, neck_pitch
                        |
 shoulder_pitch - shoulder_roll - elbow      (x2)
                        |
                     torso (IMU, Pi 5, battery)
                        |
 hip_yaw - hip_roll - hip_pitch - knee - ankle_pitch - ankle_roll - sole (copper pad + 4 FSR)   (x2)
```

Hip axes intersect at one point; thigh and shank 130 mm; sole 120 x 55 mm. Joint ranges are in `spec.JOINTS`.

## Parts (about $1,040)

| Part | Qty | Each | Total |
|---|---|---|---|
| Feetech STS3250, 12 V, 4.9 N m (hip pitch/roll, knees, ankle pitch) | 8 | $45 | $360 |
| Feetech STS3215, 12 V, 2.9 N m (hip yaw, ankle roll, arms, neck) | 12 | $22 | $264 |
| Bus servo USB adapter (Feetech URT-1 or Waveshare) | 1 | $12 | $12 |
| Raspberry Pi 5 (8 GB) | 1 | $80 | $80 |
| Pi Camera Module 3 Wide | 1 | $35 | $35 |
| BNO085 IMU | 1 | $25 | $25 |
| FSR402 force sensors | 8 | $6 | $48 |
| ADS1115 ADC for the FSRs | 2 | $6 | $12 |
| INA226 battery monitor | 1 | $6 | $6 |
| 940 nm photodiodes + op-amp (beacon strength per side) | 4 | $4 | $16 |
| 3S 18650 pack with BMS, 3.5 Ah | 1 | $35 | $35 |
| 12 V to 5 V 5 A buck converter | 1 | $10 | $10 |
| Ideal-diode bridge (LT4320) for the foot contacts | 1 | $12 | $12 |
| Copper sole pads, wiring, connectors | 1 | $25 | $25 |
| PETG filament for brackets | 1 kg | $22 | $22 |
| Fasteners, bearings, horns | 1 | $30 | $30 |
| Dock: 3S CC/CV charger + 12.6 V supply | 1 | $25 | $25 |
| Dock: copper plates, IR LED + 38 kHz driver, small MCU | 1 | $20 | $20 |

Servo figures are reseller listings, not an official datasheet. Before training anything for real,
put one STS3250 and one STS3215 on a bench, measure stall torque, no-load speed and current at 12 V,
and update `spec.py` (a fitting script is planned).

## Power path

```
 dock plate A ─┐                      ┌─ left sole pad
               ├─ (feet) ─ ideal-diode bridge ─ charge input ─ BMS ─ 3S pack ─ INA226 ─┬─ servo bus 12 V (20 servos)
 dock plate B ─┘                      └─ right sole pad                                 └─ buck 5 V ─ Pi 5, sensors
```

* The dock keeps its plates dead until its MCU measures the robot's sense resistor across them, then
  enables the CC/CV charger (2 A to 12.6 V). Nothing is live when no robot is there; a foot bridging
  both plates reads as a short and is refused.
* The robot keeps running while charging: the charger supplies the load first, the rest goes into the pack.

## Sensors and the observation vector (125 numbers)

| Slice | Count | Hardware |
|---|---|---|
| joint angle, speed, load, temperature | 20 x 4 | servo bus readback |
| gravity direction, gyro, accelerometer | 9 | BNO085 |
| sole forces | 8 | FSR402 via ADS1115 |
| pack voltage, current, state of charge, charging | 4 | INA226 + coulomb count, charger flag |
| beacon strength front/left/back/right | 4 | photodiodes; line of sight is ray-cast in sim, furniture blocks it |
| previous action | 20 | (memory) |

The head camera can be added as a second observation (`EnvConfig(camera=(64, 64))`); it costs render time.

## Known sim-to-real gaps

* Servo model is a first-order DC motor; real Feetech PWM mode has firmware deadband and a nonlinear
  gearbox. Bench identification will narrow this.
* Hip, shoulder and torso parts do not collide with each other (the real brackets are designed not to);
  legs, feet and hands do collide with each other.
* The floor heightfield has 4 cm cells: it captures warps, slopes and steps but not grout lines or
  carpet pile; those come in only as friction and softness.
* Rugs are rigid and fixed. Real rugs slide on hard floors and their edges curl.
* Friction is Coulomb with one coefficient per surface; real rubber on carpet depends on pressure and speed.
* IR beacon is an idealised cos^2 / r^2 with occlusion; real photodiodes see reflections and sunlight.
* Contacts are MuJoCo soft contacts; the copper pads are assumed to conduct whenever they touch.

## What next (the open decision)

The body and world are done. How the brain should learn is a choice:

1. **Pure survival.** One reward: alive. Closest to "they just do", and almost certainly too sparse to
   discover walking from random twitches. Useful as the final test, not the curriculum.
2. **Developmental curriculum** (recommended): stage the same brain through the order a child learns
   (hold posture, get up, stand, step, walk to a visible beacon, dock), each stage with a short-lived
   battery (`capacity_scale`) and a floor that gets harder as it improves (`terrain_level` 0 to 1, then
   past 1 so a real home feels easy), then remove the shaping and keep only survival. GPU-parallel training
   (MJX on the RTX 3080) would make this hours, not days.
3. **Two-level brain**, like malecns-test: a learned motor cortex (muscles -> walking/turning skills)
   under a learned or evolved drive system (hunger = low battery). Faster, but it reintroduces
   high-level commands between the layers.
4. **Evolution + lifetime learning**, like humanoid-civilization's inner values: a population of
   brains, each living until its battery dies; the survivors' weights seed the next generation.

Hardware can start in parallel: print one leg, bench-test the servos, fit the motor model.
