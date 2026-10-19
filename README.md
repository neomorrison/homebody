# homebody

A house robot you can build for about $1,250, and a simulation of the home it works in.

It's designed for real chores: picking trash and clothes up off the floor, carrying laundry to the
laundry room, opening the lever-handle door, loading the washer and dryer, and keeping itself charged
by backing onto its dock. The design copies the idea behind Hello Robot's Stretch, built from cheap
parts: a vertical lift plus a telescoping arm, so heavy loads push along slides instead of twisting a
joint.

![demo](media/demo_strip.png)

The earlier version, a 0.6 m walking humanoid, is kept at the tag `humanoid-v1`.

## The robot

| | |
|---|---|
| Base | two hub motors from a used hoverboard (reflashed with hoverboard-firmware-hack-FOC), casters, anti-tip outrigger wheels at ±32 cm, 12 kg steel ballast, a tote on the deck for carrying things |
| Reach | a 1.8 m mast with a counterbalanced belt lift (the arm's height runs 0.12 to 1.62 m) and a telescoping arm reaching 0.54 m out to the side. The gripper reaches from the floor into the dryer drum |
| Hand | a three-axis wrist (yaw, pitch, roll) and parallel-jaw soft gripper on Feetech bus servos, with a camera in the palm |
| Senses | lidar (RPLIDAR C1), pan/tilt head camera, palm camera, IMU, bumper, 4 cliff sensors, battery monitor, dock beacon, every motor's encoder, load and temperature |
| Power | 36 V 10S2P Li-ion (180 Wh). About 19 W idle, so 9 h waiting; 30 W driving |
| Brain | runs on your desktop GPU over Wi-Fi; a Raspberry Pi 5 on the robot handles motors, sensors and safety |
| Mass, size | 26 kg, 0.5 m long, 0.64 m wide at the outriggers (fits your 1 m hallway and doorways), 1.88 m tall |

Every number lives in [homebody/spec.py](homebody/spec.py). The simulator and the hardware runtime both
read it, so the robot you train is the robot you build. Parts list, power, safety and known gaps are in
[DESIGN.md](DESIGN.md).

## The home: your apartment, from your phone scan

The sim's home is the owner's actual apartment, from a LiDAR scan taken with the free 3d Scanner App on
an iPhone Pro (2026-10-19). `scripts/import_scan.py` squares the scan to the walls and turns it into a
4 cm height map ([homebody/assets/home_scan.npz](homebody/assets/home_scan.npz)). It holds the
bedroom with its walk-in closet, the 1 m hallway, the living room and kitchen, the foyer, and the
laundry closet with a stacked 24-inch washer and dryer. [homebody/home.py](homebody/home.py) strips the
scanned clutter out of that map, and puts simulated versions of what the robot operates where they
were measured:

- **Machines:** washer and dryer (door centres at 0.5 m and 1.55 m, measured from the scan).
- **Doors:** the bedroom door (lever latch), and the laundry closet door (knob and magnetic catch,
  standing in for the real louvered folding doors).
- **Containers and zones:** the kitchen trash can, a recycling spot beside it, the laundry basket
  and a shoe rack in the closet, a tray for the iron on the desk, and the head of the bed for
  pillows.
- **Also:** the charging dock on the foyer wall, the rolling desk chair, and the living-room rug.

The owner's real clutter comes back as movable items, starting where it was scanned
([homebody/items.py](homebody/items.py)):

| Item | Starts | Belongs |
|---|---|---|
| Takeout bag (with a handle loop) | on the bed | kitchen trash can |
| 2 aluminium cans | on the desk | recycling |
| 2 paper bags, a cardboard box (hollow, pinch a wall) | by the bedroom door | recycling |
| 2 shirts, jeans, 2 socks (floppy patch grids) | floor and bed | laundry basket or washer |
| Sneakers, slides | bedroom floor | shoe rack |
| 2 pillows (soft) | stacked at the foot of the bed | head of the bed |
| Clothing iron with its cord (linked segments) | on the floor, cord strung out | its tray on the desk |
| 4 laundry pods | in the pod bucket | washer |

`EnvConfig(scan_visual=True)` draws the textured scan, so the robot's cameras see the real rooms.
`scatter=True` puts the clutter anywhere on free floor instead.

![your home in the sim](media/home_scan_views.png)

What else the robot has to cope with:

- **Floors:** small bumps and a slight tilt (regenerated every episode), floor friction from laminate
  to hardwood, rugs, rubber mats, wet and dusty spots, sills in the doorways, cables.
- **Movers:** the desk chair rolls away when bumped.
- **Power:** motors draw from the pack, and the hub motors weaken as it sags. Servos overheat if
  stalled. Below the BMS cutoff, everything stops.
- **People:** `push_rate` adds accidental shoves near the top of the mast.
- **Damage:** hard knocks break specific parts. A broken wrist or mast, tipping over, or a dead battery
  ends the episode.

## Use it

```bash
pip install -r requirements.txt
python scripts/smoke_test.py           # 8 physical checks, ~1 min
python scripts/view.py --scan          # drive it yourself around your home (keys printed on start)
python scripts/record.py --scan        # the scripted demo above, to media/demo.mp4
python scripts/import_scan.py scans/<date>/textured_output.obj   # after a new scan
```

```python
from homebody import HomebodyEnv, EnvConfig
env = HomebodyEnv(EnvConfig(reward="chores", cameras=(96, 128)))
obs, info = env.reset(seed=0)          # obs["state"]: 255 sensor readings (env.obs_names), plus camera images
obs, reward, dead, timeout, info = env.step(env.action_space.sample())   # 10 motor commands
```

The commands are what the motor drivers take:
- **Hub motors:** duty cycle.
- **Lift and telescope steppers:** speed.
- **Six bus servos:** target positions. Each servo runs its own fast position loop; `servo_mode="pwm"`
  gives raw duty instead, which makes servos chatter and overheat from a 20 Hz brain, in sim as on
  hardware.

`info` carries ground truth (room, position, distance to the dock, where every item is, door angles,
part damage) and is never part of the observation.

Rewards are pluggable:
- `survival`: staying alive
- `energy`: plus stored energy
- `chores`: plus one per item put where it belongs (see the table above)
- `dock_shaped`
- or any function of `(info, prev_info)`

The sim runs at about 35 control steps per second on one CPU core with no clutter out (1.7x real time),
and about 21 (real time) with all 22 items out.

## What's proven so far, in your apartment (`scripts/smoke_test.py`)

| Check | Result |
|---|---|
| Drives down the hallway | 0.94 m in 3 s at 27 W |
| Lift reaches dryer height, holds through a power cut | arm at 1.57 m (the dryer door centre is 1.55 m) |
| Picks a can off the foyer floor into its tote | yes (the sneaker and slides work too) |
| Pulls the laundry closet door off its catch | 11° after a 20 cm pull |
| Charges on the foyer dock | yes; an empty battery is death |
| Survives an 80 N shove at the mast top, lift up and arm out | 0/8 tips. A 200 N shove still tips it about 5 times in 8 |
| All 22 items start in the house, none fall through | yes |

Not yet:
- **Grasps:** the paper bags (pinch the rim), clothes (pinch the fabric) and the takeout bag (hook the
  handle) each need their own grasp; demonstrations should teach these first.
- **Bigger chores:** loading the washer, the dryer transfer, the bedroom door from the hallway side,
  and docking from across the apartment.

## Implanting the brain

[homebody/interface.py](homebody/interface.py) defines `Body`, with `SimBody` and a `RobotBody`
skeleton. The skeleton covers the servo bus, with TODOs for the hub motors, stepper drivers and sensor
drivers. Moving a trained brain to the robot is `RobotBody(...).run(brain)`.
