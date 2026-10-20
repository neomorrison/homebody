"""Build the GitHub Pages site (docs/index.html) from docs/tasks.json and the spec.

    python scripts/record_tasks.py     # record the task videos first
    python scripts/build_site.py
"""
import html
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import spec as S  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
REPO = "https://github.com/neomorrison/homebody"

ORDER = ["drive", "pick_can", "pick_sneaker", "pick_sock", "pick_paper_bag", "door", "dryer", "dock", "shove"]

WHY_FAIL = {
    "pick_sock": "Cloth is floppy: the scripted grasp closes on the patches and slips. "
                 "Pinching fabric is the first thing the demonstrations will teach.",
    "pick_paper_bag": "A paper bag needs a pinch on its rim, not a squeeze across it. "
                      "This grasp gets its own demonstrations too.",
}

CSS = """
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1d1b;--mute:#6b6862;--line:#e3dfd6;--ok:#1f7a4d;--okbg:#e3f3ea;
--bad:#a3341f;--badbg:#f8e6e1;--accent:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--card:#1f1f1c;--ink:#ecebe6;--mute:#a3a097;--line:#34332e;
--ok:#6fcf9b;--okbg:#17321f;--bad:#f09a85;--badbg:#3a1d16;--accent:#8fb8e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:40px;line-height:1.1;margin:0 0 12px;letter-spacing:-.02em}
h2{font-size:24px;margin:56px 0 8px}
p.lede{font-size:19px;color:var(--mute);max-width:720px;margin:0 0 20px}
a{color:var(--accent)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px}
.stat span{color:var(--mute);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px;margin-top:18px}
.task{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.task video{width:100%;aspect-ratio:960/544;background:#000;display:block}
.body{padding:14px 16px 16px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px}
.row h3{margin:0;font-size:17px}
.badge{font-size:12px;font-weight:600;padding:3px 9px;border-radius:999px;white-space:nowrap}
.ok{color:var(--ok);background:var(--okbg)}.bad{color:var(--bad);background:var(--badbg)}
.note{margin:8px 0 0;font-size:14px}
.why{margin:8px 0 0;font-size:13px;color:var(--mute)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
@media (max-width:760px){.two{grid-template-columns:1fr}h1{font-size:30px}}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
figure img{width:100%;display:block}
figcaption{padding:10px 14px;font-size:13px;color:var(--mute)}
table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
td,th{text-align:left;padding:8px 12px;border-top:1px solid var(--line);vertical-align:top}
th{width:150px;color:var(--mute);font-weight:500}
tr:first-child td,tr:first-child th{border-top:0}
td.num{text-align:right;white-space:nowrap}
ul.plain{padding-left:20px}
footer{margin-top:56px;color:var(--mute);font-size:13px}
"""


def task_card(key, t):
    ok = t["success"]
    why = WHY_FAIL.get(key, "") if not ok else ""
    return f"""<article class="task">
  <video src="{t['video']}" poster="posters/{key}.jpg" controls muted playsinline preload="none"></video>
  <div class="body">
    <div class="row"><h3>{html.escape(t['title'])}</h3>
      <span class="badge {'ok' if ok else 'bad'}">{'Worked' if ok else 'Not yet'}</span></div>
    <p class="note">{html.escape(t['note'])}</p>
    {f'<p class="why">{html.escape(why)}</p>' if why else ''}
  </div>
</article>"""


def main():
    tasks = json.load(open(os.path.join(DOCS, "tasks.json"), encoding="utf-8"))
    keys = [k for k in ORDER if k in tasks] + [k for k in tasks if k not in ORDER]
    n_ok = sum(tasks[k]["success"] for k in keys)
    bom = sum(q * p for _, q, p in S.bill_of_materials())
    kit = sum(q * p for _, q, p in S.HOUSE_KIT)
    bom_rows = "\n".join(f'<tr><td>{html.escape(name)}{f" x{q}" if q > 1 else ""}</td><td class="num">${q * p:,.0f}</td></tr>'
                         for name, q, p in S.bill_of_materials())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>homebody</title>
<meta name="description" content="A DIY house robot for about ${bom:,.0f}, trying chores in a simulation of its owner's apartment.">
<style>{CSS}</style></head>
<body><main>
<h1>homebody</h1>
<p class="lede">A house robot you can build for about ${bom:,.0f}, trying real chores in a simulated copy of
its owner's apartment. These are scripted attempts, recorded straight from the physics sim. The ones marked
"Not yet" are what the learned brain has to fix.</p>
<div class="stats">
  <div class="stat"><b>{n_ok} of {len(keys)}</b><span>tasks worked</span></div>
  <div class="stat"><b>${bom:,.0f}</b><span>parts, plus a ${kit:,.0f} house kit</span></div>
  <div class="stat"><b>1.82 m</b><span>mast, reaches the dryer</span></div>
  <div class="stat"><b>10</b><span>motor commands, 20 times a second</span></div>
</div>

<h2>Watch it try</h2>
<p>Each clip is one attempt with no retakes. The caption bar shows the time, the room, the battery and the power draw.</p>
<div class="grid">
{chr(10).join(task_card(k, tasks[k]) for k in keys)}
</div>

<h2>The apartment</h2>
<p>The rooms are rebuilt from an iPhone LiDAR scan: walls, doorways and furniture are measured from the
scan, and the floor and rug textures are cut from its photos. It is a clean recreation rather than the raw
scan mesh, so colours, lighting and the movable furniture can change every episode. That keeps a camera-based
brain from memorising one exact picture of the room.</p>
<div class="two">
  <figure><img src="media/home_plan.png" alt="Floor plan of the simulated apartment" loading="lazy">
    <figcaption>Plan: bedroom with walk-in closet, 1 m hallway, living room and kitchen, foyer with the dock
    (orange), laundry closet with the stacked washer and dryer. Blue squares are bins, pink lines are doors.</figcaption></figure>
  <figure><img src="media/home_rooms.png" alt="Six camera views inside the simulated apartment" loading="lazy">
    <figcaption>Inside: bedroom, closet, living room, kitchen, laundry closet, hallway.</figcaption></figure>
</div>

<h2>The robot</h2>
<table>
<tr><th>Base</th><td>Two hub motors from a used hoverboard, sprung casters, anti-tip outrigger wheels, 12 kg steel ballast, a tote on the deck</td></tr>
<tr><th>Reach</th><td>A counterbalanced lift on a 1.82 m mast (the arm runs from 0.12 to 1.62 m high) and a telescoping arm reaching 0.54 m to the side</td></tr>
<tr><th>Hand</th><td>Yaw, pitch and roll wrist with a parallel-jaw gripper on bus servos, and a camera in the palm</td></tr>
<tr><th>Senses</th><td>Lidar, pan/tilt head camera, palm camera, IMU, bumper, cliff sensors, battery monitor, dock beacon, motor encoders</td></tr>
<tr><th>Power</th><td>36 V 180 Wh Li-ion; backs onto a spring-contact dock to charge itself</td></tr>
<tr><th>Brain</th><td>Runs on a desktop GPU over Wi-Fi; a Raspberry Pi 5 on board handles motors, sensors and safety</td></tr>
</table>

<h2>Parts</h2>
<table>
{bom_rows}
<tr><th>Total</th><td class="num"><b>${bom:,.0f}</b></td></tr>
</table>

<h2>What comes next</h2>
<ul class="plain">
  <li>Teleoperated demonstrations of the grasps that fail here (cloth, paper bag rims, the takeout bag handle), then imitation learning on them.</li>
  <li>Reinforcement learning for driving, docking from across the apartment and recovering from bumps.</li>
  <li>Longer chores: loading the washer, moving laundry to the dryer, putting the iron and its cord away.</li>
</ul>

<footer>Code, specs and the sim: <a href="{REPO}">{REPO.replace('https://', '')}</a>.
Rebuild this page with <code>python scripts/record_tasks.py</code> then <code>python scripts/build_site.py</code>.</footer>
</main></body></html>
"""
    open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8", newline="\n").write(page)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    print(f"wrote docs/index.html ({n_ok}/{len(keys)} tasks worked)")


if __name__ == "__main__":
    main()
