"""Write synthetic final-shaped cases to backtest/synthetic/: ~60 h of reads from launch night to 2026-11-21T11:30Z,
several refreshes, growing players tracked, the 55 real trophies, some null rates; target 6,510 minutes later.
Also writes edge cases (empty history, a trophy null in every read, one read only) that must still give valid output.

    python3 backtest/make_synthetic.py
"""

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim  # noqa: E402

TROPHIES = json.loads((HERE / "practice" / "p00.in").read_text())["trophies"]
out = HERE / "synthetic"
out.mkdir(exist_ok=True)

case, answer = sim.make_case(7, 59.5, 168.0, TROPHIES)
case["id"] = "f00"
# sprinkle nulls: a few reads miss a trophy, and the last read misses two
for k, read in enumerate(case["history"]):
    if k % 37 == 5:
        read["rates"][k % 55] = None
case["history"][-1]["rates"][10] = None
case["history"][-1]["rates"][40] = None
assert case["horizon_minutes"] == 6510 and case["at"] == "2026-11-21T11:30:00Z"
(out / "f00.in").write_text(json.dumps(case), encoding="utf-8")
(out / "f00.answer.json").write_text(json.dumps(answer), encoding="utf-8")

e1 = copy.deepcopy(case)
e1["id"], e1["history"] = "x01", []
e2 = copy.deepcopy(case)
e2["id"] = "x02"
for read in e2["history"]:
    read["rates"][20] = None
e3 = copy.deepcopy(case)
e3["id"], e3["history"] = "x03", e3["history"][-1:]
for c in (e1, e2, e3):
    (out / f"{c['id']}.in").write_text(json.dumps(c), encoding="utf-8")
n_ref = len({(r["players_tracked"]) for r in case["history"]})
print(f"f00: {len(case['history'])} reads, {n_ref} distinct pages, players "
      f"{case['history'][0]['players_tracked']} -> {case['history'][-1]['players_tracked']}")
