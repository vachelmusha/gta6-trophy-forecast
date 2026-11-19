"""Check every <id>.out in an output dir against its <id>.in: the contest's shape rules.

    python3 backtest/validate.py INPUT_DIR OUTPUT_DIR
Exits non-zero on any problem.
"""

import json
import math
import sys
from pathlib import Path


def check(case, out_path):
    errs = []
    if not out_path.is_file() or out_path.is_symlink():
        return ["missing or not a regular file"]
    raw = out_path.read_bytes()
    if len(raw) > 65536:
        errs.append(f"too large: {len(raw)} bytes")
    try:
        out = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"not JSON: {e}"]
    if not isinstance(out, dict) or list(out.keys()) != ["trophies"] or not isinstance(out["trophies"], list):
        return ["top level must be exactly {\"trophies\": [...]}"]
    want = sorted(t["id"] for t in case["trophies"])
    got = []
    for row in out["trophies"]:
        if not isinstance(row, dict) or set(row.keys()) != {"id", "q"}:
            errs.append(f"bad row keys: {row}")
            continue
        got.append(row["id"])
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            errs.append(f"id {row['id']}: q must have 5 numbers")
            continue
        for x in q:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 100:
                errs.append(f"id {row['id']}: bad quantile {x}")
        if any(q[j] < q[j - 1] for j in range(1, 5)):
            errs.append(f"id {row['id']}: quantiles decrease {q}")
    if sorted(got) != want:
        errs.append(f"ids differ: missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}, "
                    f"dupes {len(got) - len(set(got))}")
    return errs


def main():
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    cases = sorted(inp.glob("*.in"))
    for p in cases:
        case = json.loads(p.read_text(encoding="utf-8"))
        errs = check(case, outp / f"{p.stem}.out")
        print(f"{p.stem}: {'OK' if not errs else 'INVALID'}  ({len(case['trophies'])} trophies)")
        for e in errs[:10]:
            print("   ", e)
        bad += bool(errs)
    if not cases:
        print("no cases found")
        bad = 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
