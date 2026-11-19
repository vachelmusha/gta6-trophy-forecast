"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01). Python 3.13, standard library only.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input/) in name order, writes <id>.out to $MAKEFUN_OUTPUT_DIR (else
/output/): five quantiles of each trophy's Exophase earn rate at the case's target.

Model (logit space, fitted only on distinct refreshed pages, never on repeated reads):
  1. Collapse the history into the pages Exophase actually refreshed (a read identical to the one before is a repeat).
  2. A handful of early-access players hold every trophy from launch night (0.63% of 638 and 0.10% of 3,868 are both
     ~4 players). Estimate them as K = the smallest earner count on the last page, and model each trophy's share among
     the other, real players, so a trophy nobody real has reached is not read as "falling" while N grows.
  3. Per trophy, a least-squares line of logit(real share) on ln(1 + hours since 2026-11-19T00:00Z) over the last
     24 h of refreshed pages; ln(players tracked) gets the same line. Both slopes are damped (x0.7) and extrapolated to
     the target; the rate is (K + share x (N - K)) / N.
  4. Medians of the nested main-story chain (Chapter 1..4, final mission) are made non-increasing.
  5. Spread: normal in logit with sd^2 = 0.5^2 x days ahead (the random walk) + (0.25 x the extrapolated move)^2
     + the fit's residual^2.
Any exception falls back to the house baseline (last rate, random-walk spread), which is always valid.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]
T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Model parameters (chosen on the synthetic backtest, see backtest/).
P = {
    "floor": True,       # separate the fixed early-access holders (who hold every trophy) from real players
    "window_h": 24.0,    # fit the trend on distinct refreshed pages from the last this-many hours
    "min_frac": 0.10,    # ...whose players tracked is at least this share of the last page's
    "damp": 0.7,         # shrink the fitted log-time slope toward zero
    "sigma": 0.5,        # random-walk spread, logit per sqrt(day) (the house's; not narrowed on purpose)
    "kappa": 0.25,       # extra spread per unit of extrapolated logit move
}


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv(lg):
    if lg > 30:
        return 100.0
    if lg < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def hours(ts):
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (dt - T0).total_seconds() / 3600.0


def round4(x):
    v = math.floor(x * 10000 + 0.5) / 10000
    return int(v) if v.is_integer() else v


def snapshots(case):
    """Distinct refreshed pages: [(hours, players_tracked, rates)], nulls filled from the trophy's previous read."""
    n = len(case["trophies"])
    last = [None] * n
    snaps = []
    for read in case["history"]:
        rates = read.get("rates") or []
        for i in range(n):
            v = rates[i] if i < len(rates) else None
            if isinstance(v, (int, float)) and math.isfinite(v):
                last[i] = float(v)
        key = (read.get("players_tracked"), tuple(last))
        if snaps and snaps[-1][3] == key:
            continue
        t = hours(read.get("at") or read["slot"])
        snaps.append((t, read.get("players_tracked") or 0, list(last), key))
    return [(t, nt, r) for t, nt, r, _ in snaps]


def fill_missing(rates):
    known = sorted(x for x in rates if x is not None)
    med = known[len(known) // 2] if known else 1.0
    return [med if x is None else x for x in rates]


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)) if n > 2 else 0.0
    return b, a, res


def model(case, p=P):
    """Returns [(median logit, logit sd)] per trophy."""
    snaps = snapshots(case)
    t_last, n_last, r_last = snaps[-1]
    r_last = fill_missing(r_last)
    H = hours(case["target"])
    d = max(0.0, H - t_last) / 24.0
    fit = [s for s in snaps if s[0] >= t_last - p["window_h"] and s[1] >= p["min_frac"] * n_last and s[0] > 1.0]
    xH, xl = math.log(1 + H), math.log(1 + t_last)
    if p.get("floor"):
        return floor_model(r_last, n_last, fit, xH, xl, d, p)
    out = []
    for i in range(len(r_last)):
        L = logit(r_last[i])
        pts = [(math.log(1 + t), logit(r[i])) for t, _, r in fit if r[i] is not None]
        b, res = 0.0, 0.0
        if len(pts) >= 2:
            b, _, res = linfit([x for x, _ in pts], [y for _, y in pts])
        move = p["damp"] * b * (xH - xl)
        sd = math.sqrt(p["sigma"] ** 2 * d + (p["kappa"] * move) ** 2 + res ** 2)
        out.append((L + move, sd))
    return out


def lg(s):
    return math.log(s / (1.0 - s))


def floor_model(r_last, n_last, fit, xH, xl, d, p):
    """Separate the fixed early-access holders (K, who hold every trophy) from real players: model the real players'
    share of each trophy in logit vs ln(1 + hours), and players tracked N in ln N vs ln(1 + hours)."""
    if n_last < 50 or len(fit) < 2:
        p = dict(p, floor=False)
        return [(logit(r), math.sqrt(p["sigma"] ** 2 * d)) for r in r_last]
    K = max(0.0, min(r * n_last / 100.0 for r in r_last))
    bN, _, _ = linfit([math.log(1 + t) for t, _, _ in fit], [math.log(max(1, n)) for _, n, _ in fit])
    NH = n_last * math.exp(p.get("damp_n", p["damp"]) * max(0.0, bN) * (xH - xl))
    out = []
    for i in range(len(r_last)):
        pts = []
        for t, n, r in fit:
            if r[i] is None or n <= K + 1:
                continue
            s = (r[i] * n / 100.0 - K) / (n - K)
            lo = 0.5 / (n - K)
            pts.append((math.log(1 + t), lg(min(1 - lo, max(lo, s)))))
        s_last = (r_last[i] * n_last / 100.0 - K) / (n_last - K)
        lo = 0.5 / (n_last - K)
        Ls = lg(min(1 - lo, max(lo, s_last)))
        b, res = 0.0, 0.0
        if len(pts) >= 2:
            b, _, res = linfit([x for x, _ in pts], [y for _, y in pts])
        damp = p["damp"]
        if "damp_c" in p:  # rejected variant, kept for the backtest: less damping far from saturation
            damp = p["damp"] * (1 - s_last) + p["damp_c"] * max(0.0, s_last)
        Ls_H = Ls + damp * b * (xH - xl)
        sH = 1.0 / (1.0 + math.exp(-Ls_H))
        rate = 100.0 * (K + sH * (NH - K)) / NH
        m = logit(rate)
        move = m - logit(r_last[i])
        sd = math.sqrt(p["sigma"] ** 2 * d + (p["kappa"] * move) ** 2 + res ** 2)
        out.append((m, sd))
    return out


def story_chain(trophies):
    """Indices of the nested main-story trophies in order (Chapter 1..n, then the final mission), from descriptions."""
    chapters, final = {}, None
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "").strip().lower()
        if desc.startswith("complete chapter "):
            num = desc[len("complete chapter "):].rstrip(".").strip()
            if num.isdigit():
                chapters[int(num)] = i
        elif desc == "complete the final mission.":
            final = i
    chain = [chapters[k] for k in sorted(chapters)]
    if final is not None:
        chain.append(final)
    return chain


def nonincreasing(vals):
    """Pool-adjacent-violators: the closest non-increasing sequence (least squares)."""
    blocks = []  # [sum, count]
    for v in vals:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out += [s / c] * c
    return out


def order_story(case, params):
    chain = story_chain(case["trophies"])
    if len(chain) < 2:
        return params
    fixed = nonincreasing([params[i][0] for i in chain])
    params = list(params)
    for i, m in zip(chain, fixed):
        params[i] = (m, params[i][1])
    return params


def fallback(case):
    """The house baseline: last rate, random-walk spread."""
    last = fill_missing(snapshots(case)[-1][2]) if case.get("history") else [1.0] * len(case["trophies"])
    d = case.get("horizon_minutes", 6510) / 1440
    return [(logit(r), 0.5 * math.sqrt(d)) for r in last]


def rows(case, params):
    rows = []
    for t, (m, s) in zip(case["trophies"], params):
        q = [min(100.0, max(0.0, inv(m + z * s))) for z in Z]
        q = [round4(x) for x in q]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def valid(out, case):
    ids = [t["id"] for t in case["trophies"]]
    got = [r["id"] for r in out["trophies"]]
    if sorted(got) != sorted(ids) or len(set(got)) != len(got):
        return False
    for r in out["trophies"]:
        q = r["q"]
        if len(q) != 5 or any(not (isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 100) for x in q):
            return False
        if any(q[j] < q[j - 1] for j in range(1, 5)):
            return False
    return True


def forecast_case(case):
    try:
        out = rows(case, order_story(case, model(case)))
        if valid(out, case):
            return out
    except Exception as e:  # noqa: BLE001 - any model failure falls back to a valid simple forecast
        print(f"model failed on {case.get('id')}: {e!r}", file=sys.stderr)
    try:
        return rows(case, fallback(case))
    except Exception as e:  # noqa: BLE001
        print(f"fallback failed on {case.get('id')}: {e!r}", file=sys.stderr)
        return {"trophies": [{"id": t["id"], "q": [0.1, 1, 5, 20, 60]} for t in case["trophies"]]}


def main():
    inp = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    outp = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    outp.mkdir(parents=True, exist_ok=True)
    for path in sorted(inp.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"cannot read {path.name}: {e!r}", file=sys.stderr)
            continue
        out = forecast_case(case)
        (outp / f"{path.stem}.out").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
