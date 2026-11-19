"""Backtest: score candidate models on synthetic final-shaped cases (59.5 h of reads -> forecast at 168 h).

    python3 backtest/run.py [n_scenarios] [cut_h]

Scenarios are cached under backtest/cache/. Loss is the contest's: mean over trophies x levels of
2 * max(tau d, (tau - 1) d), d = logit(y) - logit(q), rates clamped to 0.01..99.99.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import main  # noqa: E402
import sim  # noqa: E402

LEVELS = main.LEVELS
TROPHIES = json.loads((HERE / "practice" / "p00.in").read_text())["trophies"]


def pinball(out, answer):
    tot, n = 0.0, 0
    for row, y in zip(sorted(out["trophies"], key=lambda r: r["id"]), answer):
        ly = main.logit(y)
        for tau, q in zip(LEVELS, row["q"]):
            d = ly - main.logit(q)
            tot += 2 * max(tau * d, (tau - 1) * d)
            n += 1
    return tot / n


def coverage(out, answer):
    rows = sorted(out["trophies"], key=lambda r: r["id"])
    return sum(r["q"][0] <= y <= r["q"][4] for r, y in zip(rows, answer)) / len(answer)


def load(n, cut_h=59.5, target_h=168.0):
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    cases = []
    for s in range(n):
        f = cache / f"s{s:04d}_{cut_h}_{target_h}.json"
        if f.exists():
            c, a = json.loads(f.read_text())
        else:
            c, a = sim.make_case(s, cut_h, target_h, TROPHIES)
            f.write_text(json.dumps([c, a]))
        cases.append((c, a))
    return cases


# ---- candidates: each maps a case to [(median logit, sd)] ----

def flat(case):
    """House baseline: last read, random walk 0.5 logit per sqrt(day)."""
    return main.fallback(case)


def house_trend(case):
    """House curve baseline, as RULES.txt describes it: per trophy, LS line of logit rate on ln(1 + hours since
    2026-11-19T00:00Z) over the last 24 h of reads (repeats included), read at the target; spread the random walk's
    and the fit's residual combined."""
    hist = case["history"]
    t_last = main.hours(hist[-1]["at"])
    reads = [r for r in hist if main.hours(r["at"]) >= t_last - 24]
    if len(reads) < 3:
        return flat(case)
    H = main.hours(case["target"])
    d = case["horizon_minutes"] / 1440
    rw = 0.5 * math.sqrt(d)
    last = main.fill_missing(main.snapshots(case)[-1][2])
    out = []
    for i in range(len(case["trophies"])):
        pts = [(math.log(1 + main.hours(r["at"])), main.logit(r["rates"][i])) for r in reads if r["rates"][i] is not None]
        if len(pts) < 3:
            out.append((main.logit(last[i]), rw))
            continue
        b, a, res = main.linfit([x for x, _ in pts], [y for _, y in pts])
        out.append((a + b * math.log(1 + H), math.sqrt(rw * rw + res * res)))
    return out


def damped(**kw):
    p = dict(main.P, **kw)
    return lambda case: main.model(case, p)


def evaluate(cands, cases):
    res = {}
    for name, f in cands.items():
        losses, covs = [], []
        for c, a in cases:
            out = main.rows(c, f(c))
            assert main.valid(out, c), name
            losses.append(pinball(out, a))
            covs.append(coverage(out, a))
        res[name] = (sum(losses) / len(losses), sum(covs) / len(covs), losses)
    return res


def report(res, base="flat"):
    b = res[base][2]
    for name, (loss, cov, ls) in sorted(res.items(), key=lambda kv: kv[1][0]):
        wins = sum(x < y for x, y in zip(ls, b))
        print(f"{name:32s} loss {loss:.4f}  cov90 {cov:.2f}  beats {base} in {wins}/{len(ls)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else 59.5
    cases = load(n, cut)
    cands = {"flat": flat, "house_trend": house_trend, "damped(main.P)": main.model}
    for damp in (0.0, 0.4, 0.6, 0.8, 1.0):
        cands[f"damped d={damp}"] = damped(damp=damp)
    report(evaluate(cands, cases))
