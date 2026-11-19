"""Synthetic GTA VI trophy-page histories shaped like the final case (backtest only; main.py never imports this).

World model (each scenario draws every parameter at random, so no candidate is tuned to one world):
  - K early-access "testers" hold every trophy from launch night (the real page: 0.63% of 638 = 4 players,
    0.10% of 3,868 = 4 players), so trophies nobody real has reached dilute as K / N.
  - Real players arrive (earn their first trophy, the prologue's) at a rate that slows over the week; Exophase only
    counts players with a trophy, so the prologue trophy stays at 100%.
  - Each player accumulates play hours at a personal intensity (lognormal, with fatigue); a trophy is earned when
    play hours pass its threshold times the player's pace, if the player pursues it at all (probability p).
    Story trophies have thresholds at depth x story length; side, grind and collectible ones their own.
  - Exophase refreshes the page at random intervals; the house reads every 30 then every 10 minutes, so most reads
    repeat. The answer is the last refresh before the target.
Rates are computed analytically over hourly arrival cohorts (no per-player loop), rounded to 2 decimals.
"""

import json
import math
import random
from datetime import datetime, timedelta, timezone

T0 = datetime(2026, 11, 19, tzinfo=timezone.utc)

# (kind, story depth or play-hours threshold, pursuit probability) for each of the 55 trophies, by my reading of names.
# kind: plat, first, story (value = depth 0..1), side (value = play hours), grind (hours), never (100% completion).
SPEC = [
    ("plat", 0, 0), ("first", 0, 1),
    ("story", .12, 1), ("story", .3, 1), ("story", .5, 1), ("story", .75, 1), ("story", 1.0, 1),  # ch1-4, final
    ("story", .06, 1), ("story", .55, 1), ("story", .4, 1), ("story", .65, 1), ("story", .8, 1), ("story", .9, 1),
    ("never", 0, 0),
    ("grind", 60, .3), ("side", 20, .5), ("grind", 40, .3), ("grind", 50, .25), ("grind", 40, .3),
    ("side", 3, .5), ("side", 5, .5),
    ("side", 30, .5), ("side", 35, .45), ("side", 40, .4), ("grind", 45, .4), ("side", 30, .45), ("side", 28, .45),
    ("grind", 80, .2), ("side", 25, .4), ("grind", 60, .25), ("grind", 50, .25), ("grind", 50, .3),
    ("side", 6, .6), ("side", 8, .4), ("side", 20, .6), ("side", 6, .6),
    ("story", .6, 1), ("side", 10, .3), ("side", 12, .35), ("grind", 40, .3), ("side", 8, .4), ("grind", 60, .3),
    ("grind", 35, .3),
    ("story", .35, 1), ("story", .45, 1), ("story", .2, 1), ("story", .7, 1), ("story", 1.0, .97), ("story", .85, 1),
    ("story", .95, 1), ("side", 10, .15),
    ("grind", 30, .4), ("side", 15, .5), ("grind", 60, .25), ("side", 15, .4),
]
assert len(SPEC) == 55


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def scenario(seed, cut_h=59.5, target_h=168.0):
    rng = random.Random(seed)
    K = rng.randint(3, 8)
    alpha = rng.uniform(0.45, 0.95)          # arrival-rate shape a(t) ~ t^(alpha-1)
    tau = rng.uniform(40, 400)               # extra slowdown after the first weekend
    m_int = rng.uniform(0.15, 0.45)          # median play hours per real hour
    s_tot = rng.uniform(0.6, 1.1)            # log-sd of intensity x pace
    gamma = rng.uniform(0.75, 1.0)           # fatigue
    story_len = rng.uniform(30, 100)          # play hours to the final mission at median pace
    drop = rng.uniform(0.1, 0.5)             # share who stop the story by its end
    lag = rng.uniform(0, 4)                  # Exophase sync lag in hours (progress only)
    spec = []
    for kind, v, p in SPEC:
        if kind == "story":
            spec.append((kind, max(0.02, min(1.0, v * rng.uniform(0.7, 1.3))) * story_len, p * (1 - drop * v)))
        elif kind in ("side", "grind"):
            spec.append((kind, v * math.exp(rng.gauss(0, 0.5)), min(0.95, p * math.exp(rng.gauss(0, 0.35)))))
        else:
            spec.append((kind, v, p))
    # chapters 1-4 and the final mission (trophies 3-7) are nested in a real game: keep their thresholds in order
    ch = sorted(spec[2:7], key=lambda x: x[1])
    spec[2:7] = [(k, th, pp) for (k, th, _), (_, _, pp) in zip(ch, sorted(spec[2:7], key=lambda x: -x[2]))]
    # arrivals per hour, cohort centres from -1 h (time zones that launch first) on
    a = [((t + 1.5) ** (alpha - 1)) * (math.exp(-(max(0, t - 60)) / tau)) for t in range(-1, int(target_h) + 2)]
    early = 638 - K
    scale = rng.uniform(2600, 4200) / sum(a[:6])  # real players by 04:30
    cohorts = [(t, n * scale) for t, n in zip(range(-1, int(target_h) + 2), a)]

    def page(t):
        n_real = 0.0
        earned = [0.0] * 55
        for t_arr, n in cohorts:
            if t_arr + 0.5 > t:
                break
            n_real += n
            age = t - t_arr - lag
            lp = math.log(m_int * 24 * (age / 24) ** gamma) if age > 0 else -9
            for k, (kind, th, p) in enumerate(spec):
                if kind == "first":
                    earned[k] += n
                elif kind in ("plat", "never"):
                    continue
                else:
                    earned[k] += n * p * phi((lp - math.log(th)) / s_tot)
        N = K + n_real
        return round(N), [round(100 * (K + e) / N, 2) for e in earned]

    # launch night: 638 tracked, mostly testers' early friends; keep the real page's flavour
    ln_rates = [round(100 * (K + (1 if k in (19, 2) else 0) * 4) / 638, 2) for k in range(55)]
    ln_rates[1] = 100.0
    snaps = [(-0.2, 638, ln_rates)]
    t = rng.uniform(4.0, 4.5)
    mean_gap = rng.uniform(1.5, 12)
    while t < target_h + 1:
        N, r = page(t)
        snaps.append((t, N, r))
        t += mean_gap * rng.uniform(0.3, 1.7)
    history = []
    slot = -1 / 6
    null_p = rng.choice([0, 0, 0.002, 0.01])
    while slot < cut_h - 1e-9:
        cur = [s for s in snaps if s[0] <= slot][-1]
        rates = [None if rng.random() < null_p else x for x in cur[2]]
        at = T0 + timedelta(hours=slot)
        history.append({"slot": iso(at), "at": iso(at + timedelta(seconds=1)), "players_tracked": cur[1],
                        "trophies_earned": int(sum(x for x in cur[2]) * cur[1] / 100), "rates": rates})
        slot += 0.5 if slot < 24 else 1 / 6
    answer = [s for s in snaps if s[0] <= target_h][-1][2]
    return history, answer


def make_case(seed, cut_h=59.5, target_h=168.0, trophies=None):
    hist, ans = scenario(seed, cut_h, target_h)
    at = T0 + timedelta(hours=cut_h)
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": f"s{seed:04d}", "at": iso(at),
            "horizon_minutes": round((target_h - cut_h) * 60), "target": iso(T0 + timedelta(hours=target_h)),
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5", "release_date": "2026-11-19",
                     "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
            "trophies": trophies, "history": hist, "notes": "synthetic"}
    return case, ans
