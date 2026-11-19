# Backtest

```sh
python3 backtest/run.py 80            # candidate sweep on 80 synthetic final-shaped cases (cut 59.5 h -> 168 h)
python3 backtest/make_synthetic.py    # synthetic f00 + edge cases into backtest/synthetic/
mkdir -p out && MAKEFUN_INPUT_DIR=backtest/synthetic MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 backtest/validate.py backtest/synthetic out
```

- `sim.py`: synthetic page histories. A fixed set of 3-8 early-access holders own every trophy (as on the real
  launch-night page: 0.63% of 638 and 0.10% of 3,868 are both ~4 players). Real players arrive at a slowing rate,
  play at lognormal intensities with fatigue, and earn story trophies at depth x story length (30-100 play hours) and
  side/grind trophies at their own thresholds with pursuit probabilities. Exophase refreshes at random 0.5-20 h gaps,
  and the house reads every 30 then every 10 minutes. Every parameter is redrawn per scenario.
- `run.py`: the contest's pinball loss on the logit scale, the two house baselines rebuilt from RULES.txt, and candidates.
- `validate.py`: output-shape checker. `practice/`: the kit's p00/p01.

## Results (mean loss; lower is better; cov90 = share inside the 5-95 band)

| candidate | cut 59.5 h (80 cases) | cut 48 h (40 cases) |
|---|---|---|
| **chosen: tester floor + damped share trend (w24, x0.7) + story order** | **0.307** (cov 0.96) | **0.476** (cov 0.91) |
| (c') same, damping by saturation level (0.8 rare / 0.4 common) | 0.318 | 0.512 |
| (b) damped logit trend, no floor, w18 x0.8 | 0.352 | 0.561 |
| (d) tester floor, undamped | 0.426 | 0.702 |
| house curve baseline (log-time line on last 24 h of reads) | 0.473 (cov 0.90) | 0.879 |
| (b) damped logit trend, w36 x0.6 | 0.501 | 0.890 |
| house baseline (flat, random walk) | 1.076 (cov 0.52) | 1.441 |

Rejected along the way: narrowing the random-walk spread to 0.3 logit/sqrt(day) scored better here (0.243) and was
**not** taken. The simulator is smoother than the real page, so a narrow band would be fitted to my own assumptions.

## How weak this evidence is

Every number above comes from a simulator I wrote, so it ranks candidates under my assumptions about player behaviour,
not under GTA VI's. The only real data is two launch-night pages (638 and 3,868 players). They confirm the tester floor
and the 100% prologue trophy and nothing about week-one dynamics. Exophase and PSNProfiles refuse automated fetches, so
no later real page could be used as a pseudo-case. The practice cases (one unseen refresh, 90 minutes) have no refreshed
page to fit, so there the model is exactly the house baseline (about 1.51).
