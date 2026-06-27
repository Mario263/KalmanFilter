# PPO Raw Baseline — XAU/USD (Paper Reproduction)

Faithful reproduction of the **"PPO without Kalman filtering" (PPO Raw)** baseline from:

> Kili, Raouyane, Rachdi, Bellafkih. *Kalman-Enhanced Deep Reinforcement Learning for Noise-Resilient Algorithmic Trading in Volatile Gold Markets.* IJACSA 16(11), 2025.

Reproduces **only** PPO Raw (no Kalman, no DQN, no RPPO). Trade frequency is an **output**, never tuned.

## Methodology

- **Data:** XAU/USD OHLCV with volume from [Dukascopy](https://github.com/Eghosa-Osayande/dukascopy) (`scripts/download_dukascopy_xauusd.py`). Native **H1** and **D1** bars — no resampling. Fallback HF dataset in `config/experiment.json` (1-min → hourly).
- **State:** 22 features = 5 OHLCV + 17 technical indicators (`ta` library, §IV.B).
- **Normalization:** causal rolling z-score — **252 bars** on daily, **1440 bars** (60×24h) on hourly (§IV.B, Eq.13).
- **Split:** train through 2022-12-31; eval **2023-01-02 → 2024-09-12** (§IV.A).
- **Action:** `Discrete(3)` → {sell −1, hold 0, buy +1}; fixed **lots per trade** (default 1.0 lot = 100 oz).
- **Reward (Eq.22):** `1.0·Return − 2.0·Drawdown − 0.5·Cost + 0.1·Stability` (§IV.F).
- **Costs:** $7/lot round-trip commission (scaled by `lots_per_trade`).
- **Agent:** Stable-Baselines3 **PPO**, actor & critic `[512,512,256,128]` Tanh, lr 3e-4 linear decay, 500k timesteps (§IV.G.2/H).

## Install

```bash
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python dukascopy-python   # data download only
```

## Data

```bash
.venv/bin/python scripts/download_dukascopy_xauusd.py
# -> data/dukascopy/xauusd_1h.csv, data/dukascopy/xauusd_1d.csv
```

## Train

```bash
# Dukascopy daily (paper-aligned timeframe)
.venv/bin/python train.py --config config/experiment_dukascopy_1d.json \
  --save-dir models/dukascopy_1d

# Dukascopy hourly
.venv/bin/python train.py --config config/experiment_dukascopy_1h.json \
  --save-dir models/dukascopy_1h

# Legacy HF hourly pipeline
.venv/bin/python train.py --config config/experiment.json

# Utilities
.venv/bin/python train.py --smoke              # 20k-step sanity check
.venv/bin/python train.py --mode eval --config config/experiment_dukascopy_1d.json \
  --save-dir models/dukascopy_1d
```

Outputs per run: `models/<run>/ppo_xauusd_*.zip`, `ppo_raw_metrics.json`, `logs/<run>/`.

## Paper target (Table I/II)

| Cumulative return | CAGR | Sharpe | Max drawdown | Win rate |
|---|---|---|---|---|
| 15.39% | 6.00% | 0.69 | −11.22% | 50.16% |

## Latest results (Dukascopy, 1 lot, eval window above)

| | 1D final | 1D best val | 1H final | 1H best val | Paper |
|---|---|---|---|---|---|
| Return | +20.87% | +32.44% | −26.01% | +45.74% | +15.39% |
| Sharpe | 1.17 | 0.98 | −0.30 | 2.12 | 0.69 |
| Max DD | −10.65% | −12.42% | −60.43% | −9.42% | −11.22% |
| Round trips | 18 | 49 | 572 | 144 | — |

**1D** exceeds paper targets on return and Sharpe with similar drawdown. **1H** overtrades and degrades after early validation peaks — only the final checkpoint is saved (no best-model checkpointing yet).

Compare runs:

```bash
.venv/bin/python scripts/compare_timeframes.py
# -> results/dukascopy_1h_vs_1d.json
```

## Nautilus event-driven backtest

Bar-by-bar inference with the trained PPO weights. Fixed lot sizing and env-matched commission; observations are precomputed 22-vectors (no indicator recomputation inside Nautilus).

```bash
.venv/bin/python nautilus/run_backtest.py --config config/experiment_dukascopy_1d.json
# -> nautilus/dukascopy_1d_metrics.json
```

**1D validation:** Nautilus (+20.87%) matches the RL env (+20.87%) on the same eval window — 18 round trips, 36 fills, 528 daily bars.

## Configuration

| File | Purpose |
|---|---|
| `config/experiment_dukascopy_1d.json` | Native Dukascopy daily, z-score 252 |
| `config/experiment_dukascopy_1h.json` | Native Dukascopy hourly, z-score 1440 |
| `config/experiment.json` | Legacy HF 1-min → hourly resample |

Override at runtime: `--config path/to/experiment.json` or `RL_EXPERIMENT_CONFIG=...`.

## Project structure

```
config/                  # experiment JSON (data, env, train, nautilus)
scripts/
  download_dukascopy_xauusd.py
  compare_timeframes.py
train.py                 # CLI entry (train / eval)
src/rl_gold_trading/     # PPO Raw pipeline
nautilus/                # Nautilus Trader backtest (inference only)
  sim.py  strategy.py  run_backtest.py
models/  logs/  results/  data/   # gitignored artifacts
```

## Notes

- Paper under-specifications (indicator details, Sharpe annualization, data vendor) mean exact numeric match is not guaranteed across vendors/timeframes.
- Derived from `JonusNattapong/Reinforcement-Learning-for-Gold-Trading`. MIT License. Research/educational use only; trading involves risk.
