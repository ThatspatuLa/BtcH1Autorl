# BTC H1 AutoRL — Weighted DCA Evolutionary Search System

One self-evolving BTC H1 long-only Weighted DCA trading strategy, discovered through genetic-algorithm-based search over 500-candidate generations.

## Architecture

- **Market:** BTC, H1 timeframe, 5-year window (2021-06 to 2026-06)
- **Product:** Leveraged BTC CFD (configurable: leverage, margin mode, funding cost)
- **Direction:** Long-only
- **Two evolving engines:** DCA weighting/accumulation + TP/exit/harvest
- **Goal:** Most robust organism across time, drawdown, monthly consistency, volatility, friction, margin stress — NOT highest profit
- **Framework:** Freqtrade for data/indicators/config/deployment; custom research engine for evolution

## Quick Start

```bash
# Use the freqtrade venv (already configured with all deps)
/home/spatula/freqtrade/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from configs import Settings
s = Settings.from_files(
    'configs/freqtrade/config.json',
    'configs/experiments/default.json',
    'configs/research/default.json',
)
print(f'Leverage: {s.leverage}, candidates/gen: {s[\"candidates_per_gen\"]}')
"
```

## Project Structure

```
BtcH1Autorl/
├── configs/                  # Immutable config system (Stage 1)
│   ├── freqtrade/            # Freqtrade-compatible config (starting_balance, leverage, margin, dry_run)
│   ├── experiments/          # Per-experiment overrides (10x leverage test, etc.)
│   ├── research/             # Custom evolution-engine config (candidates_per_gen=500, parallel_workers)
│   └── loader.py             # Immutable Settings merger
│   └── ids.py                # experiment_id, candidate_id, genome_id, run_id generators
│   └── metadata.py           # Run metadata JSON schema + writer
├── data/                     # BTC H1 candles (raw + processed) — Stage 2
│   ├── raw/
│   ├── processed/
│   └── metadata/             # monthly_windows.json, worst_adverse_move.json, etc.
├── results/                  # Per-experiment immutable outputs
├── logs/                     # Structured JSON logs per experiment
├── genomes/                  # Genome files (5 examples committed, others gitignored)
├── reports/                  # Phase 1 + Phase 2 reporting — Stage 15
├── scripts/                  # Ad-hoc runner scripts
├── tests/                    # pytest test suite
├── paper_trading/            # Paper mode framework — Stage 17
├── docs/                     # Project documentation
├── requirements.txt          # Pinned Python deps
├── pyproject.toml            # pytest + ruff config
└── README.md
```

## Locked Decisions

- **Reward weights:** 0.55 profit / 0.15 DD quality / 0.10 Sharpe / 0.10 PF / 0.10 TPM (LOCKED for v1)
- **Hard rejects:** net_profit ≤ 0, drawdown > 35%, TPM < 5, invalid equity, too few trades
- **DD penalty:** 25-30% × 0.85, 30-35% × 0.50
- **Candidates per generation:** 500
- **Combination matrix (Stage 13):** K=5 DCA × L=10 TP = 50 combos
- **Joint evolution (Stage 14):** 500 candidates = 20 elite + 360 mutate/crossover + 120 fresh
- **Stress test (Stage 16):** top 10 × 10 regimes
- **Compute budget:** 8h hard cap per evolution stage (28800s), save unfinished_status on cap
- **Buffer_pct:** Configurable per experiment (default 20%)
- **Min reporting (Phase 1):** summary + breakdown + leaderboard + rejected reasons + best_genome.json + generation_history.json (no PNGs blocking Stage 10)

## Status

- **Gate 0:** ACCEPTED 2026-06-20
- **Stage 1 (Project Foundation):** Complete — ready for Gate 1 review
- **Next gates:** Gate 1 (foundation), Gate 2 (data pipeline), Gate 3 (DCA skeleton)

See Kanban `task-btc-h1-autorl` in `~/Projects/ZenNew/data/kanban-tasks.json` for full stage breakdown.
