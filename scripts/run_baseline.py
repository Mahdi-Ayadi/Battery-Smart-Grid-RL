#!/usr/bin/env python
"""Baseline 'expert bête' : P&L et violations réseau de la stratégie sans IA."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.rule_based import RuleBasedPolicy
from src.config import DEFAULT_CONFIG
from src.data.generate_data import load_data


def run_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Fait jouer la baseline heure par heure, renvoie le journal horaire."""
    cfg = DEFAULT_CONFIG
    policy = RuleBasedPolicy()

    from src.envs.grid_env import SmartGridArbitrageEnv

    env = SmartGridArbitrageEnv(data, cfg, random_start=False)
    obs, _ = env.reset()
    rows = []
    while True:
        hour = env.df.index[env.t].hour
        obs, reward, terminated, truncated, info = env.step(policy.act(hour))
        rows.append(info)
        if truncated or terminated:
            break
    return pd.DataFrame(rows)


def main() -> None:
    cfg = DEFAULT_CONFIG
    df = load_data()

    # Périodes train (référence) et validation
    train = df[df.index <= cfg.train.train_end]
    val = df[(df.index > cfg.train.train_end) & (df.index <= cfg.train.val_end)]
    test = df[df.index > cfg.train.val_end]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    summaries = []
    for name, period in {"train": train, "val": val, "test": test}.items():
        if len(period) < 48:
            continue
        log = run_baseline(period)
        log.to_csv(out_dir / f"baseline_hourly_{name}.csv", index=False)
        summaries.append(
            {
                "period": name,
                "hours": len(log),
                "PnL_brut_eur": float(log["profit"].sum()),
                "penalites_eur": float(log["thermal_penalty"].sum()),
                "reward_net_eur": float(log["reward"].sum()),
                "hours_over_100pct": int((log["max_loading_pct"] > 100.0).sum()),
                "max_loading_pct": float(log["max_loading_pct"].max()),
                "nonconvergence": int((~log["converged"]).sum()),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
