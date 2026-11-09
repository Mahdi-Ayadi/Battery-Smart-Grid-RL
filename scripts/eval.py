#!/usr/bin/env python
"""Évaluation de l'agent entraîné (plan §4.5) — les 3 livrables de R&D :

  1. P&L cumulé agent vs baseline           (graphique financier)
  2. Histogramme de la charge max de ligne  (graphique physique / safety)
  3. SoC vs Prix vs Solaire sur 3 journées  (comportement vs météo)

Usage :
  python scripts/eval.py --model models/ppo_gnn --period test
  python scripts/eval.py --period val          # baseline seule
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO

from src.baselines.rule_based import RuleBasedPolicy
from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.data.generate_data import load_data
from src.envs.grid_env import SmartGridArbitrageEnv


def _period_slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    if period == "val":
        d = df[(df.index > cfg.train.train_end) & (df.index <= cfg.train.val_end)]
    else:
        d = df[df.index > cfg.train.val_end]
    return d.copy().sort_index()


def run_policy(env: SmartGridArbitrageEnv, policy_fn) -> pd.DataFrame:
    """Déroule un épisode complet ; policy_fn(hour, obs) -> action (3,)."""
    obs, _ = env.reset()
    rows, soc_hist = [], []
    while True:
        action = policy_fn(env.df.index[env.t].hour, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        soc_hist.append(info["soc_mwh"])
        rows.append(info)
        if truncated or terminated:
            break
    df = pd.DataFrame(rows).reset_index(drop=True)
    soc_arr = np.asarray(soc_hist)
    df["soc1"], df["soc2"], df["soc3"] = soc_arr[:, 0], soc_arr[:, 1], soc_arr[:, 2]
    return df


def evaluate(model_path: str | None, period: str) -> None:
    cfg = DEFAULT_CONFIG
    test = _period_slice(load_data(), period)
    print(f"Évaluation sur {len(test)} h ({test.index[0]} -> {test.index[-1]})")

    # --- Baseline -----------------------------------------------------------
    env0 = SmartGridArbitrageEnv(test, cfg, random_start=False)
    policy = RuleBasedPolicy()
    baseline_log = run_policy(env0, lambda hour, obs: policy.act(hour))

    # --- Agent GNN + PPO ------------------------------------------------------
    agent_log = None
    if model_path is not None:
        model = PPO.load(model_path)
        env = SmartGridArbitrageEnv(test, cfg, random_start=False)
        agent_log = run_policy(
            env, lambda hour, obs: model.predict(obs, deterministic=True)[0]
        )

    # ------------------------------------------------------------------ PLOTS --
    plots_dir = PROJECT_ROOT / "results/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) P&L cumulé
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline_log["datetime"], baseline_log["profit"].cumsum(),
            label="Baseline 'expert bête'", color="tab:red")
    if agent_log is not None:
        pnl = agent_log["profit"].cumsum()
        ax.plot(agent_log["datetime"], pnl, label="Agent GNN + PPO", color="tab:blue")
    ax.set_xlabel("Heure")
    ax.set_ylabel("P&L cumulé (€OR)")
    ax.set_title(f"Fleet d'arbitrage — P&L cumulé (période {period})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "pnl_cumulatif.png", dpi=150)
    plt.close(fig)

    # 2) Safety : histogramme de la charge max de ligne
    fig, ax = plt.subplots(figsize=(9, 5))
    base_u = np.clip(baseline_log["max_loading_pct"] / 100.0, 0.0, 2.5)
    bins = np.linspace(0.0, 2.5, 40)
    ax.hist(base_u, bins=bins, alpha=0.55, label="Baseline", color="tab:red")
    if agent_log is not None:
        ax.hist(np.clip(agent_log["max_loading_pct"] / 100.0, 0.0, 2.5),
                bins=bins, alpha=0.55, label="Agent GNN+PPO", color="tab:blue")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="Limite thermique (100 %)")
    ax.set_xlabel("Charge max du réseau (fraction de la limite thermique)")
    ax.set_ylabel("Nombre d'heures")
    ax.set_title("Histogramme de charge de la ligne la plus contrainte (max par ligne/h)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "charge_ligne_max.png", dpi=150)
    plt.close(fig)

    # 3) SoC vs Prix vs Solaire (3 journées contrastées)
    daily_solar = (
        test["solar_pu"].groupby(test.index.date).mean().sort_index()
    )
    d_sorted = daily_solar.sort_values()
    n = len(d_sorted)
    # 3 journées contrastées : haute, médiane, basse lumière
    picks = [
        d_sorted.index[int(n * 0.9) - 1],
        d_sorted.index[int(n * 0.5)],
        d_sorted.index[max(int(n * 0.1) - 1, 0)],
    ]
    logs = agent_log if agent_log is not None else baseline_log
    day_keys = pd.Series([d.strftime("%Y-%m-%d") for d in logs["datetime"]])

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 9), squeeze=False)
    for ax_row, day in zip(axes, picks):
        ax = ax_row[0]
        mask = day_keys == str(day)
        hours = logs.loc[mask, "datetime"].dt.hour
        ax.bar(hours.to_numpy(), logs.loc[mask, "price_raw"].to_numpy(),
               color="tab:orange", alpha=0.45, label="Prix DA (€/MWh)")
        ax.plot(hours.to_numpy(), logs.loc[mask, "solar_pu"].to_numpy(),
                color="tab:green", lw=1.8, label="Solaire (pu)")
        ax2 = ax.twinx()
        for i, col in enumerate(["soc1", "soc2", "soc3"], start=1):
            ax2.plot(hours.to_numpy(), logs.loc[mask, col].to_numpy(),
                     color=f"C{i}", lw=2, label=f"SoC batt. {i}" if day == picks[0] else None)
        ax.set_ylabel("Prix (€/MWh) | Solaire (pu)")
        ax2.set_ylabel("SoC (MWh)")
        ax.set_title(f"{day} (solaire moyen : {daily_solar[day]:.2f} pu)")
    fig.suptitle("SoC des batteries face au soleil et aux prix (comportement appris)")
    fig.tight_layout()
    fig.savefig(plots_dir / "soc_vs_prix_soleil.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- RESUME --
    summary: dict = {
        "period": period,
        "hours": int(len(baseline_log)),
        "baseline_pnl_eur": float(baseline_log["profit"].sum()),
        "baseline_max_loading_pct": float(baseline_log["max_loading_pct"].max()),
        "baseline_hours_over_limit": int((baseline_log["max_loading_pct"] > 100.0).sum()),
        "baseline_no_convergence": int((~baseline_log["converged"]).sum()),
    }
    if agent_log is not None:
        summary.update(
            {
                "agent_pnl_eur": float(agent_log["profit"].sum()),
                "agent_max_loading_pct": float(agent_log["max_loading_pct"].max()),
                "agent_hours_over_limit": int((agent_log["max_loading_pct"] > 100.0).sum()),
                "agent_no_convergence": int((~agent_log["converged"]).sum()),
            }
        )
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    out_json = PROJECT_ROOT / f"results/eval_summary_{period}.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Chemin du fichier .zip SB3 ; omis = baseline seule")
    parser.add_argument("--period", choices=["val", "test"], default="test")
    args = parser.parse_args()
    evaluate(args.model, args.period)
