#!/usr/bin/env python
"""BLOC C — Entraînement PPO avec extracteur GNN sur l'environnement pandapower."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.data.generate_data import load_data
from src.envs.grid_env import SmartGridArbitrageEnv
from src.models.gnn_extractor import GNNFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne l'agent GNN+PPO")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total de timesteps (défaut : config)")
    parser.add_argument("--fast", action="store_true",
                        help="Mode fumée : 20k steps pour vérifier le pipeline")
    parser.add_argument("--conv-type", choices=["gat", "gcn"], default="gat")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.fast:
        cfg.train.total_timesteps = 20_000
        cfg.train.n_envs = 4
        cfg.train.n_epochs = 6
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps

    # --- Données : train / validation -------------------------------------------
    df = load_data()
    train_df = df[df.index <= cfg.train.train_end].copy()
    val_df = df[(df.index > cfg.train.train_end) & (df.index <= cfg.train.val_end)].copy()
    print(f"train: {len(train_df)} h | val: {len(val_df)} h")

    # --- Environments ------------------------------------------------------------
    train_env = make_vec_env(
        lambda: SmartGridArbitrageEnv(train_df, cfg, random_start=True),
        n_envs=cfg.train.n_envs,
        seed=cfg.train.seed,
    )
    eval_env = SmartGridArbitrageEnv(val_df, cfg, random_start=False)

    topologie = train_env.envs[0].unwrapped.edge_index_list
    print(f"edge_index: {len(topologie)} arêtes")

    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(
            edge_index=topologie,
            conv_type=args.conv_type,
            hidden_dim=64,
            features_dim=32,
            n_heads=2,
        ),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        seed=cfg.train.seed,
        verbose=1,
        tensorboard_log=str(PROJECT_ROOT / cfg.train.log_dir),
    )

    model_dir = PROJECT_ROOT / cfg.train.model_dir
    best_dir = model_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=max(cfg.train.n_steps * cfg.train.n_envs, 4096) * 2,
        best_model_save_path=str(best_dir),
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.train.total_timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(model_dir / cfg.train.model_name))
    print(f"Modèle final : {model_dir / cfg.train.model_name}")


if __name__ == "__main__":
    main()
