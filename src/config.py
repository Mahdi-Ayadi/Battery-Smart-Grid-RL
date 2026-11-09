"""Configuration centrale du projet (Blocs A / B / C).

Tous les hyperparamètres physiques, de données et d'apprentissage sont regroupés
ici pour pouvoir faire des expériences reproductibles.

Convention d'index : le plan du projet parle de "nœuds 18, 25, 33" (1-indexé).
Pandapower indexe les bus à partir de 0, donc : bus 17, 24, 32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Constance topologique ----------------------------------------------------
# Buses pandapower (0-indexé) qui reçoivent les systèmes hybrides
# Solaire + Batterie = nœuds "18, 25, 33" du plan (extrémités du réseau).
BATTERY_BUSES = (17, 24, 32)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BatteryConfig:
    """Paramètres d'une batterie (identique pour les 3 unités)."""

    p_max_mw: float = 1.0        # Puissance max en charge ET décharge (MW)
    capacity_mwh: float = 2.0    # Capacité utile (MWh)
    soc_init: float = 0.5        # SoC initial (fraction de la capacité)
    eff_charge: float = 0.95     # Rendement charge
    eff_discharge: float = 0.95  # Rendement décharge
    dt_h: float = 1.0            # Pas de temps du marché (h)


@dataclass
class PhysicsConfig:
    """Limites physiques et pénalités 'Physics-Informed'."""

    thermal_limit_pct: float = 100.0  # Limite thermique des lignes (%)
    # Pénalité linéaire (€ / point de surcharge absolu, par ligne et par h)
    thermal_penalty: float = 500.0
    # Pénalité quadratique complémentaire (dissuade les grosses surcharges)
    thermal_quad_factor: float = 2.0
    # Plafond de la pénalité thermique par step (stabilise la valeur V(s))
    max_penalty: float = 5000.0
    # Récompense si le solveur de flux échoue à converger
    flow_fail_penalty: float = -2000.0
    # Puissance solaire max par site hybride (MW) [profil 0..1 dans les données]
    solar_max_mw: float = 0.6
    # Hétérogénéité des 3 sites solaires (per-unit)
    pv_node_factors: tuple = (1.0, 0.95, 1.05)
    # Multiplicateur global sur la consommation (tuning des congestions)
    load_multiplier: float = 1.0
    # --- Dimensionnement des câbles (aucune limite n'est fournie par case33bw) ---
    # rating = clip(headroom * I_base * peak_load_factor, min, max) [kA]
    rating_headroom: float = 2.2      # marge d'exploitation du DSO
    rating_peak_load_factor: float = 1.35  # charge de pointe vs charge moyenne
    rating_min_ka: float = 0.040      # gouttière : ~1 MW à 12.66 kV -> congestion possible
    rating_max_ka: float = 0.28       # plafond : câble de tête de feeder


@dataclass
class DataConfig:
    """BLOC A — données synthétiques (compatibles ENTSO-E / Renewables.ninja)."""

    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"        # inclusif (24h du jour)
    price_base: float = 45.0            # €/MWh, moyenne pluri-hebdo EPEX FR
    price_noise_std: float = 8.0        # bruit aléa horaire
    price_spike_prob: float = 0.012     # proba d'un pic de prix par heure
    price_spike_factor: tuple = (1.8, 2.2)  # x1.8 à x4.0
    seed: int = 42


@dataclass
class TrainConfig:
    """BLOC C — hyperparamètres PPO."""

    # Découpage temporel de l'année 2023 (train / validation / test)
    train_end: str = "2023-08-31"
    val_end: str = "2023-10-31"

    total_timesteps: int = 1_000_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 512       # pas de rollout par env (512 * 8 = 4096 / update)
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    seed: int = 42

    log_dir: str = "runs"          # tensorboard
    model_dir: str = "models"      # sauvegardes checkpoint/best
    model_name: str = "ppo_gnn"


@dataclass
class Config:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT_CONFIG = Config()
