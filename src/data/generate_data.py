"""BLOC A — Pipeline de données.

Génère (ou charge) un DataFrame horaire indexé par DateTime avec :
  - `price`      : prix Day-Ahead (€/MWh), signature d'un profil EPEX
  - `solar_pu`   : production solaire (per-unit, 0..1) d'un site type France
  - `load_scale` : consommation résidentielle (per-unit, facteur horaire)

En mode réalité, il suffira de remplacer ces 3 colonnes par des téléchargements
ENTSO-E (prix) et Renewables.ninja (solaire) : les colonnes et l'index restent
identiques, donc ni l'environnement ni l'agent ne changeront.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DataConfig, PROJECT_ROOT

PRICE_COL, SOLAR_COL, LOAD_COL = "price", "solar_pu", "load_scale"


# --- Formes horaires ------------------------------------------------------------------------
def _price_daily_shape() -> np.ndarray:
    """Signature journalière EPEX : creux nocturne, pic du soir ~19h."""
    return np.array(
        [
            0.75, 0.65, 0.60, 0.58, 0.62, 0.70, 0.85, 1.00, 1.05, 1.00, 0.92,
            0.88, 0.86, 0.85, 0.88, 0.95, 1.10, 1.35, 1.60, 1.80, 1.70, 1.35,
            1.05, 0.88,
        ]
    )


def _load_daily_shape() -> np.ndarray:
    """Signature résidentielle : pointe matinale + pic 19h-21h."""
    return np.array(
        [
            0.72, 0.68, 0.66, 0.66, 0.68, 0.75, 0.92, 1.08, 1.06, 1.00, 0.98,
            0.96, 0.95, 0.95, 0.97, 1.02, 1.10, 1.18, 1.28, 1.32, 1.26, 1.12,
            0.94, 0.80,
        ]
    )


def _seasonal_factor(doy: np.ndarray, amplitude: float = 0.15) -> np.ndarray:
    """Facteur saisonnier, min vers le 15 juillet (indice DOY ~196)."""
    return 1.0 + amplitude * np.cos(2.0 * np.pi * (doy - 196) / 365.0)


# --- Génération ------------------------------------------------------------------------------
def generate_synthetic_data(cfg: DataConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)

    idx = pd.date_range(cfg.start_date, cfg.end_date, freq="h", inclusive="both")
    hour = idx.hour.values
    dow = idx.dayofweek.values                     # 0 = lundi
    doy = idx.dayofyear.values
    weekend = (dow >= 5).astype(float)

    # --- Prix Day-Ahead (€/MWh) -------------------------------------------------
    shape = _price_daily_shape()[hour]
    price = (
        cfg.price_base
        * shape
        * (1.0 - 0.15 * weekend)                   # week-end plus calme
        * _seasonal_factor(doy, 0.08)              # hiver légèrement plus cher
        + rng.normal(0.0, cfg.price_noise_std, len(idx))
    )
    # Pics de prix rares (cold snap, effondrement solaire...)
    spike_mask = rng.random(len(idx)) < cfg.price_spike_prob
    spike = cfg.price_spike_factor[0] + rng.random(spike_mask.sum()) * cfg.price_spike_factor[1]
    price[spike_mask] *= spike
    # Quelques heures de prix négatifs (excès de production, réaliste sur EPEX)
    negative_mask = (rng.random(len(idx)) < 0.004) & ~spike_mask
    price[negative_mask] = -cfg.price_base * 0.1 * rng.random(negative_mask.sum())
    price = price.clip(-10.0, 400.0)

    # --- Solaire (per-unit 0..1) ------------------------------------------------
    decl = np.radians(23.45 * np.sin(2.0 * np.pi * (doy - 81) / 365.0))
    lat = np.radians(45.75)                        # latitude type France Sud
    solar_angle = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(np.radians(15.0 * (hour - 12)))
    )
    clear_sky = np.clip(solar_angle, 0.0, None)
    clear_sky = clear_sky / max(clear_sky.max(), 1e-6)

    # Couverture nuageuse : processus AR(1) jour par jour (corrélation entre heures)
    cloud_ar = np.zeros(len(idx))
    noise = rng.normal(0.0, 0.25, len(idx))
    for t in range(1, len(idx)):
        same_day = idx[t].date() == idx[t - 1].date()
        cloud_ar[t] = (0.90 * cloud_ar[t - 1] + noise[t]) if same_day else noise[t]
    cloud = np.clip(0.85 + cloud_ar, 0.15, 1.0)
    solar_pu = (clear_sky * cloud).round(4)

    # --- Consommation (per-unit horaire) -----------------------------------------
    load = (
        _load_daily_shape()[hour]
        * _seasonal_factor(doy, 0.12)
        + rng.normal(0.0, 0.05, len(idx))
    )
    weekend_mask = (dow >= 5) & (hour >= 8) & (hour <= 22)
    load[weekend_mask] *= 0.92                              # week-end plus calme
    load = load.clip(0.4, 1.6).round(4)

    df = pd.DataFrame(
        {PRICE_COL: price.round(2), SOLAR_COL: solar_pu, LOAD_COL: load}, index=idx
    )
    df.index.name = "datetime"
    return df


def save_data(df: pd.DataFrame, processed_dir: str | None = None) -> Path:
    processed = PROJECT_ROOT / (processed_dir or "data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "synthetic_2023.csv"
    df.to_csv(path)
    return path


def load_data(processed_dir: str | None = None) -> pd.DataFrame:
    path = PROJECT_ROOT / (processed_dir or "data/processed") / "synthetic_2023.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Données introuvables : {path}. Lance d'abord `python scripts/generate_data.py`."
        )
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    return df
