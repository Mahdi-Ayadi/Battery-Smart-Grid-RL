#!/usr/bin/env python
""" Génère et sauvegarde le jeu de données synthétique 2023 (BLOC A)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG
from src.data.generate_data import generate_synthetic_data, save_data


def main() -> None:
    cfg = DEFAULT_CONFIG
    df = generate_synthetic_data(cfg.data)
    path = save_data(df)
    print(f"OK : {len(df)} heures générées -> {path}")
    print(df.describe().loc[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
