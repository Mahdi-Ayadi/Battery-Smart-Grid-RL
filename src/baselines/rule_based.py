"""Baseline 'expert bête' (plan §4.1) : trading pur, sans conscience du réseau.

Stratégie :
  - charge la nuit (heure <= 5 h) quand le prix est bas
  - décharge au pic du soir (18 h-21 h) quand le prix est haut
  - ne regarde JAMAIS les flux du réseau -> génère les congestions attendues.

Sert de référence P&L / Safety pour les livrables 1 et 2.
"""

from __future__ import annotations

import numpy as np

CHARGE_HOURS = set(range(1, 6))    # 01h-05h
DISCHARGE_HOURS = set(range(18, 22))  # 18h-21h


class RuleBasedPolicy:
    """API compatible `model.predict(obs)` de SB3 (mais il ignore l'obs... sauf l'heure)."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError("utiliser `act(hour)` ; l'heure n'est pas dans l'obs.")

    def act(self, hour: int) -> np.ndarray:
        if hour in DISCHARGE_HOURS:
            return np.ones(3, dtype=np.float32)   # vente max
        if hour in CHARGE_HOURS:
            return -np.ones(3, dtype=np.float32)  # charge max
        return np.zeros(3, dtype=np.float32)
