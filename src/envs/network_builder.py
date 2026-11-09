"""BLOC B — Construction du réseau physique (IEEE 33-bus + systèmes hybrides).

Le réseau IEEE 33-bus est un feeder de distribution 12.66 kV radial :
33 bus, 32 lignes actives, ~3.7 MW de charge totale. C'est le banc d'essai
standard pour les études de congestion en distribution.

On y greffe 3 systèmes hybrides (Solaire + Batterie) aux extrémités du réseau
(nœuds 18, 25, 33 du plan = buses 17, 24, 32 pandapower), là où les congestions
sont les plus critiques.

Constat de principe (vérifié empiriquement) :
  - `case33bw` contient 37 lignes dont 5 lignes d'interconnexion HORS service ;
  - aucune limite thermique n'est définie (max_i_ka = 99999).
On dimensionne donc les câbles de façon réaliste :
      rating = clip(headroom * I_de_base * facteur_pointe, min, max)
  -> les câbles proches des extrémités (~40 A) deviennent le goulot :
     une batterie pleine puissance (~45 A à 12.66 kV) les sature à ~115 %,
     tandis que le tronc de feeder (~0.28 kA) sert de limite structurelle.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from src.config import BATTERY_BUSES, Config


def _assign_line_ratings(net: pn.pandapowerNet, config: Config) -> None:
    """Dimensionne max_i_ka des lignes actives d'après leur flux de pointe."""
    phys = config.physics
    active_idx = net.line.index[net.line.in_service].to_numpy()

    backup_scaling = net.load["scaling"].copy()
    net.load["scaling"] = 1.0
    pp.rundcpp(net)
    i_base = (
        net.res_line.loc[active_idx, "i_ka"].to_numpy(dtype=np.float64).clip(min=1e-9)
    )
    net.load["scaling"] = backup_scaling

    i_peak = i_base * phys.rating_peak_load_factor
    rating = np.clip(
        phys.rating_headroom * i_peak,
        phys.rating_min_ka,
        phys.rating_max_ka,
    )
    net.line.loc[active_idx, "max_i_ka"] = rating


def build_network(config: Config) -> tuple[pn.pandapowerNet, dict]:
    """Construit le réseau et renvoie (net, maps).

    maps["battery"][i] : index de ligne du sgen batterie du site i (0,1,2)
    maps["pv"][i]      : index de ligne du sgen solaire du site i
    La batterie est modélisée comme un sgen à puissance réglable :
      p_discharge - p_charge  (MW), la charge étant une consommation négative.
    """
    net = pn.case33bw()

    # Slack garantit (case33bw en possède un au bus 0)
    if len(net.ext_grid) == 0:
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)

    maps: dict = {"battery": {}, "pv": {}}
    for i, bus in enumerate(BATTERY_BUSES):
        maps["battery"][i] = pp.create_sgen(
            net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"bess_site_{i}",
            type="BESS", controllable=False,
        )
        maps["pv"][i] = pp.create_sgen(
            net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"pv_site_{i}",
            type="PV", controllable=False,
        )

    _assign_line_ratings(net, config)
    return net, maps
