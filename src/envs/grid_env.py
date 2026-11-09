"""BLOC B — Environnement Gymnase wrapper de pandapower.

Un `step()` correspond à 1 heure du marché Day-Ahead :
  1. L'agent choisit l'action de charge/décharge des 3 batteries ([-1,1]^3).
  2. On borne physiquement la puissance par l'état de charge (SoC).
  3. On injecte tout dans le réseau, le solveur DC (rundcpp) calcule les flux.
  4. Récompense = Profit d'arbitrage − Pénalités thermiques (Physics-Informed).

Format d'observation (vectorisé à plat, pour rester compatible SB3 tout en
permettant au GNN de reconstruire le graphe) :
  [ nodes (33 x 3) | edges (32 x 2) | price (1) | soc_global (3) ] = 167 dims.

Note de design SoC : plutôt que de pénaliser une décharge à SoC=0 (plan §2),
on **clamp** mécaniquement la puissance à la borne physique avant injection.
L'agent n'explore donc pas d'actions incohérentes : l'apprentissage est plus
stable, et la contrainte batterie reste respectée par construction.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import pandapower as pp
from gymnasium import spaces

from src.config import BATTERY_BUSES, Config
from src.envs.network_builder import build_network

# --- Constantes topologiques et format d'observation ---------------------------
N_NODES = 33
N_LINES = 32
NODE_FEATS = 3   # [soc_norm, solar_pu, load_pu] par nœud
EDGE_FEATS = 2   # [capacité_normée, charge_courante (flux/limite)] par arête
N_SITES = len(BATTERY_BUSES)

NODE_SECTION = N_NODES * NODE_FEATS                 # = 99
EDGE_SECTION = NODE_SECTION + N_LINES * EDGE_FEATS  # = 163
PRICE_IDX = EDGE_SECTION                            # = 163
OBS_SIZE = EDGE_SECTION + 1 + N_SITES               # = 167


class SmartGridArbitrageEnv(gym.Env):
    """Arbitrage Day-Ahead d'une flotte de 3 batteries sous contraintes réseau."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        random_start: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(data.index, pd.DatetimeIndex)

        self.cfg = config
        self.batt = config.battery
        self.phys = config.physics
        self.df = data.sort_index()
        self.n_steps = len(self.df)
        self.random_start = random_start

        # Séries mémoire (accès rapide)
        self.price_raw = self.df["price"].to_numpy(dtype=np.float64)
        mu, sigma = self.price_raw.mean(), max(self.price_raw.std(), 1e-6)
        self.price_norm = ((self.price_raw - mu) / sigma).astype(np.float32)
        self.solar = self.df["solar_pu"].to_numpy(dtype=np.float64)
        self.load_scale = self.df["load_scale"].to_numpy(dtype=np.float64)

        self.dt = self.batt.dt_h
        self.capacity = self.batt.capacity_mwh  # MWh

        # --- Réseau ------------------------------------------------------------
        net, maps = build_network(config)
        self.net = net
        self.maps = maps
        self.batt_sgen_idx = np.array(list(maps["battery"].values()))
        self.pv_sgen_idx = np.array(list(maps["pv"].values()))

        assert len(net.bus) == N_NODES, "attendu IEEE 33-bus"
        # 32 lignes ACTIVES (case33bw contient aussi 5 lignes d'interconnexion H.S.)
        self.active_line_idx = net.line.index[net.line.in_service].to_numpy(dtype=np.int64)
        assert len(self.active_line_idx) == N_LINES, "attendu 32 lignes actives"

        lines = net.line.loc[self.active_line_idx]
        self.edge_from = lines["from_bus"].to_numpy(dtype=np.int64)
        self.edge_to = lines["to_bus"].to_numpy(dtype=np.int64)
        caps = lines["max_i_ka"].to_numpy(dtype=np.float64)
        caps = np.nan_to_num(caps, nan=float(np.nanmax(caps))).clip(min=1e-6)
        # Feature arête 1 : capacité relative du câble (per-unit du plus gros)
        self.edge_cap_norm = (caps / caps.max()).astype(np.float32)
        self.last_loading = np.ones(N_LINES, dtype=np.float64)

        # Charge de base par bus (profils IEEE 33) -> feature "consommation" nœud
        per_bus = net.load.groupby("bus")["p_mw"].sum()
        base = per_bus.reindex(range(N_NODES)).fillna(0.0).to_numpy(dtype=np.float64)
        self.base_load_pu = (base / base.max()).astype(np.float32)

        # --- Espaces -------------------------------------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_SITES,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    # ------------------------------------------------------------------ utils --
    @property
    def edge_index_list(self) -> list[tuple[int, int]]:
        """Topologie (from, to) pour l'extracteur GNN."""
        return list(zip(self.edge_from.tolist(), self.edge_to.tolist()))

    def _device_power(self, action) -> tuple[np.ndarray, np.ndarray]:
        """Mappe l'action [-1,1]^3 en puissances cohérentes avec le SoC."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        p_dis_des = np.maximum(a, 0.0) * self.batt.p_max_mw
        p_ch_des = np.maximum(-a, 0.0) * self.batt.p_max_mw
        # Bornes physiques dictées par l'état de charge de CHAQUE batterie :
        #   charge : soc + p_ch*eff*dt <= capacity
        #   décharge : soc - p_dis/eff*dt >= 0
        p_ch_cap = np.maximum(
            (self.capacity - self.soc) / (self.batt.eff_charge * self.dt), 0.0
        )
        p_dis_cap = np.maximum(self.soc * self.batt.eff_discharge / self.dt, 0.0)
        p_ch = np.minimum(p_ch_des, p_ch_cap)
        p_dis = np.minimum(p_dis_des, p_dis_cap)
        # Mise à jour du SoC (rendements)
        self.soc = np.clip(
            self.soc
            + (p_ch * self.batt.eff_charge - p_dis / self.batt.eff_discharge) * self.dt,
            0.0,
            self.capacity,
        )
        return p_ch, p_dis

    def _apply_state(
        self, battery_p_mw: np.ndarray, solar_pu: float, t: int
    ) -> None:
        """Positionne sgens (batteries/solaire) + scaling des charges du step t."""
        self.net.sgen.loc[self.batt_sgen_idx, "p_mw"] = battery_p_mw  # déch - ch
        pv = max(solar_pu, 0.0) * self.phys.solar_max_mw
        factors = np.asarray(self.phys.pv_node_factors, dtype=np.float64)
        self.net.sgen.loc[self.pv_sgen_idx, "p_mw"] = pv * factors
        self.net.load["scaling"] = float(
            self.load_scale[t] * self.phys.load_multiplier
        )

    def _run_powerflow(self) -> bool:
        """Solveur DC du pandapower (validation des lois de Kirchhoff)."""
        try:
            pp.rundcpp(self.net)
            # res_line est indexé comme net.line -> ne garder que les lignes actives
            loading = (
                self.net.res_line.loc[self.active_line_idx, "loading_percent"]
                .to_numpy(dtype=np.float64)
            )
            self.last_loading = (
                np.nan_to_num(loading, nan=self.phys.thermal_limit_pct)
                / self.phys.thermal_limit_pct
            )
            return True
        except Exception:
            return False  # on conserve le dernier état de flux coherant

    def _current_obs(self, t: int | None = None) -> np.ndarray:
        t = self.t if t is None else t
        nodes = np.zeros((N_NODES, NODE_FEATS), dtype=np.float32)

        # Nœud — SoC : 0 pour les 30 nœuds sans batterie (cf. plan)
        soc_norm = (self.soc / self.capacity).astype(np.float32)
        nodes[BATTERY_BUSES, 0] = soc_norm

        # Nœud — Production solaire locale des sites hybrides
        nodes[BATTERY_BUSES, 1] = np.float32(self.solar[t])

        # Nœud — Consommation locale (base IEEE33 x facteur horaire)
        nodes[:, 2] = self.base_load_pu * np.float32(self.load_scale[t])

        edges = np.stack(
            [
                self.edge_cap_norm,
                np.clip(self.last_loading, 0.0, 2.0).astype(np.float32),
            ],
            axis=1,
        )

        obs = np.concatenate(
            [
                nodes.ravel(),                                 # 99
                edges.ravel().astype(np.float32),              # 64
                np.array([self.price_norm[t]], np.float32),    # 1
                soc_norm,                                      # 3
            ]
        )
        return obs.astype(np.float32)

    def _info(self, t: int, p_dis: np.ndarray, p_ch: np.ndarray,
              reward: float, converged: bool,
              profit: float = 0.0, penalty: float = 0.0) -> dict:
        return {
            "reward": reward,          # profit - pénalité thermique (signal RL)
            "profit": profit,          # P&L financier brut de l'arbitrage (€)
            "thermal_penalty": penalty,
            "p_dis_mw": p_dis,
            "p_ch_mw": p_ch,
            "soc_mwh": self.soc,
            "price_raw": self.price_raw[t],
            "solar_pu": float(self.solar[t]),
            "max_loading_pct": float(self.last_loading.max() * self.phys.thermal_limit_pct),
            "n_lines_over_limit": int((self.last_loading > 1.0 + 1e-9).sum()),
            "converged": bool(converged),
            "datetime": self.df.index[t],
        }

    # -------------------------------------------------------------- API Gym --
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.random_start:
            self.t = int(self.np_random.integers(0, max(self.n_steps - 2, 1)))
        else:
            self.t = 0
        self.soc = np.full(N_SITES, self.batt.soc_init * self.capacity)
        self._apply_state(np.zeros(N_SITES), self.solar[self.t], self.t)
        self._run_powerflow()
        obs = self._current_obs()
        info = self._info(self.t, np.zeros(N_SITES), np.zeros(N_SITES), 0.0, True)
        return obs, info

    def step(self, action):
        p_ch, p_dis = self._device_power(action)
        t = self.t

        self._apply_state(p_dis - p_ch, self.solar[t], t)  # injection nette batterie
        converged = self._run_powerflow()

        if converged:
            over = np.maximum(0.0, self.last_loading - 1.0)  # per-unit de la limite
            penalty_raw = self.phys.thermal_penalty * (
                over.sum() + self.phys.thermal_quad_factor * np.square(over).sum()
            )
            penalty = float(min(penalty_raw, self.phys.max_penalty))
            price = self.price_raw[t]  # €/MWh connu à J-1 (Day-Ahead)
            profit = float((p_dis.sum() - p_ch.sum()) * price * self.dt)
            reward = profit - penalty
        else:
            profit, penalty = 0.0, 0.0
            reward = float(self.phys.flow_fail_penalty)

        self.t = t + 1
        truncated = self.t >= self.n_steps
        terminated = False
        obs = self._current_obs(min(self.t, self.n_steps - 1))
        info = self._info(t, p_dis, p_ch, reward, converged, profit, penalty)
        return obs, float(reward), terminated, truncated, info
