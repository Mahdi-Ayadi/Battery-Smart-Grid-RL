# État technique du projet — Arbitrage Graph-RL (GNN + PPO)

> Document de référence pour reprendre / modifier / planifier. Dernière mise à jour : après le pipeline de fumée complet (env + baseline + GNN + PPO 20k + eval).

## 1. Situation générale

| Composant | État | Preuve |
|---|---|---|
| Données synthétiques 2023 (8737 h) | ✅ Générées | `data/processed/synthetic_2023.csv` |
| Environnement pandapower (DC-OPF/step) | ✅ Validé | Solveur converge 100 %, obs 167 dims |
| Baseline "expert bête" 3 périodes | ✅ Chiffrée | `results/baseline_summary.csv` |
| GNN (GAT + GCN) intégré SB3 | ✅ Validé | forward batché OK, 19.8k params |
| Entraînement PPO bout-en-bout | ✅ Fumée 20k steps | `models/ppo_gnn.zip`, logs `runs/` |
| Livrables graphiques (×3) | ✅ Générés | `results/plots/*.png` |
| Contrôle de version (git) | ❌ À faire | — |
| Tests unitaires | ❌ Absents | smoke tests ad-hoc seulement |
| **Vrai run** (~1M steps) | ⏳ Pas lancé | ~3.5 h CPU à 85 fps |

### Chiffres de référence (le story-board)

| Période | Baseline P&L brut | Pénalités réseau | Heures >100 % | Max ligne |
|---|---|---|---|---|
| train (Jan–Août, 5809 h) | +61 857 € | −715 k€ | 753 (13 %) | 143.7 % |
| val (Sept–Oct, 1464 h) | +15 344 € | −175 k€ | 182 (12 %) | 139.5 % |
| test (Nov–Déc, 1464 h) | +14 464 € | −158 k€ | 182 (12 %) | 137.3 % |

Agent après la fumée (20k steps, val) : **+26 € de P&L, 0 h surchargée, max 90.5 %** → la sécurité
physique est apprise immédiatement ; le trading reste à construire avec le vrai run.
Cible : ≈ +50–60 k€ (train) avec ≈ 0 violation.

## 2. Cartographie du code (où modifier quoi)

```
src/config.py                 ← LE fichier de pilotage (aucune constante codée en dur ailleurs)
  BatteryConfig               bateau essieu : p_max_mw=1.0, 2 MWh, rendements 0.95
  PhysicsConfig               pénalités thermiques (500 €/pu, cap 5000), dim. câbles
                              rating = clip(2.2 × I_pointe × 1.35, 0.04, 0.28) kA
  DataConfig                  signature EPEX synthétique (pic 19h, spikes 1-4×, prix négatifs)
  TrainConfig                 split temporel Ja–Août / Sept–Oct / Nov–Déc, hyperparam PPO

src/data/generate_data.py     BLOC A — 3 colonnes uniquement : price, solar_pu, load_scale
  generate_synthetic_data()   Replace ici pour brancher ENTSO-E / Renewables.ninja
  load_data()                 Lecture CSV ; le reste du code ne connaît que ces 3 colonnes

src/envs/network_builder.py   BLOC B — case33bw + 3 hybrides (buses 17/24/32 = nœuds 18/25/33)
  _assign_line_ratings()      Les limites thermiques recomposées (case33bw n'en a pas)
  build_network()             renvoie (net, maps{"battery","pv"}) ; sgens réglables

src/envs/grid_env.py          BLOC B — SmartGridArbitrageEnv(gym.Env)
  Layout obs (167) : [nodes 33×3 | edges 32×2 | price 1 | soc 3]
      node feats : soc_norm (0 si pas de batterie), solar_pu, load_pu
      edge feats : capacité relative du câble, charge courante (loading/limit)
  _device_power()             action [-1,1]³ → P_charge/P_décharge cherchées + clamp SoC
  _apply_state()              injection réseaux : sgens + net.load["scaling"]
  _run_powerflow()            pp.rundcpp() ; renvoie False si non-convergence
  step()                      R = profit − min(pénalité, cap) ; info={profit, thermal_penalty, ...}
  ⚠ Convention : seul le PRIX est standardisé (z-score) ; solar/load restent en per-unit bornés.

src/models/gnn_extractor.py   BLOC C — GNNFeatureExtractor(BaseFeaturesExtractor)
  obs plate → graphe → GATConv×2 (edge_dim=2) → mean+max pool → dense → (+price,+soc) → 32
  Option GCNConv : features d'arêtes injectées dans le nœud aval (concat).

src/baselines/rule_based.py   charge 01-05h / décharge 18-21h ; ignore l'obs réseau

scripts/  generate_data · run_baseline · train · eval  (thin wrappers, CLI argparse)
  train.py --fast             20k steps / 4 envs pour valider le pipeline
  train.py --conv-type gcn    ablation GAT vs GCN
  eval.py --model --period    génère les 3 graphiques + JSON de métriques
```

**Invariants à connaître avant de toucher** :
1. L'ordre de l'obs est partagé entre `grid_env.py` (constantes `NODE_SECTION`, `EDGE_SECTION`…)
   et `gnn_extractor.py` qui les importe — changer l'un casse l'autre si on ne garde pas l'import.
2. L'assertion 32 lignes porte sur les lignes **ACTIVES** (`in_service`), les 5 tie-lines hors
   service de case33bw sont ignorées partout.
3. `random_start=True` (train) vs `False` (eval) : les épisodes d'entraînement démarrent à une
   heure aléatoire, l'évaluation est déterministe sur la période complète.
4. `info["profit"]` = P&L financier **brut** ; `info["reward"]` = signal RL (brut − pénalité).
   Les graphiques livrables 1 & 2 consomment `profit` / `max_loading_pct`.

## 3. Limites connues (à assumer ou corriger)

### Méthodologie / physique
- **Flux DC** : pas de réactif, pas de tension, pas de pertes. Les violations de tension
  (le vrai symptôme de l'IEEE 33 : fin de feeder à 0.91 pu) ne sont pas contraintes. OK pour
  le livrable thermique du plan, mais une extension AC (`runpp`) est un axe tracé.
- **Ratings synthétiques** : le goulot de 0.04 kA en bout de feeder est *construit* pour rendre
  la congestion apprenable (~115 % en pleine puissance). Ce n'est pas une donnée DSO réelle —
  à documenter dans tout livrable R&D.
- **Pas de surdimension saisonnier des prix** : la signature EPEX synthétique est trop
  régulière (pic 19h constant). Le couplage réel gaz/soleil/cold-snap manque → l'agent risque
  d'apprendre une heuristique calendaire (heure → action) plutôt qu'une vraie stratégie.
  Les spikes aléatoires atténuent un peu ce risque.
- **Split temporel biaisé** : train = Jan–Août (dominé par l'été), test = Nov–Déc (hiver).
  Covariate shift probable. Une séparationISO par années (2023 train / 2024 test) est
  la vraie réponse, en attente des données réelles.
- **Site unique solaire** : les 3 sites partagent le même profil × facteurs (1.0/0.95/1.05).

### RL / optimisation
- **Échelle de récompense** : rewards de −1e5, value_loss ~1.6e7 → le critic démarre sur des
  valeurs énormes. Risque : convergence lente. Levier nº1 avant le vrai run : `VecNormalize`
  (norm reward/obs) ou baisser `thermal_penalty` (500 → 150) + `max_penalty`.
- **Horizon épisode = 5.8k steps** (8 mois de données en continu avec `random_start`) alors que
  l'horizon utile d'un cycle batterie ≈ 24–48 h et γ=0.99 ≈ 100 steps. Un épisode de 1–7 jours
  (fenêtre glissante) rendrait le crédit de retour beaucoup plus net → meilleure propagation
  du signal de congestion.
- **Pas d'incertitude** : prix connu à J-1, PV certain. L'arbitrage DA réel a une part
  d'incertitude (PV forecast) qui différencierait vraiment l'agent d'un solveur déterministe.
- **Un seul extracteur "graphe" testé** : sans un ablateur MLP-only (pas de GNN), la
  démonstration "le graphe apporte quelque chose" ne tient pas. `conv_type="mlp"` à ajouter.

### Ingénierie
- **Pas de git** — priorité nº1.
- **Structure doublée** : le projet vit dans `...\Battery Smart Grid RL\Battery-Smart-Grid-RL\`
  (déplacement fortuit). À aplatir avant tout commit.
- **Pas de tests** : les invariants ci-dessus devraient devenir 4-5 tests pytest (shape obs,
  clamp SoC, une step avec surcharge attendue, forward GNN).
- Vitesse 85 fps (CPU) : suffisant pour 1M (3.5 h). Multiplier les envs (SubprocVecEnv,
  numba déjà installé) si on vise 5–10M.

## 4. Prochaines étapes recommandées (ordre)

**P1 — Avant le vrai run (quelques heures de travail)**
1. `git init` + aplatir le dossier + commit initial (README = documentation du design).
2. Ajouter `conv_type="mlp"` (extracteur sans graphe) dans `gnn_extractor.py` — c'est
   l'ablation qui prouve la valeur du GNN.
3. Stabiliser le signal : `VecNormalize` dans `train.py` (ou pénalité × 0.3) + épisodes
   courtes échantillonnées-fenêtres (ex : 168 h aléatoires) avec réinitialisation SoC.
4. Lancer le vrai run 1M steps → `train.py`, suivre TensorBoard : `ep_rew_mean` ↑ et
   `% heures surchargées` (via callback maison) ↓.

**P2 — Itérations R&D (les vrais livrables)**
5. Ablations : GAT vs GCN vs MLP ; sensibilité `thermal_penalty` (courbe P&L vs violations).
6. Optim `Optuna` : lr, ent_coef, gamma, n_heads. Grille courte (20 trials × 200k steps).
7. Données réelles ENTSO-E + Renewables.ninja (même schéma CSV, code inchangé).
8. Éval 2024 hors-échantillon + 3 seeds → intervalles de confiance sur les livrables.

**P3 — Extensions (si le socle tiendra)**
9. Contraintes de tension (AC power flow, `vm_pu` borné) ou batteries à différents buses.
10. Intraday / imbalance (2 actions par heure) : là où le RL battra un solveur DP.
11. Comparaison avec un MPC/DP déterministe (oracle) — la référence académique qu'un RL
    doit battre pour être publiable.

## 5. How-to rapide

```powershell
# tout retenter de zéro
.venv\Scripts\python.exe scripts\generate_data.py
.venv\Scripts\python.exe scripts\run_baseline.py
.venv\Scripts\python.exe scripts\train.py --fast          # pipeline OK ?
.venv\Scripts\python.exe scripts\train.py                 # vrai run
.venv\Scripts\python.exe scripts\eval.py --model models/ppo_gnn --period test
```
Encodage Windows (cosmétique) : `chcp 65001` avant les scripts si les accents s'affichent mal.
