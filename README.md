# Arbitrage de marché sous contraintes physiques via Graph-RL (GNN + PPO)

Agent autonome qui maximise la valorisation (arbitrage EPEX Day-Ahead) d'une
flotte de 3 batteries branchées sur un feeder IEEE 33-bus, **sans jamais
surcharger thermiquement les lignes**. Le GNN donne à l'agent PPO une
"conscience spatiale" du réseau pour anticiper les congestions.

> Plan détaillé : `docs/PLAN.md`

## Démarrage rapide

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\tensorboard --logdir runs           # (pendant l'entraînement)

# 1. Données synthétiques 2023 (BLOC A)
python scripts/generate_data.py        # NB : lancer avec le python du venv

# 2. Baseline 'expert bête' (P&L + congestions de référence)
python scripts/run_baseline.py

# 3. Entraînement PPO + GNN (BLOC C)
python scripts/train.py                         # ~1M timesteps (défaut)
python scripts/train.py --fast                  # fumée : 20k steps

# 4. Évaluation + 3 livrables graphiques
python scripts/eval.py --model models/ppo_gnn --period test
```

## Structure

```
src/
  config.py               # tous les hyperparamètres (batteries, pénalités, PPO)
  data/generate_data.py   # BLOC A : données synthétiques (prix/solaire/conso)
  envs/network_builder.py # BLOC B : IEEE 33-bus + 3 hybrides (17, 24, 32)
  envs/grid_env.py        # BLOC B : Gym env, DC-OPF (rundcpp) à chaque step
  models/gnn_extractor.py # BLOC C : GNN -> BaseFeaturesExtractor SB3
  baselines/rule_based.py # Baseline sans IA
scripts/
  generate_data.py · run_baseline.py · train.py · eval.py
data/processed/           # CSV horaire 2023
results/                  # métriques JSON + graphiques livrables
runs/                     # logs TensorBoard
models/                   # checkpoints SB3 (.zip)
```

## Design decisions clés

- **Observation à plat, graphe reconstruit par le GNN** :
  `[nodes (33×3) | edges (32×2) | price | SoC global (3)] = 167 dims`.
  Compatible nativement avec `MlpPolicy` de SB3 ; l'extracteur redécoupe en
  graphe et applique 2 convolutions (GATConv par défaut, GCNConv en option
  `--conv-type gcn`), puis mean/max pooling puis concat prix + SoC.
- **Actions bornées physiquement (SoC)** : la puissance est clampée par
  l'état de charge avant injection plutôt que pénalisée : l'agent n'explore
  pas d'actions incohérentes, la contrainte batterie est respectée par
  construction. La pénalité thermique, elle, reste 'Physics-Informed'
  (`R = profit − pénalité(lignes > 100 %)`).
- **Dimensionnement des câbles** : `case33bw` n'a pas de limites thermiques ;
  elles sont recomposées : `rating = clip(2.2 × I_base × 1.35, 0.04, 0.28) kA`.
  Une batterie pleine puissance (~45 A à 12.66 kV) sature les câbles des
  extrémités à ~115 % -> l'agent doit apprendre à écrêter ses ventes.
- **Données synthétiques** compatibles ENTSO-E / Renewables.ninja : mêmes
  colonnes (`price`, `solar_pu`, `load_scale`), mêmes index horaire. Pour passer
  aux données réelles, remplacer le CSV — ni l'env ni l'agent ne changent.

## Roadmap

1. Vrais prix ENTSO-E (token API) +PROFILE Renewables.ninja 2023/2024.
2. Comparaison GAT vs GCN (ablation), n_heads, profondeur.
3. Commutation de l'épisode : basculement `random_start` pour les rollouts.
4. Contraintes Q/voltage (AC power flow) en 2e étape.
5. Multi-agent (3 politiques indépendantes) ou méta-flotte.
