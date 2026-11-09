# Architecture et fonctionnement du code — expliqué pas-à-pas

> Objectif de ce document : qu'à la fin de sa lecture, tu saches **ce que fait
> chaque fichier, comment, et pourquoi** — sans avoir à ouvrir le code.
> C'est le complément didactique de `docs/ETAT_TECHNIQUE.md` (où l'on est, vers où aller).

---

## 1. L'idée du projet en une phrase, puis la vue d'ensemble

**En une phrase :** on veut qu'une flotte de 3 batteries gagne de l'argent en
achetant de l'électricité quand c'est bon marché et en revendant quand c'est
cher, mais SANS faire fondre les câbles du quartier auquel elles sont branchées.

**L'image fil conducteur à garder en tête :**

Le réseau électrique local d'un quartier, c'est comme un système de canalisations
d'eau : un arbre de « tuyaux » (les lignes) de grosseurs différentes, qui part du
poste de transformation (la racine) et se ramifie vers chaque
consommateur/producteur. L'eau qui circule ne peut pas choisir son chemin : à
chaque embranchement, elle se répartit selon la géométrie des « tuyaux ». Si on
pompe trop d'un bout du réseau pour le réinjecter à l'autre, certains tuyaux
transportent alors plus que leur capacité ne le permet (« le câble chauffe »).

- **Le métier de l'agent** : décider combien charger/décharger chaque batterie,
  heure par heure, en flairant à la fois le marché ET les « tuyaux ».
- **Notre outil** : un réseau de neurones qui « voit » la géométrie du réseau
  (un graphe) — c'est le GNN — combiné à un algorithme d'apprentissage par
  renforcement (PPO).

Trois empilements logiciels, qui se recouvrent exactement :

```
  BLOC A   les "données du monde"   : prix, soleil, consommation (2023, 8737 heures)
  BLOC B   la "maquette physique"   : le réseau des câbles canalisations + les batteries
  BLOC C   le "cerveau"             : réseau de neurones qui regardent le graphe + PPO
```

---

## 2. Schéma des dépendances — qui parle à qui

```
scripts/*.py                  (lignes de commande : "génère","entraîne","évalue")
    │
    ├──► src/config.py        LA table de mixage : TOUS les chiffres de reglages
    │
    ├──► src/data/generate_data.py    BLOC A
    │         │  un tableau horaire : price / solar_pu / load_scale
    │         ▼
    └──► src/envs/
          network_builder.py         BLOC B — monte la maquette du réseau (câbles, bus)
              │  renvoie (net, maps)
              ▼
          grid_env.py                BLOC B — SmartGridArbitrageEnv(gym.Env)
              │  expose obs(167) / action(3) / reward / info{}
              ▼
    src/models/gnn_extractor.py      BLOC C — le regard « graphe » du cerveau (torch/PyG)
              │  importe le format de l'obs défini dans grid_env   ← COUPLAGE VOLONTAIRE
              ▼
    stable_baselines3.PPO            le chef d'orchestre de l'apprentissage
              ▼
    runs/ (TensorBoard) · models/ (.zip) · results/ (csv, json, png)
```

Les deux endroits où les fichiers « se tiennent par la main » :
1. **L'observation** : `grid_env.py` définit QUEL ordre on empile les nombres
   dans l'observation (167 chiffres). `gnn_extractor.py` ne devine pas cet
   ordre — il l'importe de `grid_env.py`. Donc si un jour tu changes les
   features (ajoute la température, les prix D+1…), tu changes `grid_env.py`
   et le GNN suit automatiquement (toutefois, les reshapes internes doivent
   rester logiquement corrects).
2. **Le Python** : les scripts font `sys.path.insert(0, racine)` pour pouvoir
   écrire `from src...`. Le package n'est pas installé proprement (pas de
   setup.py) — simple mais suffisant à ce stade.

---

## 3. Fiche par fichier

Chaque fiche répond aux mêmes questions : rôle / fonctionnement intérieur /
pourquoi ce choix / où intervenir.

### 3.1 `src/config.py` — la table de mixage

**Rôle.** Dans TOUT le projet, aucun chiffre de réglage n'est codé en dur
« ailleurs » : la puissance des batteries, la sévérité des pénalités, les
hyperparamètres PPO… tout est ici, dans 4 groupes.

| Groupe | Ce que ça représente | Champs et valeurs par défaut |
|---|---|---|
| `BatteryConfig` | la batterie type (les 3 sont identiques) | `p_max_mw=1.0` (MW), `capacity_mwh=2.0`, soc initial 50 %, rendements 0.95, `dt_h=1.0` |
| `PhysicsConfig` | les « lois de la physique » du jeu | limite thermique 100 %, pénalité 500 €/point de surcharge (+ terme quadratique), plafond 5000 €, pénalité solveur −2000 €, solaire 0.6 MW max, dimensionnement câbles |
| `DataConfig` | la fabrique des données | fenêtre 2023, prix de base 45 €/MWh, intensité du bruit, probabilité des pics, seed 42 |
| `TrainConfig` | le protocole d'apprentissage | découpage Jan–Août / Sept–Oct / Nov–Déc ; PPO : 1M steps, 8 envs, taux d'apprentissage 3e-4, etc. |

**Pourquoi ce choix.** Les dataclasses rendent l'expérience reproductible : tu
recopies `config.py` dans une variante (ex. `config_v2.py`) et tu compares les
runs — c'est un journal scientifique minimal.

**Où intervenir.** Vite fait bien fait :
- Une batterie plus grosse ? → `BatteryConfig.p_max_mw / capacity_mwh`
- Un réseau plus congestible ? → jouer sur `rating_min_ka` (0.040) qui
  est le facteur-clef de la sévérité de la congestion.
- Sévérité vs rendement financier ? → `thermal_penalty` (500) : c'est
  sa valeur qui calcule combien l'agent préfère « respecter le réseau »

⚠ Deux pièges :
1. `BATTERY_BUSES = (17, 24, 32)` : le plan parle des « nœuds 18, 25, 33 »
   en comptant à partir de 1 ; pandapower compte à partir de 0 donc c'est
   17, 24, 32 — c'est RÉGLÉ ici une fois pour toutes.
2. `DEFAULT_CONFIG` est un **singleton muté** par les options `--fast` :
   attention si tu écris des systèmes d'entrainements parallèles qui
   partagent cette config.

### 3.2 `src/data/generate_data.py` — BLOC A : la fabrique du monde

**Rôle.** Créer un fichier CSV horaire qui imite ce que fourniront les
données réelles. Le code entier d'après ne connaît QUE ce fichier.

**Ce que le fichier contient (le « contrat ») — 3 colonnes seulement :**

| Colonne | Unité / plage | signifie |
|---|---|---|
| `price` | €/MWh, −10 à 400 | le prix du marché (Day-Ahead) de cette heure |
| `solar_pu` | 0 à 1 | facteur de capacité du soleil (1 = panneaux à pleine puissance) |
| `load_scale` | ~0.4 à 1.6 | « à cette heure le quartier consomme 1.28× sa valeur de base » |

Avec l'index `datetime` horaire de 2023 : de 01/01 00 h à 31/12 00 h, ce qui
donne 365 × 24 + 1 = 8737 lignes (bornes incluses).

**Comment les 3 séries sont générées (algorithmiquement) :**

- **Prix** : une forme de base qui descend la nuit (×0.58 vers 4 h) et
  monte en pic du soir (×1.80 vers 19 h — c'est là que les batteries font
  l'argent), ×0.85 le weekend, un peu plus cher l'hiver, + bruit
  gaussien N(0, 8 €). À cela s'ajoutent deux évènements : des **pics rares**
  (1.2 % des heures, le prix est ×1.8 à ×4 — un cold snap) et quelques
  **heures à prix négatifs** (0.4 % — trop d'énergie produite). Le tout est
  coupé entre −10 et 400 €.
- **Solaire** : un vrai calcul d'astronomie simplifié (déclinaison du soleil
  selon le jour de l'année et latitude 45.75°) qui donne l'ensoleillement
  théorique, multiplié par un coefficient de **nuages**
  modélisé par un petit processus auto-régressif (la couverture d'une heure
  ressemble à celle de l'heure précédente — réaliste).
- **Consommation** : une courbe de vie d'un quartier (0.66 la nuit → 1.32 au
  pic de 20h quand tout le monde rentre) × facteur saisonnier et ×0.92 le
  weekend.

**Pourquoi ce choix.** Les 3 formes se recoupent volontairement : la
batterie peut acheter la nuit/tôt le matin et vendre au pic de 19h. Mais les
pics aléatoires, le week-end et le soleil variable empêchent d'en faire une
règle figée « la nuit → acheter ». Sinon l'agent apprendrait juste une horloge.

**Où intervenir.** Pour changer la forme des prix (si tu connais la vraie
courbe EPEX), modifie `_price_daily_shape()`. Et précision importante : le
futur module `src/data/ingest/` (données réelles) ne remplacera pas ce
fichier — il écrira juste un CSV respectant le même contrat 3 colonnes.

### 3.3 `src/envs/network_builder.py` — BLOC B : monter la maquette électrique

**Rôle.** Construire, une fois, le « plateau de jeu » : le réseau (les câbles
et leurs limites) + les 3 stations solaire+batterie greffées dessus.

**Comment ça marche concrètement :**

1. `pn.case33bw()` : on charge « l'IEEE 33-bus », le banc d'essai standard
   mondial pour la distribution. 33 « nœuds » (buses — des points de connexion,
   pas des équipements physiques), connectés par 37 câbles dont 32 sont en
   service (réseau radial, comme un arbre) et 5 sont des « bretelles de secours »
   normalement ouvertes (`in_service=False`) qu'on ignore partout.
2. Si le réseau n'a pas de point d'équilibre (source), on en crée un au nœud 0
   (le poste source relié par l'`ext_grid` — le « grand canal » qui apporte
   l'électricité
   du reste du pays).
3. On greffe, sur CHACUN des 3 nœuds d'extrémité (17, 24, 32), deux producteurs
   consommateurs virtuels appelés `sgen` :
   - un **PV** : injection égale à `solar_pu × 0.6 MW` (toujours ≥ 0) ;
   - une **batterie** : injection = (puissance déchargée − puissance chargée),
     donc **négative** quand elle charge (elle consomme du réseau) et positive
     quand elle décharge (elle injecte). Un seul équipement, deux usages.
4. **Le problème trouvé et résolu** : le réseau IEEE 33-bus donne les câbles
   avec `max_i_ka = 99999` kA — soit « pas de limite ». Sans limites, jamais de
   congestion ! On les reproduit donc de façon physiquement plausible :

   ```
   I_base   = courant mesuré dans chaque câble à charge moyenne (flux DC)
   I_pointe = I_base × 1.35                              (le "pic" du soir)
   rating   = clip(2.2 × I_pointe, 0.040 kA, 0.280 kA)
   ```

   Traduction : chaque câble reçoit une marge d'exploitation de 2.2× sa
   charge de pointe, mais jamais moins que 0.040 kA.
   À 12.66 kV : 0.040 kA ≈ 0.88 MVA de puissance transportable. Une batterie
   de 1 MW ⇔ 0.0455 kA ⇒ un câble au goulot est saturé à ~114 % ! C'est
   exactement le mécanisme du plan : les congestions naissent aux extrémités.
   ⚠ À savoir : ce calibrage est un **choix de design**, pas une donnée
   du réseau réel — c'est ce qui rend la congestion apprenable et il doit
   être documenté dans tout livrable R&D.

   De l'autre côté, la « tête de feeder » (les premiers câbles près du poste)
   est à 0.28 kA ≈ 6 MVA — bien au-dessus de ce qu'une batterie de 1 MW
   déforme seule : les batteries seules ne cisaillent pas la structure du
   réseau, seule l'agrégation crée la congestion.

**Où intervenir.** C'est ici qu'on modifie la topologie (déplacer les
batteries, changer le réseau…). Les 3 facteurs du dimensionnement
(`rating_*` dans `PhysicsConfig`) finiront probablement dans ton plan
d'expériences « congestion vs rentabilité ».

### 3.4 `src/envs/grid_env.py` — BLOC B : la salle de jeu (le cœur du projet)

**Rôle.** Traduire le monde physique en langage standard d'apprentissage par
renforcement (API « Gymnasium ») : l'agent reçoit une **observation**, propose
une **action**, reçoit une **récompense** — et un pas = une heure du marché.

**Une heure de marché racontée — le déroulé du `step()` :**

1. **L'agent propose** une action `a = [a1, a2, a3]`, chaque nombre entre −1
   et +1 (−1 charge maximum, +1 décharge maximum, 0 on ne fait rien).
2. **On traduit en mégawatts** (`_device_power`) :
   `P_décharge_désirée = max(a, 0) × 1.0 MW` et `P_charge_désirée = max(−a, 0) × 1.0 MW`.
3. **Le garde-fou mécanique** — le clamp SoC. Une batterie ne peut pas
   encaisser plus que ce qu'il reste, c'est impossible physiquement : cette
   **limite vient de la capacité**, pas d'une règle du jeu arbitraire. Avant
   d'injecter, on réduit la demande :
   - pour charger : maximum intégrable = `(2.0 MWh − SoC) / (0.95 × 1 h)` ;
   - pour décharger : maximum extractable = `SoC × 0.95 / 1 h`.
   Puis le SoC est mis à jour avec les rendements (on perd 5 % à chaque
   allée-retour) : `SoC' = SoC + (P_ch·0.95 − P_dis/0.95)×1h`, borné [0, 2 MWh].
   **Design assumé** : le plan prévoyait de PÉNALISER une décharge à SoC=0.
   On a préféré le **clamp** : l'agent ne gaspille pas d'épisodes à explorer
   des actions interdites — la contrainte est respectée par construction.
4. **On pousse dans les « tuyaux »** (`_apply_state`, puis `_run_powerflow`) :
   batteries et solaire sont injectées comme `sgen`, la consommation des 32
   charges est réglée à `scaling = load_scale[t]`, et le
   **solveur** `pp.rundcpp` calcule où va chaque courant. C'est la « machine à
   conséquences » : on ne simule pas le réseau à la main, on laisse les
   lois de Kirchhoff parler. Les charges de chaque ligne en % de leur limite
   sont renvoyées → « loading_percent » (110 = le câble subit 10 % de trop).
5. **Le bilan** — la récompense « Physics-Informed » :
   ```
   Profit   = (puissance nette déchargée) × prix de l'heure
   Pénalité = 500 × (surcharge_totale + 2 × surcharge_totale²)  plafonnée à 5000 €
   Récompense R = Profit − Pénalité
   ```
   - la surcharge est mesurée par ligne au-delà de 100 % (somme, en per-unit) ;
   - le terme **quadratique** dissuade l'agent de « tout défoncer » plus
     fortement que le terme linéaire ne récompense l'argent ;
   - le **plafond** évite un chiffre invraisemblable (−10 000 € en un seul pas)
     qui déséquilibrerait l'apprentissage (le critic doit approximer ces valeurs) ;
   - si le solveur échoue à calculer quelque chose (jamais arrivé jusqu'ici),
     récompense −2000 € et on passe l'heure.

**L'observation — ce que l'agent voit (167 chiffres).** L'idée délicate :
SB3 consomme des vecteurs, mais l'agent a besoin de « voir un GRAPHE ».
Solution : on aplatit le graphe dans un ordre fixe, et le GNN le redécoupe :

```
[ 0    →   98 ]  les 33 nœuds × 3 features  : [SoC, solaire local, conso locale]
[ 99 →  162 ]  les 32 câbles × 2 features : [limite relative, charge actuelle]
[ 163       ]  le prix du marché (normalisé)
[ 164 → 166 ]  les 3 SoC (les mêmes que dans les nœuds, redonnés au plateau)
```

En détail :
- **SoC** : la 1re feature de nœud vaut 0 pour les 30 nœuds sans batterie
  (comme exigé par le plan) ;
- **solaire** : mesuré seulement sur les 3 sites hybrides ;
- **charge du câble** : `loading/100` coupé à 2.0 (le GNN n'a pas besoin de
  voir « 640 % », « vaut déjà 2× la limite » suffit) ;
- **le prix est normalisé** (z-score : (prix − moyenne)/écart-type de la
  période du fichier) — les réseaux de neurones apprennent bien mieux sur des
  nombres « autour de 0 » que sur des « 45,00 € ».

**Autres points qui comptent :**
- La télémétrie : le dictionnaire `info` renvoie tout ce qu'il faut pour
  l'analyse d'après-coup : `profit` (P&L brut), `thermal_penalty`,
  `max_loading_pct`, `n_lines_over_limit`, `p_dis_mw/p_ch_mw`, `soc_mwh`, etc.
  → `profit` et `reward` sont DEUX chiffres DISTINCTS (le second soustrait
  la pénalité) : utile, sinon on mélange vie financière et physique.
- `reset()` : une partie commence à une heure **aléatoire** de la période
  d'entraînement (`random_start=True`) — sinon l'agent apprendrait une
  séquence fixe. Pour l'évaluation : départ fixe, période complète.
- `terminated` reste False, `truncated=True` quand fenêtre achève (`t ≥ n_steps`) :
  convention Gymnasium — « fenêtre finie » pas « échec ».

**Où intervenir.** Pour ajouter une feature temporelle (ex. : les prix des
24 h suivantes, qui sont connus J-1 sur un vrai marché DA) il faut :
(a) augmenter `OBS_SIZE` via les constantes en tête de fichier, (b) alimenter
la nouvelle section dans `_current_obs` : le GNN suit automatiquement (il lit
ces constantes importées de `grid_env`).

### 3.5 `src/models/gnn_extractor.py` — BLOC C : le regard « graphe »

**Rôle.** Répondre à la question : comment un réseau de neurones « voit-il »
un réseau électrique ? Ce fichier est l'interface entre notre observation
(aplatie, 167 nombres) et ce que SB3 attend (un vecteur de taille libre).

**Pourquoi un réseau à GRAPHE ?** Un MLP classique verrait 167 chiffres
« ensemble » sans savoir que les positions 0–2 et 51–53 parlent de deux
extrémités du MÊME câble. Or la congestion dépend de cette relation locale :
câble chargé ← sa limite ← les injections de ses deux nœuds ← les injections
de leurs voisins... Le réseau à graphe (GNN) propage l'information de nœud en
nœud le long des câbles : la couche 1 donne à chaque nœud « une idée de ses
voisins directs », la couche 2 « des voisins de voisins ». Pour un réseau
radial de profondeur ~6, deux couches voient à mi-chemin ; c'est un
compromis volontaire (profondeur 2 = coût raisonnable).

**Le pipeline exact (comment un lot d'observations devient un vecteur de 32) :**

1. **Redécoupage** : le vecteur de 167 chiffres est redécoupé en :
   `x = (B, 33, 3)` les nœuds, `e = (B, 32, 2)` les câbles, `price`, `soc`.
2. **Trick du « batch disjoint »** : PyG traite plusieurs graphes simultanés
   en chiffrant leur numérotation (le graphe n° 1 a des nœuds 0–32, le n° 2 a
   les nœuds 33–65...). `_tiled_edge_index` crée cette numérotation disjointe —
   avec un cache pour ne pas la recalculer à chaque forward.
3. **Deux convolutions GAT** (`GATConv(..., edge_dim=2)`) :
   GAT = « Graph Attention » : au lieu que tous les voisins comptent autant,
   chaque câble apprend à **pondérer** ses voisins — et, comme `edge_dim=2`,
   la pondération lit les features du câble (sa limite, sa charge). En
   clair : *« le câble plein à 98 % qui me relie au voisin pèse dix fois plus
   que le câble fade à 20 % »*. C'est le mécanisme exact qui permet à
   l'agent d'« anticiper » le pic de charge — si le câble est déjà à 95 %
   l'heure d'avant, l'attention raconte au nœud qu'il faudra réduire.
   (Option GCNConv via `--conv-type gcn` : plus simple, pas d'attention, les
   features câbles sont alors collées au nœud aval — comparaison ablation.)
4. **Résumé global** : chaque nœud a maintenant un vecteur de 64. On n'a
   plus besoin du détail — on résume le réseau entier par deux images :
   la **moyenne** (état global du feeder) + le **max** (où est la pointe de
   congestion) → concat → Linear → 64.
5. **Fusion décisionnelle** : ce résumé spatial (64) est collé au **prix** (1)
   et aux **SoC** (3) — le « contexte marché » — puis Linear → 32.
   C'est LE vecteur que PPO consomme.

En aval (dans SB3, pas dans ce fichier) : le vecteur de 32 passe dans un MLP
64×64 qui donne **l'acteur** (les 3 actions, distribution Gaussienne) et le
**critic** (la valeur estimée V(s)) — les deux partagent l'extracteur GNN.

**Taille : ~19.8k paramètres.** Minuscule — parfait pour 1M steps sur CPU.

**Où intervenir.** `hidden_dim` (64), `n_heads` (2), `features_dim` (32)
sont des kwargs du constructeur, réglés dans `train.py`. Ajouter une 3e couche
de convolution ? Tout se joue entre `conv2` et `post_pool`.

### 3.6 `src/baselines/rule_based.py` — la référence « sans intelligence »

**Rôle.** Le plus simple de tous : 22 lignes. L'action suit l'horaire et
rien que l'horaire : charge max de 1h à 5h du matin, décharge de 18h à 21h,
zéro sinon. Il ne rentre jamais dans la salle des machines : il ne regarde ni
le réseau, ni sa charge, ni donc si le câble est saturé.

**Pourquoi.** C'est la contre-référence du projet : tout ce que fait l'agent
doit être justifié par rapport à un acheteur naïf. D'après les chiffres
déjà mesurés : +61 857 € de P&L brut sur 8 mois, mais 753 heures
de surcharge, avec des pics à 143.7 % — réseau défoncé. Si l'agent ne bat PAS
cette référence sur le couple (profit, sécurité), il ne sert à rien.

### 3.7 Les `scripts/` — les lignes de commande

Quatre wrappers minces qui appelleront les bibliothèques ci-dessus. On
préfère les script-thin (pas de logique) : la logique vit dans `src/`, les
scripts orchestrent, parsent les arguments, sauvegardent.

| Script | Args | Effet |
|---|---|---|
| `generate_data.py` | — | fabrique le CSV synthétique + résume ses stats |
| `run_baseline.py` | — | joue la baseline sur les 3 périodes → CSV horaires + compte rendu (P&L brut ≠ pénalités ≠ reward) |
| `train.py` | `--timesteps N` / `--fast` / `--conv-type gat-ou-gcn` | découpe train/val → 8 environnements simulés (DummyVecEnv) → `PPO('MlpPolicy')` branché à notre GNN → tous les 8192 steps, un détour déterministe sur la validation (garde le meilleur modèle) → `runs/` + `models/` |
| `eval.py` | `--model X.zip` / `--period val-ou-test` | rejoue l'agent + la baseline sur la période → les 3 graphiques livrables (P&L cumulé, histogramme safety, SoC vs soleil/prix) → résultats JSON |

Détail utile de `train.py` : `make_vec_env` fabrique un Monitor (journal des
épisodes) autour de chaque env — et c'est ce wrapper qui masque les attributs :
pour récupérer la topologie on passe par la couche `unwrapped`. `--fast` sert
à un test smoke (20k steps en ~4 min, 4 environnements) avant de lancer les
gros calculs.

**Pas encore écrits :** tests pytest ; ingestion données réelles ;
`VecNormalize` ; seed global ; ablation `conv_type='mlp'` (MLP pur sans
graphe — c'est elle qui pourra prouver que le GNN apporte quelque chose).

---

## 4. Le contrat de données — et le passage aux vraies sources

### 4.1 Le contrat actuel (tout le projet tient debout sur celui-là)

Un **unique** DataFrame, index horaire trié et continu, **3 colonnes** :

```python
index  : DatetimeIndex horaire, naïf (sans tz)
price      : €/MWh               -> z-score mémoire dans l'env
solar_pu   : 0..1 (facteur capacité) -> on donne tel quel au PV
load_scale : ~0.4..1.6 (forme)   -> multiplie les 32 charges IEEE33
```

Les quatre faits que TOUT loader futur (données réelles !) doit continuer à garantir :

- **H1 pas trous** : les heures du fichier se suivent ; l'env compte en
  positions (0, 1, 2...), le datetime ne garde qu'une étiquette.
- **H2 prix J-1 connu** : cohérent avec un vrai marché Day-Ahead (publié
  à 12h30 la veille). L'agent décide en sachant le prix du pas.
- **H3 per-unit** : solar_pu et load_scale n'ont pas d'unité ; les MW réels
  viennent du réseau (0.6 MW de PV et le « shape » des charges IEEE33).
- **H4 un profil pour 3 sites** (facteurs 1.00 / 0.95 / 1.05).

### 4.2 Le chemin de la vraie donnée brute (ton tableau)

| Source brute | Ce qu'on en tire | Difficultés réelles |
|---|---|---|
| **EPEX Spot** (enchère DA France, horaire) | `price` (€/MWh), la vraie donnée de marché | accès commercial (pas gratuit). Alternative : ENTSO-E Transparency Platform (**public**) renvoie le prix de couplage FR, identique à ce que publie l'enchère EPEX — mais il faut convertir UTC → heure locale et gérer les 2 journées de changement d'heure (une heure dupliquée, une heure absente) |
| **éCO2mix (RTE)** | `load_scale` = Consommation nationale / moyenne annuelle | CSV windows-1252, séparateur `;`, fichiers PAR SEMESTRE à concaténer, format de dates FR (« 01/02/2023 »), décimales à virgule, NaN lors des « missions de mesure », résolution 30 min → à moyenner à l'heure. NB : c'est une forme NATIONALE — ce n'est pas le profil du quartier |
| **Renewables.ninja** | `solar_pu` — **c'est exactement le facteur de capacité (0–1)**, mapping direct | token requis, quota gratuit limité, retour UTC à aligner. Un seul appel suffit si l'on garde le profil unifié (H4) |

**Plan d'implémentation** (prochaine étape) — nouveau module `src/data/ingest/` :

```text
src/data/ingest/
    eco2mix.py            fetch_load_scale(début, fin) -> colonne load_scale
    entsoe.py             fetch_prices(début, fin, token) -> colonne price
    renewables_ninja.py   fetch_solar(début, fin, lat/lon, token) -> colonne solar_pu
    build_dataset.py      fusionne + QA complet -> data/processed/real_2023.csv
```

Ce qu'il faut éventuellement modifier : **rien dans l'env / le GNN / SB3**.
Uniquement :
1. `load_data()` pointera vers le bon CSV (un champ `DataConfig.path_csv`
   à ajouter — aujourd'hui codé en dur `synthetic_2023.csv`) ;
2. une **QA d'ingestion** : comptage des NaN, continuité de l'index, bornes
   (prix < −100 €, CF > 1) — bloquant avec message clair, on veut savoir AVANT
   le training que les données mentent.

**Les trois écarts attendus au passage synthétique → réel** :
1. le vrai marché a un profil moins « propre » (couplage gaz/température/
   solaire) → la signature synthétique est pédagogique, pas réaliste ;
2. éCO2mix est une consommation nationale, pas le profil d'un feeder rural :
   proxy volontaire (toute la forme est suivie, PAS l'amplitude locale) ;
3. l'agent développé sur le synthétique peut, en apparence, apprendre
   une horloge (heure → action). Le réel le « guérit » naturellement grâce
   au bruit, aux spikes et au vrai couplage offre/demande.

---

## 5. Les librairies — qui fait quoi (et pourquoi on les a choisies)

| Librairie | Version | Rôle en une ligne | Où exactement |
|---|---|---|---|
| `pandapower` | 3.5.6 | simulateur électrique : fournit le réseau et calcule les flux | `case33bw`, `rundcpp`, `create_sgen` |
| `gymnasium` | 1.4.0 | l'interface standard « agent-environnement » | `gym.Env`, `spaces.Box` |
| `stable_baselines3` | 2.9.2 | l'implémentation PPO tractable et robuste | `PPO`, callbacks, VecEnv |
| `torch` | 2.15 CPU | tensors + autograd (les gradients du RL) | extracteur GNN, actor/critic |
| `torch_geometric` | 2.8.1 | les opérations sur graphes (convs, pooling) | `GATConv`, `GCNConv`, `global_mean_pool` |
| `pandas` / `numpy` | 2.3.3 / 2.4.6 | le tableau horaire + le calcul vectorisé | BLOC A, layout obs |
| `tensorboard` | (log `runs/`) | regarde les courbes d'apprentissage en direct | `tensorboard --logdir runs` |
| `matplotlib` (Agg) | 3.11.2 | les 3 graphiques livrables PNG | `eval.py` |
| `numba` (installé) | — | JIT : rend le solveur bien plus rapide (sinon warning de fallback lent) | Solveur pandapower |
| `tqdm` + `rich` | installés | la barre de progression pendant l'entraînement | `progress_bar=True` |
| `scikit-learn` | 1.9.1 | **encore inutilisé** (prévu pour scaler, splits, features) | — |

Le CPU suffit : chaque pas coûte ~10 ms de solveur pandapower — c'est le
goulot, pas les réseaux neuronaux (triviaux ici). GPU : pas une priorité.

---

## 6. Vocabulaire — les mots-clés en 1 phrase chacun

| Mot | Sens dans ce projet |
|---|---|
| **bus / nœud** | un point de connexion électrique (pas un équipement) ; 33 dans notre réseau |
| **ligne / arête** | un câble entre deux bus ; sa « capacité » est son courant admissible |
| **per-unit (pu)** | un nombre sans dimension divisé par une référence : `loading = 1.2` signifie « 120 % de la limite » |
| **Day-Ahead (DA)** | enchère de la veille : les prix de TOUS les 24 h de demain sont publiés à 12h30 la veille |
| **MDP** | modélisation agent-observation-action-récompense-état-suivant ; avec discount gamma |
| **PPO** | « Proximal Policy Optimization » : algo de RL robuste, apprend la politique par petits pas (clippés) — peu de configuration |
| **Actor / Critic** | l'acteur CHOISIT l'action, le critic ESTIME la valeur de la situation pour le guider |
| **GAE (λ)** | « Generalized Advantage Estimation » : mesure combien cette action était meilleure que la normale |
| **GATConv** | convolution à attention sur graphe : pondération apprise entre voisins, ici en lisant les features des câbles |
| **Pooling global** | résumer à 1 000 nœuds → 1 vecteur : moyenne (ambiance globale) + max (où est la pointe) |
| **clamp SoC** | borne mécanique d'énergie : la batterie ne peut dépasser son stock, donc on pige l'action à la puissance possible |
| **z-score** | normalisation (valeur − moyenne) / écart-type ; indispensable aux RL |
| **EvalCallback** | SB3 : toutes les N étapes, un tournoi déterministe sur la validation, garde le meilleur checkpoint |
| **random_start** | commencer l'épisode à une heure aléatoire : obligatoire pour que l'agent explore des phases contrastées |

---

## 7. Invariants — les choses à ne JAMAIS casser (futurs tests)

| Invariant | Vérification |
|---|---|
| L'obs fait 167 valeurs, ordre stable | assert dans le reset |
| Une batterie vide ne décharge pas (clamp) | p_dis == 0 si SoC == 0 |
| Une batterie pleine ne charge pas | p_ch == 0 si SoC == C |
| Les 5 lignes hors service de case33bw restent ignorées | 32 lignes actives seulement |
| Une pleine puissance peut saturer un câble (congestion existe) | max_loading > 100 % attendu |
| Batching GNN : obs identique → sorties identiques | forward avec batch B, lignes identiques |
| Reproductibilité | même seed → même courbe de récompense |

---
*Document issu d'une révision ligne-à-ligne des sources. Versions vérifiées
(torch 2.15 CPU / pandapower 3.5.6 / PyG 2.8.1 / SB3 2.9.2 / gymnasium 1.4.0).*
