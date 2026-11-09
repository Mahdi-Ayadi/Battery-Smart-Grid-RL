# PROJET : Arbitrage de marché sous contraintes physiques via Graph-RL (GNN + PPO)

## 1. Origine et Objectifs du Projet

* **Le constat (Point de départ) :** Aujourd'hui, les algorithmes de trading des batteries (Virtual Power Plants) cherchent uniquement à maximiser le profit. Ils ignorent la physique. Résultat : s'ils déchargent toutes les batteries en même temps lors d'un pic de prix, ils créent des congestions locales et font sauter le réseau de distribution.

* **L'objectif principal :** Développer un agent autonome capable de maximiser la valorisation financière (arbitrage sur le marché EPEX) d'une flotte de 3 batteries, tout en respectant strictement les limites physiques (thermiques) du réseau électrique local.

* **La méthode :** Utiliser un réseau de neurones sur graphe (GNN) pour donner à l'agent RL une "conscience spatiale" du réseau, lui permettant d'anticiper les congestions.

## 2. Formulation Mathématique (Processus de Décision Markovien - MDP)

L'interaction entre l'Agent et l'Environnement est définie par ce MDP :

* **L'État ($S_t$) - Format Graphe :**

  * **Nœuds (33) :** Niveau de batterie (SoC), Météo locale (Solaire), Consommation locale. *(Note : Les 30 nœuds sans batterie auront un SoC = 0)*.

  * **Arêtes (32) :** Capacité maximale de la ligne (MW), Flux actuel circulant dans la ligne (calculé par le solveur).

  * **Contexte Global :** Le prix du marché Day-Ahead à l'instant $t$.

* **L'Action ($A_t$) :**

  * Un vecteur de dimension 3 : $[a_1, a_2, a_3] \in [-1, 1]^3$.

  * $-1$ = Charge maximale, $0$ = Attente, $1$ = Décharge maximale (Vente).

* **La Récompense ($R_t$) - L'approche "Physics-Informed" :**

  * $R_t = \text{Profit Financier} - \text{Pénalités Physiques}$

  * *Profit :* $\sum (Action \times Prix\_Marché)$

  * *Pénalité Thermique :* Très forte pénalité négative si le solveur indique qu'une ligne dépasse 100% de sa capacité thermique.

  * *Pénalité Batterie :* Pénalité si l'agent tente de décharger une batterie à 0% ou charger une batterie à 100%.

## 3. Architecture Technique et Stack Logicielle

Le projet est divisé en 3 grands blocs :

### BLOC A : Le Pipeline de Données (Data Engineering)

* **Outils :** `pandas`, `scikit-learn`

* **Sources :**

  * Prix horaires 2023 (EPEX Spot / ENTSO-E).

  * Production solaire horaire 2023 (Renewables.ninja).

  * Profil de consommation standard.

* **Traitement :** Fusion des tables sur le timestamp (DateTime Index). Normalisation de toutes les features avec `StandardScaler` pour stabiliser les gradients de l'agent RL.

### BLOC B : Le Simulateur Physique (L'Environnement Gym)

* **Outils :** `gymnasium`, `pandapower`

* **Topologie :** Réseau standard de recherche **IEEE 33-bus**.

* **Éléments ajoutés :** 3 systèmes hybrides (Générateur Solaire + Stockage Batterie) fixés manuellement aux nœuds 18, 25 et 33 (les extrémités du réseau, là où les congestions sont les plus critiques).

* **Moteur physique :** Le solveur DC-OPF (`pp.rundcpp()`) de pandapower. Il tourne à chaque `step()` pour valider si les actions de l'agent respectent les lois de Kirchhoff et calculer la charge des câbles.

### BLOC C : Le Cerveau IA (GNN + Agent RL)

* **Outils :** `PyTorch`, `PyTorch Geometric (PyG)`, `Stable-Baselines3 (SB3)`

* **Le Feature Extractor (GNN) :**

  1. Reçoit le graphe (nœuds, arêtes) et passe le tout dans 2 couches de Convolution de Graphe (`GCNConv` ou `GATConv`).

  2. Compresse le graphe en un vecteur (Global Mean Pooling).

  3. Concatène ce résumé spatial avec le prix du marché.

* **L'Acteur-Critique (RL) :**

  * L'algorithme **PPO (Proximal Policy Optimization)** prend ce vecteur final et décide des actions (Charge/Décharge) pour les 3 batteries.

## 4. Méthodologie et Étapes de Développement

1. **Baseline (La référence experte) :** Coder une stratégie "bête" sans IA. La batterie se charge la nuit (quand c'est pas cher) et se décharge à 19h (pic de prix), sans regarder le réseau. Cela te donnera le P&L financier de base, mais fera sûrement exploser les lignes du réseau.

2. **Création de l'Environnement (`env.py`) :** Wrapper le réseau pandapower dans la classe `gym.Env`. S'assurer que le solveur tourne bien à chaque pas de temps et renvoie le bon dictionnaire d'observations.

3. **Création du Cerveau (`gnn.py`) :** Coder l'architecture réseau avec PyTorch Geometric et l'interfacer comme `BaseFeaturesExtractor` pour Stable-Baselines3.

4. **Entraînement (`train.py`) :** Lancer l'apprentissage de PPO sur plusieurs millions de timesteps (rapide avec le solveur DC). Observer les courbes tensorboard (le reward doit monter, les violations de réseau doivent descendre à zéro).

5. **Évaluation (`eval.py`) :** Faire jouer l'agent sur des données jamais vues (ex: l'année 2024).

## 5. Livrables

Le repository final devra contenir ces 3 graphiques (les vraies métriques de R&D) :

1. **Le Graphique Financier (Valorisation) :** Courbe de ton P&L cumulé sur un mois, comparé à la stratégie "Baseline". Prouve que ton IA sait trader.

2. **Le Graphique Physique (Safety) :** Un histogramme de la charge (en %) de la ligne la plus congestionnée du réseau.

   * *Baseline :* Fait des pics à 120% (Réseau détruit).

   * *Ton Agent GNN :* Reste sagement bloqué à 99% (L'IA a appris les lois de Kirchhoff et écrête ses ventes pour sauver le réseau).

3. **L'Action vs Météo :** Un zoom sur 3 jours montrant le SoC de la batterie face au soleil et aux prix, pour voir visuellement l'IA prendre des décisions complexes ("je stocke le soleil parce que le câble est plein, et je revends ce soir").
