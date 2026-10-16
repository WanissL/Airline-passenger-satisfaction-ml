# Prédiction de la Satisfaction des Passagers Aériens (Classification Supervisée & Interprétabilité)

Ce projet développe une chaîne complète d'apprentissage automatique supervisé pour prédire le niveau de satisfaction des passagers d'une compagnie aérienne (*Satisfied* vs *Neutral or Dissatisfied*) à partir de 22 attributs d'expérience de vol, de service en cabine et de données opérationnelles.

L'enjeu managérial réside dans l'identification des leviers de service à plus fort impact sur la fidélisation client et la réputation de marque.

---

## 1. Périmètre des Données

* **Volume global** : 129 880 observations passagers.
* **Variable cible (`satisfaction`)** : Binaire (*neutral or dissatisfied* : 56,5 %, *satisfied* : 43,5 %).
* **Variables d'entrée** :
  * Profil passager : Genre, type de fidélité (*Loyal / Disloyal*), âge, motif de voyage (*Personal / Business Travel*), classe de cabine (*Eco, Eco Plus, Business*).
  * Évaluations de service (notes de 1 à 5) : Confort du siège, divertissement à bord, service Wi-Fi, facilité de réservation, embarquement en ligne, propreté, enregistrement, ponctualité.
  * Exploitation : Distance du vol, retard au départ en minutes.

---

## 2. Préparation & Pipeline de Prétraitement

1. **Nettoyage des données** :
   * Suppression des identifiants techniques sans valeur informative (`Unnamed: 0`, `id`).
   * Diagnostic de multicolinéarité : Retrait de `Arrival Delay in Minutes` en raison de sa redondance mathématique directe avec `Departure Delay in Minutes` ($r > 0{,}96$).
2. **Découpage stratifié** :
   * Partition 80 % train / 20 % test avec stratification sur la variable cible (`stratify=Y`), garantissant le maintien exact des proportions de satisfaction entre les ensembles.
3. **Encapsulation Scikit-Learn (`ColumnTransformer`)** :
   * Variables numériques (17 colonnes) : Imputation de la moyenne + normalisation unitaire (`StandardScaler`).
   * Variable ordinale (`Class`) : `OrdinalEncoder` respectant la hiérarchie naturelle `['Eco', 'Eco Plus', 'Business']`.
   * Variables catégorielles nominales (`Gender`, `Customer Type`, `Type of Travel`) : `OneHotEncoder(drop='first')`.

---

## 3. Évaluation Comparative des Modèles (Test Set)

Six approches d'apprentissage ont été entraînées et comparées sur les mêmes données de test :

| Modèle | Accuracy | Précision | Rappel | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVC** | 87,38 % | 87,10 % | 83,29 % | 0,8515 | — |
| **Régression Logistique** | 87,40 % | 86,99 % | 83,49 % | 0,8521 | 0,9269 |
| **Arbre de Décision (Decision Tree)** | 94,55 % | 93,53 % | 93,96 % | 0,9375 | 0,9448 |
| **SVC (Noyau RBF)** | 95,07 % | 95,40 % | 93,15 % | 0,9426 | — |
| **Voting Classifier (Ensemble)** | 95,41 % | — | — | 0,9541 | — |
| **Forêt Aléatoire (Random Forest)** | **96,20 %** | **97,23 %** | **93,92 %** | **0,9555** | **0,9938** |

Le modèle **Random Forest** surpasse l'ensemble des compétiteurs, y compris le modèle ensembliste de vote pondéré, en offrant une séparation quasi-parfaite des classes avec une aire sous la courbe ROC de **0,9938**.

---

## 4. Validation Croisée & Optimisation

* **Validation croisée (Stratified 5-Folds)** :
  * Accuracy moyenne : **$96{,}36\,\%$**
  * F1-score moyen : **$0{,}9628$**
  * ROC-AUC moyen : **$0{,}9941$**
* **Réglage fin (`RandomizedSearchCV`)** :
  * Meilleurs hyperparamètres : `n_estimators=400`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=2`, `criterion='entropy'`.
* **Évaluation train vs test finale** :
  * F1-score : Train $0{,}9901$ vs Test $0{,}9558$.
  * ROC-AUC : Train $0{,}9998$ vs Test $0{,}9941$.

---

## 5. Analyse d'Importance des Variables (Explicabilité)

Deux méthodes indépendantes d'importance ont été appliquées pour interpréter les décisions du modèle :
1. **MDI (*Mean Decrease Impurity*)** : Mesure basée sur la réduction d'entropie au sein des arbres.
2. **Permutation Importance** : Mesure non biaisée sur le jeu de test évaluant la baisse réelle du macro-F1 lors du mélange aléatoire d'une feature.

### Facteurs déterminants de la satisfaction
1. **Type of Travel** (Motif personnel vs professionnel) : Facteur premier discriminant sur les attentes en cabine.
2. **Inflight Wifi Service & Online Boarding** : La qualité des services digitaux (connexion en vol et enregistrement fluide) conditionne fortement l'expérience globale.
3. **Class & Customer Type** : La classe voyageur et le statut de fidélité modulent le seuil d'exigence.
4. **Baggage Handling & Inflight Entertainment** : Services opérationnels clés dans la perception de valeur ajoutée.

---

## 6. Structure du Dépôt

```text
airline-passenger-satisfaction-ml/
├── README.md                     # Documentation complète et conclusions métier
├── train.csv                     # Données d'entraînement (103k lignes)
├── test.csv                      # Données de test (26k lignes)
└── Classification_case.ipynb     # Notebook avec pré-traitement, modèles et interprétabilité
```

---

## 7. Exécution Locale

Pour reproduire l'analyse et exécuter le modèle sur votre machine :

```bash
# 1. Cloner le projet
git clone [https://github.com/WanissL/airline-passenger-satisfaction-ml.git](https://github.com/WanissL/airline-passenger-satisfaction-ml.git)

# 2. Accéder au répertoire
cd airline-passenger-satisfaction-ml

# 3. Installer les dépendances requises
pip install numpy pandas scikit-learn matplotlib seaborn

# 4. Lancer Jupyter Notebook
jupyter notebook Classification_case.ipynb
