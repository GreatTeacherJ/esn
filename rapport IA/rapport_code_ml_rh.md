# Rapport pédagogique — comparaison de modèles ML sur un dataset RH

## 1. Contexte du problème

Le jeu de données étudié décrit des salariés dans un contexte de **ressources humaines**. Il contient :

- **1 470 lignes**, chaque ligne correspondant à une observation/salarié ;
- **25 variables explicatives** (*features*) ;
- des variables de plusieurs types : **numériques**, **booléennes** et **catégorielles** ;
- une variable cible binaire `y`, indiquant si le salarié a démissionné ou non :
  - `0` : pas de démission ;
  - `1` : démission.

La classe positive représente environ **16,1 %** des observations. Le problème est donc **déséquilibré** : les démissions sont nettement moins nombreuses que les non-démissions.

Cette particularité est importante pour la modélisation. Une métrique comme l’accuracy pourrait donner une impression trompeuse de bonne performance : un modèle qui prédirait toujours « pas de démission » obtiendrait déjà environ 83,9 % d’accuracy, tout en ne détectant aucune démission. La suite du pipeline privilégie donc une métrique adaptée à la détection de la classe positive rare : l’**AUC-PR**, calculée ici par `average_precision`.

---

## 2. Imports

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import average_precision_score
import numpy as np
```

### Explication ligne par ligne

#### `from sklearn.ensemble import RandomForestClassifier`

Cette ligne importe `RandomForestClassifier` depuis le module `ensemble` de scikit-learn.

Une forêt aléatoire (*Random Forest*) est un ensemble de nombreux arbres de décision. Chaque arbre est entraîné sur une variation des données et des variables, puis les arbres sont combinés pour produire une prédiction plus robuste qu’un arbre isolé.

Le classifieur est adapté à une cible binaire comme la démission oui/non. Il peut également modéliser des relations non linéaires et des interactions entre variables sans imposer une forme linéaire aux effets des caractéristiques.

#### `from xgboost import XGBClassifier`

Cette instruction importe le classifieur `XGBClassifier` de la bibliothèque XGBoost.

XGBoost construit une séquence d’arbres de décision. Chaque nouvel arbre cherche à corriger les erreurs des arbres précédents. Cette stratégie de *gradient boosting* est souvent très performante sur des données tabulaires.

Le but du rapport est de comparer cette famille de modèles avec la forêt aléatoire, et non de supposer à l’avance qu’un modèle est meilleur que l’autre.

#### `from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split`

Cette ligne importe trois outils de séparation et d’évaluation :

- `train_test_split` réalise la séparation initiale entre les données utilisées pour développer les modèles et les données conservées pour l’évaluation finale ;
- `StratifiedKFold` définit une validation croisée qui conserve autant que possible la proportion des classes dans chaque pli ;
- `cross_val_score` entraîne et évalue automatiquement un modèle sur chacun des plis définis par le schéma de validation croisée.

#### `from sklearn.metrics import average_precision_score`

Cette ligne importe la fonction de calcul de l’average precision. Elle mesure la qualité du classement des observations positives en utilisant les valeurs de précision et de rappel lorsque le seuil de décision varie.

Dans le bloc de validation croisée ci-dessous, le nom de scoring utilisé par scikit-learn est directement `"average_precision"`. L’import de `average_precision_score` est utile lorsque l’on veut calculer explicitement cette métrique, notamment lors d’une évaluation finale.

#### `import numpy as np`

Cette instruction importe NumPy sous l’alias court `np`.

NumPy fournit les structures et opérations numériques courantes en Python scientifique. Il peut notamment servir à manipuler des tableaux, calculer des moyennes ou résumer les scores obtenus par les différents plis. L’alias `np` est une convention largement utilisée.

---

## 3. Split initial train/test

```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### Explication ligne par ligne et paramètre par paramètre

#### `X_train_val, X_test, y_train_val, y_test =`

La fonction retourne quatre objets :

- `X_train_val` contient les variables explicatives destinées au développement, à la comparaison et à la sélection des modèles ;
- `X_test` contient les variables explicatives mises de côté pour l’évaluation finale ;
- `y_train_val` contient les cibles correspondant à `X_train_val` ;
- `y_test` contient les cibles correspondant à `X_test`.

Le nom `train_val` rappelle que cette partie pourra être utilisée à la fois pour l’entraînement et pour la validation croisée. Elle ne doit toutefois pas être confondue avec le jeu de test final.

#### `train_test_split(`

`train_test_split` réalise une séparation aléatoire des observations. Il veille à ce que les lignes de `X` restent correctement associées aux valeurs correspondantes de `y`.

Cette séparation initiale crée une frontière méthodologique importante : les décisions prises pendant le développement — comparaison des modèles et choix du meilleur candidat — se font à partir de `X_train_val` et `y_train_val`, tandis que `X_test` et `y_test` restent à l’écart.

#### `X,`

`X` représente la matrice des 25 variables explicatives. Chaque ligne correspond à un salarié et chaque colonne à une feature numérique, booléenne ou catégorielle déjà préparée selon le prétraitement utilisé dans le code.

#### `y,`

`y` représente la cible binaire : la présence ou l’absence d’une démission.

#### `test_size=0.2,`

`test_size=0.2` réserve **20 %** des observations au jeu de test et laisse **80 %** pour le développement (`X_train_val`, `y_train_val`). Avec 1 470 lignes, cela correspond approximativement à 294 observations de test et 1 176 observations de développement, selon le mode exact d’arrondi de la fonction.

Un test de 20 % est un compromis courant :

- il fournit un nombre suffisamment important d’observations pour évaluer la performance finale ;
- il conserve la majorité des données pour entraîner et comparer les modèles ;
- il évite de consacrer une portion excessive des données à une évaluation qui ne sera effectuée qu’à la fin.

Dans un problème déséquilibré, il est également essentiel que le jeu de test contienne suffisamment de positifs pour que l’évaluation soit informative.

#### `stratify=y,`

`stratify=y` demande à `train_test_split` de réaliser une séparation **stratifiée** par rapport aux classes de `y`.

Sans stratification, un tirage aléatoire pourrait produire des proportions de démissions différentes entre l’entraînement et le test. Le risque est particulièrement important lorsque la classe positive est minoritaire. Ici, la proportion globale de positifs est d’environ **16,1 %** : la stratification cherche à conserver cette proportion, à l’arrondi près, dans les deux sous-ensembles.

Ainsi :

- `y_train_val` reste représentatif de la distribution de la cible ;
- `y_test` contient lui aussi une proportion de démissions proche de celle du dataset initial ;
- la comparaison et l’évaluation sont moins dépendantes d’un tirage exceptionnellement favorable ou défavorable.

#### `random_state=42`

`random_state=42` fixe la graine du générateur pseudo-aléatoire utilisé pour le split.

Le nombre `42` n’a pas de signification statistique particulière. L’important est de fixer une valeur afin que le découpage soit **reproductible** : en réexécutant le code dans les mêmes conditions, on obtient les mêmes observations dans les ensembles d’entraînement et de test. Cela facilite le débogage, la comparaison des expériences et la traçabilité du travail.

### Pourquoi effectuer ce split initial ?

Le jeu de test doit rester une estimation aussi honnête que possible de la performance sur des données non vues. Si les mêmes observations servaient à la fois à comparer les modèles et à annoncer la performance finale, cette performance pourrait être trop optimiste : le processus de sélection aurait indirectement été adapté au test.

Le split initial sert donc à :

1. **développer et comparer** les modèles sur `X_train_val` et `y_train_val` ;
2. préserver `X_test` et `y_test` pour une **évaluation finale honnête** ;
3. éviter que le choix du modèle bénéficie d’informations provenant du jeu de test.

---

## 4. Définition du schéma de validation croisée

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Explication ligne par ligne et paramètre par paramètre

#### `cv =`

La variable `cv` stocke le schéma de validation croisée qui sera transmis à `cross_val_score`.

Il ne s’agit pas encore d’un modèle ni de scores calculés : c’est la règle qui décrit comment `X_train_val` et `y_train_val` seront découpés à répétition.

#### `StratifiedKFold(`

`StratifiedKFold` divise les données en plusieurs plis (*folds*) tout en essayant de préserver la proportion des classes dans chaque pli.

Il est préférable à un `KFold` classique ici parce que la classe positive est minoritaire. Avec un `KFold` non stratifié, certains plis pourraient contenir trop peu de démissions — voire une proportion très différente de la proportion globale — ce qui rendrait les scores instables ou peu représentatifs.

La stratification est donc particulièrement importante pour une métrique orientée vers la classe positive comme l’average precision.

#### `n_splits=5,`

`n_splits=5` demande de créer **5 plis**.

À chaque itération, quatre plis servent à entraîner le modèle et le pli restant sert à l’évaluer. Le processus est répété de sorte que chacun des cinq plis joue une fois le rôle de validation.

Cinq plis constituent un compromis entre plusieurs objectifs :

- utiliser une grande partie des données pour l’entraînement à chaque itération ;
- obtenir plusieurs estimations du score afin de mesurer sa variabilité ;
- limiter le temps de calcul, puisqu’un modèle est réentraîné pour chaque pli ;
- éviter le coût supplémentaire d’une validation à 10 plis, tout en étant plus robuste qu’un seul split de validation.

Le choix de 5 n’est pas une constante universelle : il s’agit d’un compromis raisonnable pour ce dataset de taille modérée.

#### `shuffle=True,`

`shuffle=True` mélange les observations avant de construire les plis.

Ce mélange évite que l’ordre initial du dataset influence les folds. Par exemple, si les données étaient regroupées par service, ancienneté ou ordre d’importation, un découpage sans mélange pourrait créer des plis non représentatifs.

Le mélange est réalisé avant la stratification, de façon à obtenir des plis à la fois mélangés et équilibrés en proportion de classes.

#### `random_state=42`

La graine fixe le mélange utilisé pour construire les folds. Comme pour le split initial, elle rend la validation croisée reproductible. Les mêmes données seront affectées aux mêmes plis lors d’une nouvelle exécution dans les mêmes conditions.

---

## 5. Comparaison de Random Forest et XGBoost avec `cross_val_score`

Le code de comparaison prend la forme suivante :

```python
models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring="average_precision"
    )
    print(f"{name}: AUC-PR moyenne = {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Explication du dictionnaire des modèles

#### `models = {`

Le dictionnaire regroupe les modèles à comparer sous des noms lisibles. Cette organisation permet d’utiliser exactement la même procédure d’évaluation pour chaque algorithme, ce qui rend la comparaison plus cohérente.

#### `"Random Forest": RandomForestClassifier(),`

La clé `"Random Forest"` est le nom affiché dans les résultats. La valeur `RandomForestClassifier()` est l’instance du modèle de forêt aléatoire.

Les parenthèses sans argument signifient que l’exemple utilise les paramètres par défaut de scikit-learn, sauf si d’autres paramètres ont été définis précédemment dans le code fourni. Le modèle est ensuite cloné et réentraîné indépendamment dans chaque fold par `cross_val_score`.

#### `"XGBoost": XGBClassifier()`

La clé `"XGBoost"` sert à identifier le modèle dans l’affichage. `XGBClassifier()` crée le classifieur XGBoost.

Comme pour la forêt aléatoire, les parenthèses indiquent ici que les réglages éventuels ne sont pas détaillés dans ce bloc. L’important pour la comparaison est que les deux modèles soient évalués sur les mêmes observations, avec le même schéma de folds et la même métrique.

### Explication de la boucle

#### `for name, model in models.items():`

`models.items()` parcourt les paires nom-modèle du dictionnaire.

À chaque passage :

- `name` reçoit le nom lisible, par exemple `"Random Forest"` ;
- `model` reçoit l’instance correspondante ;
- le même appel de validation croisée est exécuté pour ce modèle.

Cette boucle évite de dupliquer le code d’évaluation et réduit le risque d’appliquer des règles différentes aux deux algorithmes.

### Explication de l’appel à `cross_val_score`

```python
scores = cross_val_score(
    model,
    X_train_val,
    y_train_val,
    cv=cv,
    scoring="average_precision"
)
```

#### `model,`

Il s’agit du classifieur actuellement traité par la boucle : la forêt aléatoire ou XGBoost.

`cross_val_score` travaille avec une copie indépendante du modèle pour chaque fold. Le modèle n’est donc pas simplement réutilisé avec son état appris lors du fold précédent.

#### `X_train_val,`

Les features utilisées pour la comparaison proviennent exclusivement de la partie de développement créée par le split initial.

#### `y_train_val,`

Les labels correspondants sont fournis à la fonction afin qu’elle puisse entraîner le modèle et calculer la métrique sur les prédictions du fold de validation.

#### `cv=cv,`

Ce paramètre indique à `cross_val_score` d’utiliser le schéma `StratifiedKFold` défini précédemment : 5 folds, mélange activé et graine fixée.

#### `scoring="average_precision"`

Ce paramètre demande à scikit-learn de calculer l’**average precision**, utilisée ici comme mesure de l’**AUC-PR**.

L’AUC-PR est plus adaptée que l’accuracy lorsque la classe positive est rare. Elle se concentre sur le compromis entre :

- la **précision** : parmi les salariés prédits comme démissionnaires, quelle proportion démissionne réellement ?
- le **rappel** : parmi les salariés qui démissionnent réellement, quelle proportion le modèle parvient-il à identifier ?

L’accuracy compte correctement les deux classes, mais elle peut être dominée par la classe majoritaire. Dans ce dataset, un modèle qui prédit presque toujours l’absence de démission pourrait avoir une accuracy élevée sans être utile pour repérer les départs. L’average precision pénalise davantage ce type de comportement et évalue la qualité du classement des positifs potentiels.

### Mécanisme interne de `cross_val_score`

Avec 5 folds, `cross_val_score` automatise le processus suivant :

1. il utilise quatre folds de `X_train_val` et `y_train_val` pour entraîner une nouvelle instance du modèle ;
2. il prédit les observations du cinquième fold, qui n’ont pas servi à cet entraînement ;
3. il calcule l’average precision sur ce fold de validation ;
4. il recommence en changeant le fold de validation ;
5. il retourne finalement un tableau de cinq scores, un par fold.

Ainsi, chaque ligne de `X_train_val` sert **une fois de test de validation** et **quatre fois de données d’entraînement**. Il ne s’agit pas du jeu de test final : le mot « test » dans la description d’un fold désigne ici le sous-ensemble temporairement utilisé pour valider l’itération.

Le réentraînement à zéro à chaque fold est essentiel. Il empêche le modèle de conserver un apprentissage provenant d’un fold précédent et permet d’obtenir une estimation plus honnête de sa capacité à généraliser. La séparation entre le sous-ensemble d’entraînement du fold et son sous-ensemble de validation évite également d’évaluer un modèle sur les mêmes lignes que celles utilisées pour son apprentissage.

`cross_val_score` renvoie donc cinq valeurs dans `scores`. Ces valeurs peuvent ensuite être résumées par leur moyenne et leur écart-type :

- `scores.mean()` estime la performance moyenne du modèle sur les folds ;
- `scores.std()` indique la variabilité de cette performance selon les sous-échantillons de validation.

### Explication de l’affichage

```python
print(f"{name}: AUC-PR moyenne = {scores.mean():.4f} (+/- {scores.std():.4f})")
```

- `name` affiche le nom du modèle évalué ;
- `scores.mean()` affiche la moyenne des cinq scores ;
- `:.4f` formate chaque valeur avec quatre chiffres après la virgule ;
- `scores.std()` affiche l’écart-type des scores ;
- `(+/- ...)` donne une indication synthétique de la dispersion entre les folds.

La comparaison doit principalement s’appuyer sur l’AUC-PR moyenne, tout en observant la dispersion : un modèle avec une moyenne légèrement supérieure mais une variabilité beaucoup plus grande peut être moins stable qu’un modèle dont les performances sont plus régulières.

### Pourquoi ne pas utiliser `X_test` pendant cette comparaison ?

La validation croisée est effectuée sur `X_train_val` et `y_train_val`, jamais sur `X_test` et `y_test` :

```python
scores = cross_val_score(
    model,
    X_train_val,
    y_train_val,
    cv=cv,
    scoring="average_precision"
)
```

Le jeu `X_test`/`y_test` a été isolé dès le départ pour représenter des données réellement nouvelles. Si ses résultats étaient consultés pendant la comparaison, puis utilisés pour décider quel modèle retenir ou modifier, le test ne serait plus indépendant. La performance finale risquerait alors d’être optimiste.

La logique correcte est donc :

- utiliser `X_train_val` et `y_train_val` pour la validation croisée et la comparaison entre Random Forest et XGBoost ;
- conserver `X_test` et `y_test` hors de cette étape ;
- réserver le jeu de test à l’évaluation finale du modèle retenu.

La procédure décrite dans ce rapport s’arrête ici, après la comparaison des deux modèles par validation croisée stratifiée avec l’average precision comme métrique AUC-PR.
