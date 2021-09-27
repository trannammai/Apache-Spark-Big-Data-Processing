import numpy as np
import pandas as pd

import sys
import dask.dataframe

from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from dask_ml.cluster import KMeans
from dask_ml.decomposition import PCA
import dask.array as da
from dask_glm.estimators import LinearRegression, LogisticRegression

#
# IMPORT DU JEU DE DONNEES
# 

### Indiquer le dossier et le fichier cible
fichier_echantillon = 'train_echantillon.csv'
fichier_complete = 'train.csv'
dossier = 'C:/Tran_Nam_Mai/Github Profile/Apache-Spark-Big-Data-Processing/Data/'

### Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)
train_echantillon = pd.read_csv(dossier + fichier_echantillon)
train_echantillon.head()

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)
train_complete = dask.dataframe.read_csv(dossier + fichier_complete)
train_complete.head()

#
# PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 
### Nettoyer et préparer les données
# Enlever les valeurs incorrectes ou manquantes (si pertinent)
# ---------- Utiliser une librairie usuelle

for x in ['pickup_longitude']:
    q75,q25 = np.percentile(train_echantillon.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    train_echantillon.loc[train_echantillon[x] < min,x] = np.nan
    train_echantillon.loc[train_echantillon[x] > max,x] = np.nan

for x in ['pickup_latitude']:
    q75,q25 = np.percentile(train_echantillon.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    train_echantillon.loc[train_echantillon[x] < min,x] = np.nan
    train_echantillon.loc[train_echantillon[x] > max,x] = np.nan

for x in ['dropoff_longitude']:
    q75,q25 = np.percentile(train_echantillon.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    train_echantillon.loc[train_echantillon[x] < min,x] = np.nan
    train_echantillon.loc[train_echantillon[x] > max,x] = np.nan

for x in ['dropoff_latitude']:
    q75,q25 = np.percentile(train_echantillon.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    train_echantillon.loc[train_echantillon[x] < min,x] = np.nan
    train_echantillon.loc[train_echantillon[x] > max,x] = np.nan

train_echantillon = train_echantillon.dropna()

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

train_complete = train_complete.dropna()

# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie
# ---------- Utiliser une librairie usuelle
train_echantillon = train_echantillon[['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete = train_complete[['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# et filter les valeurs aberrantes
# ---------- Utiliser une librairie usuelle
train_echantillon.describe()

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete.describe().compute()

# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)
# ---------- Utiliser une librairie usuelle
fig, ax = plt.subplots(3,1)
ax[0].hist(train_echantillon['fare_amount'])
ax[1].scatter(train_echantillon['pickup_longitude'], train_echantillon['pickup_latitude'], c = "g", alpha = 0.005, marker = '.')
ax[2].scatter(train_echantillon['dropoff_longitude'], train_echantillon['dropoff_latitude'], c = "g", alpha = 0.005, marker = '.')
ax[0].set_title('fare_amount')
ax[1].set_title('pickup_location')
ax[2].set_title('dropoff_location')
plt.show()

# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")
# ---------- Utiliser une librairie usuelle
train_echantillon_sortie = train_echantillon[['fare_amount']]
train_echantillon_entree = train_echantillon.drop(['fare_amount'], axis = 1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_sortie = train_complete[['fare_amount']]
train_complete_entree = train_complete.drop(['fare_amount'], axis = 1)

# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)
# ---------- Utiliser une librairie usuelle
train_echantillon_sortie_scale = sklearn.preprocessing.scale(train_echantillon_sortie)
train_echantillon_entree_scale = sklearn.preprocessing.scale(train_echantillon_entree)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_sortie_scale = sklearn.preprocessing.scale(train_complete_sortie)
train_complete_entree_scale = sklearn.preprocessing.scale(train_complete_entree)

#
# CLUSTERING DU JEU DE DONNEES
# 
### Réaliser un clustering k-means sur les données d'entrée standardisées
# ---------- Utiliser une librairie usuelle
kmeans_model = KMeans(n_clusters = 4, random_state = 1).fit(train_echantillon_entree_scale) 
labels = kmeans_model.labels_

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
kmeans_dask = KMeans(n_clusters = 4)
kmeans_dask.fit_transform(train_complete_entree_scale)
labels_dask = kmeans_dask.labels
kmeans_dask.fit()

### Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters
# ---------- Utiliser une librairie usuelle
inertia = kmeans_model.inertia_


sse = []
for k in range(1, 10):
    kmeans_model = KMeans(n_clusters = k, random_state = 1).fit(train_echantillon_entree_scale) 
    sse.append(kmeans_model.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 10), sse)
plt.xticks(range(1, 10))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

### A partir de combien de clusters on peut dire que partitionner n'apporte plus
###        grand chose? Pourquoi?

# À partir des deux graphiques, nous pouvons voir que le nombre de cluster optimal est égal à deux, 
# là où la courbe commence à avoir un rendement décroissant. C'est aussi où on peut dire que partionner
# n'apporte plus grand chose.

### Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 

# Il y a 3 clusters principaux selon la localisation 

### Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables
# ---------- Utiliser une librairie usuelle
data_cluster = pd.DataFrame(train_echantillon_entree_scale)
data_cluster['cluster'] = labels
sample_index = np.random.randint(0, len(pd.DataFrame(train_echantillon_entree_scale)), 1000)
sns.pairplot(data_cluster.loc[sample_index, :], hue = 'cluster')
plt.show()

#
# ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 
### Faire une ACP sur le jeu de données standardisé
# ---------- Utiliser une librairie usuelle
pca = PCA(n_components = 4)
pca_result = pca.fit_transform(train_echantillon_entree_scale)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
dX = da.from_array(train_complete_entree_scale, chunks = train_complete_entree_scale.shape)
big_pca = PCA(n_components = 4)
big_pca.fit(dX)

### Réaliser le diagnostic de variance avec un graphique à barre (barchart)
# ---------- Utiliser une librairie usuelle
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# ---------- Utiliser une librairie usuelle
print(big_pca.explained_variance_ratio_)
print(big_pca.singular_values_)

# On doit garder deux composantes principales parce qu'il représente déjà 82% des variances des composantes
# A partir du trosième composante, on ne peux apporter qu'un peu de variances.

### Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation
# ---------- Utiliser une librairie usuelle
xvector = pca.components_[0]
yvector = pca.components_[1]

xs = pca.transform(train_echantillon_entree_scale)[:,0]
ys = pca.transform(train_echantillon_entree_scale)[:,1]

points_plot_index = np.random.randint(0, len(xs), 1000)
for i in points_plot_index:
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(train_echantillon_entree.index)[i], color = 'b')
    
for i in range (len(xvector)):
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color = 'r', width = 0.0005, head_width = 0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2, list(train_echantillon_entree.columns.values)[i], color = 'r')
plt.show()

#
# REGRESSION LINEAIRE
# 
### Mener une régression linéaire de la sortie "fare_amount"
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données
# ---------- Utiliser une librairie usuelle
regr = linear_model.LinearRegression().fit(train_echantillon_entree_scale, train_echantillon_sortie_scale)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_regr = dask_glm.estimators.LinearRegression().fit(train_complete_entree_scale, train_complete_sortie_scale)

# Tout les variables sont significatives au seuil 1%. Cependant, seulement 2,1% de la variance trouvée dans la variable de réponse (fare_amount)
# peut être expliquée par les variables prédictives.


### Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire
# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)
# ---------- Utiliser une librairie usuelle
train, validate, test = np.split(pd.DataFrame(train_echantillon).sample(frac=1), [int(.6*len(pd.DataFrame(train_echantillon))), int(.8*len(pd.DataFrame(train_echantillon)))])

train_sortie = train[['fare_amount']]
train_entree = train.drop(['fare_amount'], axis = 1)

validate_sortie = validate[['fare_amount']]
validate_entree = validate.drop(['fare_amount'], axis = 1)

test_sortie = test[['fare_amount']]
test_entree = test.drop(['fare_amount'], axis = 1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_train, big_validation, big_test = train_complete.random_split([0.6, 0.2, 0.2], random_state = 100)

big_train_sortie = big_train[['fare_amount']]
big_train_entree = big_train.drop(['fare_amount'], axis = 1)

big_validate_sortie = big_validate[['fare_amount']]
big_validate_entree = big_validate.drop(['fare_amount'], axis = 1)

big_test_sortie = big_test[['fare_amount']]
big_test_entree = big_test.drop(['fare_amount'], axis = 1)

# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 
# ---------- Utiliser une librairie usuelle
regr = linear_model.LinearRegression()
regr_train = regr.fit(train_entree, train_sortie)
pred_regr = regr.predict(validate_entree)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_regr = dask_glm.estimators.LinearRegression()
big_regr_train = big_regr.fit(big_train_entree, big_train_sortie)
pred_big_regr = big_regr.predict(big_validate_entree)

# Calculer le RMSE et le R² sur le jeu de test.
# ---------- Utiliser une librairie usuelle
print('RMSE:' + str(mean_squared_error(validate_sortie, pred_regr)) + ' R2:' + str(r2_score(validate_sortie, pred_regr)))

# Le modèle a un faible pouvoir prédictif puisque son score R-Squared est vraiment faible,
# seule une variation de 2,1% des données est expliquée même il a un bon score RMSE à 3,88

#
# REGRESSION LOGISTIQUE
# 

### Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane)
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données
# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil
# ---------- Utiliser une librairie usuelle
fare_binaire = np.zeros(len(train_echantillon))
fare_binaire[train_echantillon['fare_amount'] > train_echantillon['fare_amount'].median()] = 1

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
fare_binaire_big = np.zeros(len(train_complete))
fare_binaire_big[train_complete['fare_amount'] > train_complete['fare_amount'].median()] = 1


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées
# ---------- Utiliser une librairie usuelle
log_reg = LogisticRegression()
log_reg.fit(train_echantillon_entree_scale, fare_binaire)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_log_reg = dask_ml.linear_model.LogisticRegression()
big_log_reg.fit(train_complete_entree_scale, fare_binaire_big)

# Tout les variables sont significatives au seuil 1%. Le score AIC est 6490567

### Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire
# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)
# ---------- Utiliser une librairie usuelle
train_echantillon['fare_binaire'] = fare_binaire
train_echantillon_log = train_echantillon.drop(['fare_amount'], axis = 1)

train, validate, test = np.split(pd.DataFrame(train_echantillon_log).sample(frac=1), [int(.6*len(pd.DataFrame(train_echantillon_log))), int(.8*len(pd.DataFrame(train_echantillon_log)))])

train_sortie = train[['fare_binaire']]
train_entree = train.drop(['fare_binaire'], axis = 1)

validate_sortie = validate[['fare_binaire']]
validate_entree = validate.drop(['fare_binaire'], axis = 1)

test_sortie = test[['fare_binaire']]
test_entree = test.drop(['fare_binaire'], axis = 1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_train, big_validation, big_test = train_complete.random_split([0.6, 0.2, 0.2], random_state = 100)

big_train_sortie = big_train[['fare_amount']]
big_train_entree = big_train.drop(['fare_amount'], axis = 1)

big_validate_sortie = big_validate[['fare_amount']]
big_validate_entree = big_validate.drop(['fare_amount'], axis = 1)

big_test_sortie = big_test[['fare_amount']]
big_test_entree = big_test.drop(['fare_amount'], axis = 1)

# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 
# ---------- Utiliser une librairie usuelle
log_reg_train = LogisticRegression()
log_reg_train.fit(train_entree, train_sortie)
pred_log_test = log_reg.predict(test_entree)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_log_reg_train = dask_ml.linear_model.LogisticRegression()
big_log_reg_train.fit(big_train_entree, big_train_sortie)
pred_big_log_test = big_log_reg.predict(big_test_entree)

# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.
# ---------- Utiliser une librairie usuelle
confusion_mat = confusion_matrix(test_sortie, pred_log_test)
print(confusion_mat)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
confusion_mat_big = confusion_matrix(big_test_sortie, pred_big_log_test)
print(confusion_mat_big)

# Le modèle a correctement prédit si une observation appartient ou non à la première classe seulement 51,8%
# Le modèle a un faible pouvoir prédictif

