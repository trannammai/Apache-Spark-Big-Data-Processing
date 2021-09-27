
#
# IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 
library(readr)
library(bigmemory)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(GGally)
library(biganalytics)
library(caret)
library(biglm)
library(neuralnet)

#
# IMPORT DU JEU DE DONNEES
# 
### Indiquer le dossier et le fichier cible
setwd('C:/Tran_Nam_Mai/Github Profile/Apache-Spark-Big-Data-Processing/Data/')
echan = "train_echantillon.csv"
compl = "train.csv"

### Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)
train_echantillon = read_csv(echan)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)
train_complete = read.big.matrix("train.csv", header = TRUE, type = 'double')
train_complete = as.big.matrix(head(train_complete, 1000)) # Remove afterwards

#
# PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 
### Nettoyer et préparer les données
# Enlever les valeurs incorrectes ou manquantes (si pertinent)
# ---------- Utiliser une librairie usuelle
train_echantillon_no_na = na.omit(train_echantillon)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_no_na = train_complete[mwhich(train_complete, 1:8, NA, 'neq', 'OR'),]

# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie
# ---------- Utiliser une librairie usuelle
train_echantillon_no_na = train_echantillon_no_na %>% select(fare_amount, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_no_na = deepcopy(train_complete_no_na, -c(1,3,8))
options(bigmemory.allow.dimnames = TRUE)
colnames(train_complete_no_na) = colnames(train_echantillon_no_na)

# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes
# ---------- Utiliser une librairie usuelle
summary(train_echantillon_no_na)

remove_outliers = function(x, na.rm = TRUE, ...) {
  qnt = quantile(x, probs = c(.25, .75), na.rm = na.rm, ...)
  H = 1.5 * IQR(x, na.rm = na.rm)
  y = x
  y[x < (qnt[1] - H)] = NA
  y[x > (qnt[2] + H)] = NA
  y
}

train_echantillon_no_outliers = sapply(train_echantillon_no_na, remove_outliers)
train_echantillon_no_na_no_outliers = as.data.frame(na.omit(train_echantillon_no_outliers))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
colmin(train_complete_no_na)
colmax(train_complete_no_na)
colmean(train_complete_no_na)

median_index = nrow(train_complete_no_na_no_outliers)/2
(median_fare_amount = train_complete_no_na_no_outliers[median_index, 1, drop = TRUE])
(median_pickup_longitude = train_complete_no_na_no_outliers[median_index, 2, drop = TRUE])
(median_pickup_latitude = train_complete_no_na_no_outliers[median_index, 3, drop = TRUE])
(median_dropoff_longitude = train_complete_no_na_no_outliers[median_index, 4, drop = TRUE])
(median_dropoff_latitude = train_complete_no_na_no_outliers[median_index, 5, drop = TRUE])

train_complete_no_outliers = sapply(as.data.frame(as.matrix(train_complete_no_na)), remove_outliers)
train_complete_no_na_no_outliers = as.big.matrix(as.data.frame(na.omit(train_complete_no_outliers)))

# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)
# ---------- Utiliser une librairie usuelle
g1 = ggplot(train_echantillon_no_na_no_outliers, aes(x = fare_amount)) + geom_histogram(binwidth = .1, position = "dodge", colour = "blue", fill = "blue")
g2 = ggpairs(train_echantillon_no_na_no_outliers[,2:3])
g3 = ggpairs(train_echantillon_no_na_no_outliers[,4:5])
gridExtra::grid.arrange(g1, g2, g3, nrow = 1, ncol = 3)

# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")
# ---------- Utiliser une librairie usuelle
train_echantillon_entre = train_echantillon_no_na_no_outliers %>% select(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
train_echantillon_sortie =  train_echantillon_no_na_no_outliers %>% select(fare_amount)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_entre = deepcopy(train_complete_no_na_no_outliers, -1)
colnames(train_complete_entre) = colnames(train_echantillon_entre)

train_complete_sortie = deepcopy(train_complete_no_na_no_outliers, 1)
colnames(train_complete_sortie) = colnames(train_echantillon_sortie)

# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)
# ---------- Utiliser une librairie usuelle
train_echantillon_entre_scale = scale(train_echantillon_entre)
train_echantillon_sortie_scale = scale(train_echantillon_sortie)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_entre_scale = as.big.matrix(scale(as.matrix(train_complete_entre)))
train_complete_sortie_scale = as.big.matrix(scale(as.matrix(train_complete_sortie)))

#
# CLUSTERING DU JEU DE DONNEES
# 
### Réaliser un clustering k-means sur les données d'entrée standardisées
# ---------- Utiliser une librairie usuelle
kmeans_cluster = kmeans(train_echantillon_entre_scale, centers = 5, algorithm = "Lloyd")

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_kmeans_cluster = bigkmeans(train_complete_entre_scale, centers = 5, dist = "euclid")

### Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters
# ---------- Utiliser une librairie usuelle
WSS_func = function(k, option = 'WSS') {
  cluster = kmeans(train_echantillon_entre_scale, centers = k, algorithm = "Lloyd")
  return (cluster$tot.withinss)
}

WSS = sapply(1:10, WSS_func)
ggplot(data.frame(1:10, WSS), aes(x = 1:10, y = WSS)) + geom_point() + geom_line() + scale_x_continuous(breaks = seq(1, 10, by = 1))

R2_func = function(k, option = 'WSS') {
  cluster = kmeans(train_echantillon_entre_scale, centers = k, algorithm = "Lloyd")
  BSS = cluster$betweenss
  TSS = cluster$totss
  R_Squared = BSS/TSS*100
  return (R_Squared)
}

R2 = sapply(1:10, R2_func)
ggplot(data.frame(1:10, R2), aes(x = 1:10, y = R2)) + geom_point() + geom_line() + scale_x_continuous(breaks = seq(1, 10, by = 1))

# À partir des deux graphiques, nous pouvons voir que le nombre de cluster optimal est égal à deux, 
# là où la courbe commence à avoir un rendement décroissant. C'est aussi où on peut dire que partionner
# n'apporte plus grand chose.

kmeans_cluster = kmeans(train_echantillon_entre_scale, centers = 3, algorithm = "Lloyd")
print(kmeans_cluster)

# Il y a 3 clusters principaux selon la localisation 

### Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables
# ---------- Utiliser une librairie usuelle
index_plot = sample(nrow(train_echantillon_entre_scale), 1000)
pairs(train_echantillon_entre_scale[index_plot,], col = kmeans_cluster$cluster[index_plot], pch = 19)

#
# ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 
### Faire une ACP sur le jeu de données standardisé
# ---------- Utiliser une librairie usuelle
ACP_transform = prcomp(train_echantillon_entre_scale, center = TRUE, scale. = TRUE)
print(ACP_transform)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
install.packages('bigpca-master.zip', lib = 'C:/Tran_Nam_Mai/Master 2 - Statistique et Econometrie/Big Data/TP2', repos = NULL)
big_ACP_transform = bigpca::big.PCA(train_complete_entre_scale, thin = FALSE)

### Réaliser le diagnostic de variance avec un graphique à barre (barchart)
summary(ACP_transform)
plot(ACP_transform)

# On doit garder deux composantes principales parce qu'il représente déjà 82% des variances des composantes
# A partir du trosième composante, on ne peux apporter qu'un peu de variances.

### Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation
# ---------- Utiliser une librairie usuelle
ACP_transform_sample = prcomp(head(train_echantillon_entre_scale, 20000), center = TRUE, scale. = TRUE)
plot(ACP_transform_sample)
biplot(ACP_transform_sample)

#
# REGRESSION LINEAIRE
# 
### Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données
# ---------- Utiliser une librairie usuelle
reg = lm(fare_amount ~., data = train_echantillon_no_na_no_outliers)
summary(reg)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_reg = bigglm(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude, data = train_complete_no_na_no_outliers)
summary(big_reg)
deviance(big_reg)
AIC(big_reg)

# Tout les variables sont significatives au seuil 1%. 
# Cependant, seulement 2,1% de la variance trouvée dans la variable de réponse (fare_amount) 
# peut être expliquée par les variables prédictives.

### Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire
# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)
# ---------- Utiliser une librairie usuelle
set.seed(7000)
cut_level = sample(seq(1, 3), size = nrow(train_echantillon_no_na_no_outliers), replace = TRUE, prob = c(.6, .2, .2))
train = train_echantillon_no_na_no_outliers[cut_level == 1,]
test = train_echantillon_no_na_no_outliers[cut_level == 2,]
val = train_echantillon_no_na_no_outliers[cut_level == 3,]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_cut_level = sample(seq(1, 3), size = nrow(train_complete_no_na_no_outliers), replace = TRUE, prob = c(.6, .2, .2))
big_train = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 1,])
big_test = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 2,])
big_val = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 3,])

# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 
# ---------- Utiliser une librairie usuelle
reg_train = lm(fare_amount ~., data = train)
pred_reg_val = predict(reg_train, newdata = val)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
reg_big_train = bigglm(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude, data = big_train)
pred_reg_big_val = predict(reg_big_train, newdata = as.data.frame(as.matrix(big_val)))

# Calculer le RMSE et le R² sur le jeu de test.
# ---------- Utiliser une librairie usuelle
RMSE(pred_reg_val, val$fare_amount)
R2(pred_reg_val, val$fare_amount)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
RMSE(pred_reg_big_val, as.matrix(deepcopy(big_val, 1)))
R2(pred_reg_big_val, as.matrix(deepcopy(big_val, 1)))

# Le modèle a un faible pouvoir prédictif puisque son score R-Squared est vraiment faible, 
# seule une variation de 2,1% des données est expliquée même il a un bon score RMSE à 3,88

#
# REGRESSION LOGISTIQUE
# 

### Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données
# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil
# ---------- Utiliser une librairie usuelle
train_echantillon_no_na_no_outliers = train_echantillon_no_na_no_outliers %>%
  mutate(fare_binaire = ifelse(fare_amount > quantile(fare_amount, 0.5), 1, 0)) %>%
  select(-fare_amount)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_complete_no_na_no_outliers = as.data.frame(as.matrix(train_complete_no_na_no_outliers)) %>% mutate(fare_binaire = ifelse(fare_amount > quantile(fare_amount, 0.5), 1, 0)) %>% select(-fare_amount)
train_complete_no_na_no_outliers = as.big.matrix(train_complete_no_na_no_outliers)

# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées
# ---------- Utiliser une librairie usuelle
logit = glm(fare_binaire~., data = train_echantillon_no_na_no_outliers, family = binomial(link = "logit"))
summary(logit)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_logit = bigglm(fare_binaire ~ pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude, family = binomial(link = "logit"), data = big_train, chunksize = 1000, maxit = 10)
summary(big_logit)

### Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?
# Réponse:
# Tout les variables sont significatives au seuil 1%. Le score AIC est 6490567

### Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire
# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)
# ---------- Utiliser une librairie usuelle
set.seed(7000)
cut_level = sample(seq(1, 3), size = nrow(train_echantillon_no_na_no_outliers), replace = TRUE, prob = c(.6, .2, .2))
train = train_echantillon_no_na_no_outliers[cut_level == 1,]
test = train_echantillon_no_na_no_outliers[cut_level == 2,]
val = train_echantillon_no_na_no_outliers[cut_level == 3,]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_cut_level = sample(seq(1, 3), size = nrow(train_complete_no_na_no_outliers), replace = TRUE, prob = c(.6, .2, .2))
big_train = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 1,])
big_test = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 2,])
big_val = as.big.matrix(train_complete_no_na_no_outliers[big_cut_level == 3,])

# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 
# ---------- Utiliser une librairie usuelle
logit_train = glm(fare_binaire ~., data = train, family = binomial(link = "logit"))
pred_logit_val = predict(logit_train, newdata = val)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
big_logit_train = bigglm(fare_binaire ~ pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude, data = big_train, family = binomial(link = "logit"))
pred_big_logit_val = predict(big_logit_train, newdata = as.data.frame(as.matrix(big_val)))

# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.
# ---------- Utiliser une librairie usuelle
pred_logit_val_binary = ifelse(pred_logit_val > 0.5, 1, 0)
cm = caret::confusionMatrix(factor(pred_logit_val_binary), factor(val$fare_binaire, levels = c("0", "1")))
cm[2] # Matrice de confusion
cm[3]$overall[1] # Accuracy
temp[4,]$table.Freq /(temp[4,]$table.Freq + temp[2,]$table.Freq) # Precision

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
prediction_val_binaire = rep(0,length(pred_big_logit_val))
prediction_val_binaire[pred_big_logit_val > 0.5] = 1
matrice_confusion = table(as.matrix(deepcopy(big_val, 5)), prediction_val_binaire)
round(matrice_confusion * 100 / sum(matrice_confusion), 1) # Matrice de confusion
(matrice_confusion[2,2] + matrice_confusion[1,1])/(matrice_confusion[1,1] + matrice_confusion[1,2] + matrice_confusion[2,2]+ matrice_confusion[2,1]) # Accuracy
matrice_confusion[2,2]/(matrice_confusion[2,2]+ matrice_confusion[1,2]) # Precision

# Le modèle a correctement prédit si une observation appartient ou non à la première classe seulement 51,8%
# Le modèle a un faible pouvoir prédictif