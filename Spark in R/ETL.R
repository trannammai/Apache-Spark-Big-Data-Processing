# IMPORTATION DES LIBRAIRIES ET VERIFICATION DE DEMARRAGE
# Parametrage des variables d'environnement
rm(list = ls())
Sys.setenv("JAVA_HOME" = "C:\\Progra~1\\Java\\jre1.8.0_181")
Sys.getenv("JAVA_HOME")

# Importation des librairies R et demarrage d'une session Spark en local
library(magrittr)
library(sparklyr)
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))

# Verification de la version de Spark et du nom de l'application
sparkR.version()

# IMPORT DU JEU DE DONNEES

### Indiquer le dossier et le fichier cible
dossier = "C:/Tran_Nam_Mai/Github Profile/Data-analysis-using-SparkR/data"
fichier = paste0(dossier, "Infections.txt")

### Importer les jeux de donnees 
### Prediction de maladies sur un jeu de donnees de maladies aux Etats-Unis
# Importation depuis le fichier texte 'Infections.txt'
data_spark = read.text(fichier)

# PREPARATION/NETTOYAGE DU JEU DE DONNEES
### Preparer les donnees
# Afficher la taille du jeu de donnees initial (nombre de lignes et colonnes)
dim(data_spark)

# Afficher les 3 premieres lignes du jeu de donnees (entieres, non abregees)
head(data_spark, num = 3L)

# Creer un nouveau jeu de donnees en extrayant de la chaine de caracteres les variables 'Hopital' (variable d'indice 1), 'Ville' (indice 3),
# 'Etat' (indice 4), 'NomMesure' (indice 8), 'IDMesure' (indice 9), 'Comparaison' (indice 10), 'Score' (indice 11)
separated_data_spark = selectExpr(data_spark, "split(value, '\t') AS value")

values = select(separated_data_spark, explode(separated_data_spark$value)) %>% distinct() %>% collect() %>% extract2(1)
exprs = lapply(values, function(x) alias(array_contains(separated_data_spark$value, x), x))
data = select(separated_data_spark, exprs)
data = data %>% select(data$Score, data$Hopital, data$Ville, data$Etat, data$NomMesure, data$IDMesure, data$Comparaison)

data = read.df(paste0(dossier, "Infections.csv"), source = "csv", header = TRUE)

### Nettoyer les donnees
# Enlever la colonne 'value', les valeurs incorrectes et manquantes
# Nous sommes interesses par la variable 'Score' en premier lieu
data = dropna(data, how = c("any", "all"), minNonNulls = NULL, cols = NULL)
data_T = filter(data, data$Comparaison != "Not Available")

# La taille (nombre de lignes et colonnes) du nouveau jeu de donnees 
# Le nombre des valeurs distinctes pour les variables 
dim(data)
temp_data = data %>% select(data$Hopital, data$IDMesure, data$Comparaison)
distinct_value = lapply(names(temp_data), function(x) alias(countDistinct(temp_data[[x]]), x))
head(do.call(agg, c(x = temp_data, distinct_value)))

# Le Schema du jeu de donnee nettoye
str(data)

# Modifier le schema de telle sorte a avoir la variable 'Score' comme numerique
data = withColumn(data, "Score", cast(data$Score, "numeric"))
str(data)

### Description des donnees
# Obtenir les caracteristiques statistiques de base des variables 
showDF(describe(data))

# Visualiser la distribution (histogramme) de la variable 'Score', a partir d'un échantillon 
# correspondant a 1% du jeu de donn?es total nettoye dans les etapes precedentes
data_sample = sample(data, FALSE, 0.01, 1000)
hist_Score = histogram(data_sample, "Score", nbins = 10)
require(ggplot2)
ggplot(hist_Score, aes(x = bins, y = counts)) + geom_bar(stat = "identity") + xlab("Frequency") + ylab("Sepal_Length")   

# OPERATIONS SUR DES DATAFRAMES

### Afficher les elements de base sur le Dataframe - Transformer le Dataframe en RDD
printSchema(data)
head(data, 5)
showDF(data, 2, truncate = TRUE)
columns(data)
length(columns(data))

data_RDD = SparkR:::toRDD(data)

### Réaliser des op?rations de base sur le Dataframe
# Les valeurs moyennes et medianes de la variable 'Score'
stat_Score = summary(data %>% select("Score"), "mean", "50%")
showDF(stat_Score)

# Median de la variable 'Score' est 2
# Moyenne de la variable 'Score' est 3427.2101

# Afficher un dataframe avec le nombre de cas (i.e. lignes) et des statistiques sur la variable 'Score'
# (moyenne, mediane et variance), agrege/groupe par Etat

showDF(data %>% groupBy('Etat') %>%
         summarize(num_cas = count(data$Etat),
                   avg_Score = avg(data$Score),
                   med_Score = expr("percentile(data$Scote, 0.5)"),
                   var_Score = var_samp(data$Score)) %>%
         arrange(desc(var_Score))
)

# L'etat a la plus grande dispersion des valeurs de 'Score'?
# C'est NY avec variance de 8.241

### Realiser des jointures entre plusieurs Dataframes
# Creer un nouveau Dataframe contenant uniquement les 5 Etats ayant le plus de cas (i.e. de lignes dans le Dataframe)
data1 = data %>% select(data$Etat) %>% groupBy('Etat') %>%
  summarize(num_cas = count(data$Etat)) %>%
  arrange(desc(count(data$Etat))) %>%
  limit(5)

# Creer un nouveau Dataframe contenant uniquement les Etats ayant les valeurs de 'Score' superieures a la mediane
data2 = data %>% select(data$Etat, data$Score) %>% 
  filter("Score > 1")

# Joindre ces 2 Dataframes et enlever les colonnes faisant doublon
final = join(data1, data2, data1$Etat == data2$Etat)

# Quelle est la taille du Dataframe obtenu par la jointure?
dim(final)
# 40192 lignes et 3 colonnes

# REQUETES AVEC SPARK SQL
### Realiser des operations de base sur le Dataframe
# Les valeurs moyennes et medianes de la variable 'Score'
createOrReplaceTempView(data, "data_tmp")
stat = sql("SELECT AVG(Score) avg_Score, PERCENTILE_APPROX(Score, 0.5) med_Score FROM data_tmp")
showDF(stat)

# Moyennes de Score: 3427.2101
# Median de Score: 2

# Afficher le resultat d'une requete SQL retournant avec le nombre de cas (i.e. lignes) et des statistiques 
# sur la variable 'Score' (moyenne, mediane et variance), agrege/groupe par Etat, par ordre alphabetique croissant des Etats
stat_group_etat = sql("SELECT Etat, COUNT(Etat) num_cas, AVG(Score) avg_Score, PERCENTILE_APPROX(Score, 0.5) med_Score, VAR_SAMP(Score) as var_Score FROM data_tmp GROUP BY Etat ORDER BY var_Score DESC")
showDF(stat_group_etat)

# L'Etat a la plus grande dispersion des valeurs de 'Score'
# C'est NY avec variance de 8.242

### Realiser des jointures avec une requete SQL
# Creer un nouveau Dataframe contenant uniquement les 5 Etats ayant le plus de cas (i.e. de lignes dans le Dataframe)
stat_group_etat_1 = sql("SELECT Etat, COUNT(Etat) num_cas FROM data_tmp GROUP BY Etat ORDER BY num_Score DESC LIMIT 5")

# Creer un nouveau Dataframe contenant uniquement les valeurs de 'Score' superieures a la mediane
stat_group_etat_2 = sql("SELECT Etat, Score FROM data_tmp WHERE Score > (SELECT PERCENTILE_APPROX(Score, 0.5) FROM data_tmp)")

# Joindre ces 2 Dataframes et enlever les colonnes faisant doublon
createOrReplaceTempView(stat_group_etat_1, "Group1")
createOrReplaceTempView(stat_group_etat_2, "Group2")
final = sql("SELECT * FROM Group1 INNER JOIN Group2 ON Group1.Etat = Group2.Etat")