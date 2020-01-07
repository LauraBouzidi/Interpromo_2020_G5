# coding: utf-8
"""
Created on Mon Jan  6 10:26:49 2020
Group 5
@authors: Group Prediction
"""

#Importation des données
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



#Chargement des données
data = pd.read_csv("Data/ALL_DATA_CLEAN_V1.csv", sep = ";")

#Liste des colonnes à drop
col_to_drop = ["Aircraft_Type4","Aircraft_Type3","Date_Visit","Aircraft_Type2","Seat_Type","Type_Of_Lounge","Airport","Category_Detail"
                ,"Category","Airline_Type","Region_Operation","Seat","Aircraft_Type1","Date_Flown","Arrival","Departure"
               ,"Airline_Name","Stop"]

#On drop ces colonnes
data.drop(col_to_drop, axis = 1, inplace = True)

#On remplace les valeurs manquantes
data["Recommended"].fillna(0,inplace = True)
data["Type_Of_Traveller"].fillna("Others",inplace = True)
data["Cabin_Class"].fillna("Economy Class",inplace = True)

#On arrondie les valeurs
data["Overall_Customer_Rating"]=np.floor(data["Overall_Customer_Rating"])

#On transforme les variables qualitatives en variables quantitatives
data = pd.get_dummies(data, columns = ["Data_Source","Cabin_Class","Type_Of_Traveller"])

#Fonction pour calculer l'erreur
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# # Création des modèles (Pour le confort des sièges)


#Separation en set de train et test:
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(["Seat_Comfort"], axis = 1), data["Seat_Comfort"],
                                                    test_size=0.30, random_state=42)


# ## Model KNN (Pour le confort des sièges)


#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#Fonction pour calculer l'erreur
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#On calcule la RMSE
rmse(Prediction,Y_test)

#Creation d'un dataframe des prédictions:
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction["Label"]),Y_test)


# ## Modèle Régression Logistique (Pour le confort des sièges)


#On entraîne le modèle
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
prediction_logistic = model_logistic.predict(X_test)

#On calcule la RMSE
rmse(prediction_logistic,Y_test)

#Creation d'un dataframe des prédictions
prediction_logistic = pd.DataFrame(prediction_logistic, columns = ["Label"])
prediction_logistic["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction_logistic["Label"]),Y_test)


# ## Modèle Random Forest (Pour le confort des sièges)


#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
rmse(prediction_random_regressor,Y_test)

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.index

#Calcule du Kappa
cohen_kappa_score(np.round(prediction_random_regressor["Label"]),Y_test)


# # Création des modèles (Pour la nourriture)


#Separation en set de train et test:
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(["Seat_Comfort"], axis = 1), data["Seat_Comfort"],
                                                    test_size=0.30, random_state=42)


# ## Model KNN (Pour la nourriture)


#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#On calcule la RMSE
rmse(Prediction,Y_test)

#Creation d'un dataframe des prédictions
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction["Label"]),Y_test)


# ## Modèle Régression Logistique (Pour la nourriture)


#On entraîne le modèle
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
prediction_logistic = model_logistic.predict(X_test)

#On calcule la RMSE
rmse(prediction_logistic,Y_test)

#Creation d'un dataframe des prédictions
prediction_logistic = pd.DataFrame(prediction_logistic, columns = ["Label"])
prediction_logistic["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction_logistic["Label"]),Y_test)


# ## Modèle Random Forest (Pour le confort des sièges)


#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
rmse(prediction_random_regressor,Y_test)

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.index

#Calcule du Kappa
cohen_kappa_score(np.round(prediction_random_regressor["Label"]),Y_test)


# # Création des modèles (Pour la note générale)


X_train, X_test, Y_train, Y_test = train_test_split(data.drop(["Overall_Customer_Rating"], axis = 1), data["Overall_Customer_Rating"]
                                                    , test_size=0.30, random_state=42)


# ## Model KNN (Pour la note générale)


#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#On calcule la RMSE
rmse(Prediction,Y_test)

#Creation d'un dataframe des prédictions
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction["Label"]),Y_test)


# ## Modèle Régression Logistique (Pour la note générale)


#On entraîne le modèle
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
prediction_logistic = model_logistic.predict(X_test)

#On calcule la RMSE
rmse(prediction_logistic,Y_test)

#Creation d'un dataframe des prédictions
prediction_logistic = pd.DataFrame(prediction_logistic, columns = ["Label"])
prediction_logistic["index"] = X_test.index

#Calcul du Kappa:
cohen_kappa_score(np.round(prediction_logistic["Label"]),Y_test)


# ## Modèle Random Forest (Pour la note générale)


#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
rmse(prediction_random_regressor,Y_test)

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.index

#Calcule du Kappa
cohen_kappa_score(np.round(prediction_random_regressor["Label"]),Y_test)

