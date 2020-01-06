# coding: utf-8
"""
Created on Mon Jan  6 10:26:49 2020
Group 5
@authors: Group Prediction
"""


#Importation des données
import pandas as pd
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


#Chargement des données
train = pd.read_excel('Data/ALL_DATA.xlsx')


#On drop certaines colonnes
train.drop(["Type_Of_Traveller","Date_Visit","Date_Flown","Airport","Route","Category","Category_Detail","Date_Review"
            ,"Review","Type_Of_Lounge","Airline_Name","Aircraft_Type","Seat"
            ,"Overall_Customer_Rating"], axis = 1, inplace = True)


#On transforme les variables qualitatives en variables quantitatives
train = pd.get_dummies(train, columns = ["Data_Source","Airline_Type","Region_Operation","Cabin_Class"
                                        ,"Seat_Type","Recommended"])



#On remplace les valeurs manquantes
for i in train.columns:
    train[i].fillna(train[i].median(),inplace = True)



#On divise les données en train et en test
X_train, X_test, Y_train, Y_test = train_test_split(train.drop(["Seat_Comfort"], axis = 1), train["Seat_Comfort"]
                                                    , test_size=0.30, random_state=42)



#On entraîne le modèle et on fournit la prédiction
model = LinearRegression(normalize = True)
model.fit(X_train,Y_train)
prediction = model.predict(X_test)



#Fonction pour calculer l'erreur
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



#On calcule l'erreur
rmse(prediction,Y_test)


prediction = pd.DataFrame(prediction, columns = ["Label"])
prediction["index"] = X_test.index
prediction.to_csv("Prediction_Seat_Comfort.csv", index = False)