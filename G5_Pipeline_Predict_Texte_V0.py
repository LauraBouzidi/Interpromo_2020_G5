# coding: utf-8
"""
Created on Mon Jan  6 10:26:49 2020
Group 5
@authors: Prediction Texte
"""

#Importation des données
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import en_core_web_sm
from tqdm import tqdm_notebook


# # Mise en place du preprocessing

#On charge les données
train = pd.read_excel("Data/ALL_DATA.xlsx")
#On enlève les individus qui ont des NaN dans la variable Review
train = train[train["Review"].notnull()]

#On drop les conmmentaires qui perturbent la suite du code
train.drop([14635,70282,123088,123158,123244], axis = 0,inplace = True)

# On chope l'index de tous les commentaires qui ne sont pas en anglais
i = 1
idx_to_drop = []
for idx,com in enumerate(train['Review'].iloc[range(139315)]):
    print(idx)
    if len(com.strip()) == 0:
        idx_to_drop.append(idx)
    elif detect(com) != 'en':
        idx_to_drop.append(idx)

#On drop tout les commentaires en anglais
train.drop(idx_to_drop, axis = 0,inplace = True)

#Fonction qui gère la lemmatization
nlp = en_core_web_sm.load()

def lemmatize(sentence):
    """
    Documentation
    
    Parameters:
        sentence: a sentence from a commentary
    
    Output:
        lemmatized: the lemmatized sentence
    """
    
    s = nlp(sentence)
    lemmatized = ''
    for w in s:
        lemmatized += w.lemma_ + ' '
        
    return lemmatized

#Fonction qui réalise le preprocessing
def preprocessing(commentary):
    """
    Documentation
    
    Parameters:
        commentary: a commentary from 'ALL_DATA.xlsx'
    
    Output:
        result: the cleaned commentary
    """
    #We drop some texte
    commentary = commentary.replace('Points positifs', ' ').replace('Points négatifs', ' ') # delete "points négatifs et positifs"
    #We drop some texte
    commentary = commentary.replace('Trip Verified', ' ').replace('Not Verified',' ').replace('Verified Review',' ').replace('|',' ') # delete "Verified"
    # Convert text to lowercase
    commentary = commentary.lower()
    #We lemmatize
    commentary = lemmatize(commentary)
    commentary = commentary.replace('-PRON-', ' ').replace("✅", " ")
    return commentary.strip()

#On réalise le preprocessing sur les donnée
train["ReviewClean"] = ""
for i in tqdm_notebook(range(len(train))):
    train["ReviewClean"].iloc[i] = preprocessing(train['Review'].iloc[i])

#On unifie le code
data = train


# # Création modèles


#On enlève toutes les colonnes qui ne nous servent à rien
data = pd.DataFrame(data[["ReviewClean","Seat_Comfort","Food_And_Beverages","Cabin_Staff_Service"]],columns = ["ReviewClean","Seat_Comfort","Food_And_Beverages","Cabin_Staff_Service"])

#Fonction qui réalise le TF-IDF
def TF_IDF(data,column_name):
    vectorizer = TfidfVectorizer(stop_words="english",min_df=50) #,max_df=0.7)
    X = vectorizer.fit_transform(list(data[column_name].values.astype('U')))
    print('Taille : ', X.shape)
    tf_idf_data = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    return (tf_idf_data)

#On supprime la moitié de nos données pour faire tourner le TF-IDF sur la moitié des données
data.drop(data.index[70000:136303],0,inplace=True)

#On applique le TF_IDF sur les data:
data = TF_IDF(data,"ReviewClean")

#On créer un dataframe par label:

data_Seat = data.drop["Food_And_Beverages","Cabin_Staff_Service"], axis = 1)

data_Food = data.drop["Seat_Comfort","Cabin_Staff_Service"], axis = 1)

data_Staff = data.drop["Food_And_Beverages","Seat_Comfort"], axis = 1)

#On delete les valeurs manquantes
data_Seat = data_Seat[data_Seat["Seat_Comfort"].notnull()]

data_Food = data_Food[data_Food["Food_And_Beverages"].notnull()]

data_Staff = data_Staff[data_Staff["Cabin_Staff_Service"].notnull()]

#Fonction pour calculer l'erreur
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# # Création des modèles (Pour le confort des sièges)


#Separation en set de train et test:
X_train, X_test, Y_train, Y_test = train_test_split(data_Seat.drop(["Seat_Comfort"], axis = 1), data_Seat["Seat_Comfort"],
                                                    test_size=0.30, random_state=42)


# ## Model KNN (Pour le confort des sièges)

#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#On calcule la RMSE
print(rmse(Prediction,Y_test))

#Creation d'un dataframe des prédictions:
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
print(accuracy_score(np.round(prediction["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_KNN_Seat_Comfort.sav'
pickle.dump(KNN, open(filename, 'wb'))


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
print(accuracy_score(np.round(prediction_logistic["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_RegLog_Seat_Comfort.sav'
pickle.dump(model_logistic, open(filename, 'wb'))


# ## Modèle Random Forest (Pour le confort des sièges)

#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
print(rmse(prediction_random_regressor,Y_test))

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.inde


#Calcule du Kappa
print(accuracy_score(np.round(prediction_random_regressor["Label"]),Y_test))

#Enregistrement du modele:

filename = 'Modeles/G5_RF_Seat_Comfort.sav'
pickle.dump(modele_random_regressor, open(filename, 'wb'))


# # Création des modèles (Pour la nourriture)

#Separation en set de train et test:
X_train, X_test, Y_train, Y_test = train_test_split(data_Food.drop(["Food_And_Beverages"], axis = 1), data_Food["Food_And_Beverages"],
                                                    test_size=0.30, random_state=42)


# ## Model KNN (Pour la nourriture)

#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#On calcule la RMSE
print(rmse(Prediction,Y_test))

#Creation d'un dataframe des prédictions
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
print(accuracy_score(np.round(prediction["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_KNN_Food_And_Beverages.sav'
pickle.dump(KNN, open(filename, 'wb'))


# ## Modèle Régression Logistique (Pour la nourriture)

#On entraîne le modèle
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
prediction_logistic = model_logistic.predict(X_test)

#On calcule la RMSE
print(rmse(prediction_logistic,Y_test))

#Creation d'un dataframe des prédictions
prediction_logistic = pd.DataFrame(prediction_logistic, columns = ["Label"])
prediction_logistic["index"] = X_test.index

#Calcul du Kappa:
print(accuracy_score(np.round(prediction_logistic["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_RegLog_Food_And_Beverages.sav'
pickle.dump(modele_logistic, open(filename, 'wb'))


# ## Modèle Random Forest (Pour food)

#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
print(rmse(prediction_random_regressor,Y_test))

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.index

#Calcule du Kappa
print(accuracy_score(np.round(prediction_random_regressor["Label"]),Y_test))

#Enregistrement du modele:

filename = 'Modeles/G5_RF_Food_And_Beverages.sav'
pickle.dump(modele_random_regressor, open(filename, 'wb'))


# # Création des modèles (Pour la note générale)


X_train, X_test, Y_train, Y_test = train_test_split(data_Staff.drop(["Cabin_Staff_Service"], axis = 1), data_Staff["Cabin_Staff_Service"]
                                                    , test_size=0.30, random_state=42)


# ## Model KNN (Pour la note générale)

#Def de KNN + fitting:
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

#Predictions:
Prediction = KNN.predict(X_test)

#On calcule la RMSE
print(rmse(Prediction,Y_test))

#Creation d'un dataframe des prédictions
prediction = pd.DataFrame(Prediction, columns = ["Label"])
prediction["index"] = X_test.index

#Calcul du Kappa:
print(accuracy_score(np.round(prediction["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_KNN_Cabin_Staff_Service.sav'
pickle.dump(KNN, open(filename, 'wb'))


# ## Modèle Régression Logistique (Pour la note générale)

#On entraîne le modèle
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
prediction_logistic = model_logistic.predict(X_test)

#On calcule la RMSE
print(rmse(prediction_logistic,Y_test))

#Creation d'un dataframe des prédictions
prediction_logistic = pd.DataFrame(prediction_logistic, columns = ["Label"])
prediction_logistic["index"] = X_test.index

#Calcul du Kappa:
print(accuracy_score(np.round(prediction_logistic["Label"]),Y_test))

# save the model to disk
filename = 'Modeles/G5_RegLog_Cabin_Staff_Service.sav'
pickle.dump(model_logistic, open(filename, 'wb'))


# ## Modèle Random Forest (Pour la note générale)

#On entraîne le modèle
model_random_regressor = RandomForestRegressor(max_depth=5, random_state=42, n_estimators = 100)
model_random_regressor.fit(X_train,Y_train)
#On prédit
prediction_random_regressor = model.predict(X_test)

#Calcule de la RMSE
print(rmse(prediction_random_regressor,Y_test))

#Creation d'un dataframe des prédictions:
prediction_random_regressor = pd.DataFrame(prediction_random_regressor, columns = ["Label"])
prediction_random_regressor["index"] = X_test.index

#Calcule du Kappa
print(accuracy_score(np.round(prediction_random_regressor["Label"]),Y_test))

#Enregistrement du modele:

filename = 'Modeles/G5_RF_Cabin_Staff_Service.sav'
pickle.dump(modele_random_regressor, open(filename, 'wb'))

