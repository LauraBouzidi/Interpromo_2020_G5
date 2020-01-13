# coding: utf-8
"""
Created on Mon Jan  6 10:26:49 2020
Group 5
@authors: Group Prediction
"""

import pandas as pd
from langdetect import detect
from tqdm import tqdm_notebook
import spacy
from tqdm import tqdm
import spacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')


train = pd.read_excel("Data/ALL_DATA.xlsx")
train = train[train["Review"].notnull()]

train.drop([14635,70282,123088,123158,123244], axis = 0,inplace = True)

i = 1
idx_to_drop = []
for idx,com in enumerate(train['Review'].iloc[range(139315)]):
    print(idx)
    if len(com.strip()) == 0:
        idx_to_drop.append(idx)
    elif detect(com) != 'en':
        idx_to_drop.append(idx)

train.drop(idx_to_drop, axis = 0,inplace = True)

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

def preprocessing(commentary):
    """
    Documentation
    
    Parameters:
        commentary: a commentary from 'OTHER_DATA_ANNOTATE.xlsx'
    
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


def TF_IDF(data,column_name):
    vectorizer = TfidfVectorizer(stop_words="english",min_df=50) #,max_df=0.7)
    X = vectorizer.fit_transform(list(data[column_name].values.astype('U')))
    print('Taille : ', X.shape)
    tf_idf_data = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    return (tf_idf_data)

train["ReviewClean"] = ""
for i in tqdm_notebook(range(len(train))):
    train["ReviewClean"].iloc[i] = preprocessing(train['Review'].iloc[i])

data.drop(data.index[100000:136303],0,inplace=True)
data = TF_IDF(data,"ReviewClean")

train.to_csv("Data/ALL_DATA_PREPROCESS_V1.csv")

