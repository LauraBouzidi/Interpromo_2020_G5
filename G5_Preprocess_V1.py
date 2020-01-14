"""
    Created on 7 January 2020
    Group 5
    Authors : A.C, M.C, J.J
"""


# # Imports
from tqdm import tqdm
import pandas as pd
import string
import numpy as np
import sklearn
import spacy
import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
import scipy
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer
from langdetect import detect
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')


# # Functions
def database(file, column):
    #data = pd.read_excel("../data/" + file + ".xlsx") # import data
    data = pd.read_excel("ALL_DATA.xlsx") # import data
    data = data[data["Review"].notnull()]
    data['index'] = range(len(data)) # create index
    data = data.fillna('')
    sentences = data.loc[:,lambda data: ["index", column]]
    return sentences


def clean_data(sentences, column):
    nlp_en = spacy.load('en_core_web_sm')

    sentences_clean = []
    
    for i in tqdm(sentences["index"]):
        sentence = sentences[column][i].replace('Points positifs', ' ').replace('Points négatifs', ' ') # delete "points négatifs et positifs"
        
        sentence = sentence.replace('Trip Verified', ' ').replace('Not Verified',' ').replace('Verified Review',' ') # delete "Verified"
        sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', sentence) # delete link
        sentence = sentence + ' ' + ' '.join(re.findall(r"#(\w+)", sentence)) # find hashtag and doubled
        sentence = sentence + ' ' + ' '.join(re.findall(r"@(\w+)", sentence)) # find @ and doubled
        sentence = strip_punctuation(sentence) # delete punctuation
        semi_clean_sentence = ''
        comments = nlp_en(sentence.lower()) # lower comments
        if len(comments) != 0: # no-empty comments
            try:
                if detect(str(comments)) == 'en': # english comments
                    for token in comments:
                        semi_clean_sentence = semi_clean_sentence + token.lemma_ + ' ' # add lemmatizer
                    semi_clean_sentence = semi_clean_sentence.replace('-PRON-', '') # delete "PRON"
                    semi_clean_sentence = remove_stopwords(strip_short(semi_clean_sentence)) # delete shorts words
                    sentences_clean.append([i, semi_clean_sentence])
            except:
                print(i)
    return sentences_clean


def create_tfidf(sentences_clean, file):
    comments = [i[1] for i in sentences_clean] # recover comments
    index = [i[0] for i in sentences_clean]  # recover index

    vectorizer = TfidfVectorizer(stop_words="english",min_df=0.10)#, max_df=0.7)
    X = vectorizer.fit_transform(comments)

    M = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()) # creation of matrice

    tfidf = np.concatenate((pd.DataFrame(index), pd.DataFrame(comments), M), axis=1) # add comments to tfidf
    col = vectorizer.get_feature_names()
    col = ['index','commentaire'] + col # rename columns

    pd.DataFrame(tfidf, columns = col).set_index('index').to_csv("ALL_DATA_Processed.csv", sep = ",") # save matrice
    
    print('Matrice TF-IDF de ' + file + ' enregistrée')

def TF_IDF(file, column = 'Review'):
    sentences = database(file, column)
    sentences_clean = clean_data(sentences, column)
    create_tfidf(sentences_clean, file)


# # Tests

TF_IDF("ALL_DATA.xlsx")

