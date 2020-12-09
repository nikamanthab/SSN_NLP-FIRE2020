'''
Runs for AILA 2020 task2
authors: Nitin Nikamanth Appiah Balaji, Bharathi B, Bhuvana J
'''
import pandas as pd
import numpy as np

'''
sklearn imports
'''
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from pymagnitude import *
from sentence_transformers import SentenceTransformer
from sklearn_crfsuite import CRF

'''
FastText model imports
'''
from pymagnitude import *
from nltk import word_tokenize

'''
Loading the prepared tsv file
'''
train = pd.read_csv('../../tsv/task2-train/train.tsv','\t')
X_ori = train['text']
y = train['labels']

'''
Generating word TFIDF vectorizer and vectorizing train documents to vectors
'''
vectorizer = TfidfVectorizer(ngram_range=(1,5))
X = vectorizer.fit_transform(X_ori)

'''
Fitting models with k fold cross validation
printing accuracy and f1 scores
'''
clf = LogisticRegression(verbose=True,n_jobs=-1)
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(clf, X, y, scoring=scoring,cv=5)
print('acc:',np.average(scores['test_accuracy']))
print('f1:',np.average(scores['test_f1_macro']))

clf = RandomForestClassifier(n_jobs=-1)
scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy']
scores = cross_validate(clf, X, y, scoring=scoring,cv=5)
print('acc:',np.average(scores['test_accuracy']))
print('f1:',np.average(scores['test_f1_macro']))


'''
Loading pretrained FastText model
Converting word embedding to sentence embedding by averaging the generated word vectors
'''
fast = Magnitude("Magnitude_files/fasttext-magnitude/english.magnitude")
def fasttext(x):
    vectors = []
    for title in tqdm_notebook(x):
        vectors.append(np.average(fast.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)
X_train = fasttext(X_ori)

'''
Fitting MLP model on the FastText embedding vectors 
'''
clf = MLPClassifier(hidden_layer_sizes=(512,128),verbose=True)
scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
scores = cross_validate(clf, X_train, y, scoring=scoring,cv=3)
print('f1:',np.average(scores['test_f1_macro']))
print('MLP model detailed scores:')
print(scores)


