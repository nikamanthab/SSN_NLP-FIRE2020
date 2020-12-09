'''
Runs for Fake news detection in the Urdu language (UrduFake) 2020
authors: Nitin Nikamanth Appiah Balaji, Bharathi B
'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
import re
'''
sklearn imports - classifier and vectorizer
'''
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

'''
BERT model import
'''
from sentence_transformers import SentenceTransformer

'''
FastText mdoel imports
'''
from pymagnitude import *
from nltk import word_tokenize

'''
Word2Vec gensim model imports
'''
from gensim.models import Word2Vec
from nltk.corpus import stopwords


'''
Converting corpus files to arrays of text and arrays of label
'''
train = pd.DataFrame()
test = pd.DataFrame()
X_train = []
y_train = []
X_test = []
y_test = []
root = '../Datasets-for-Urdu-news-master/Urdu Fake News Dataset/1.Corpus/'
for file in os.listdir(root+'Train/Real/'):
    with open(root+'Train/Real/'+file) as f:
        data = f.read()
        X_train.append(data)
        y_train.append(1)
for file in os.listdir(root+'Train/Fake/'):
    with open(root+'Train/Fake/'+file) as f:
        data = f.read()
        X_train.append(data)
        y_train.append(0)

for file in os.listdir(root+'Test/Real/'):
    with open(root+'Test/Real/'+file) as f:
        data = f.read()
        X_test.append(data)
        y_test.append(1)
for file in os.listdir(root+'Test/Fake/'):
    with open(root+'Test/Fake/'+file) as f:
        data = f.read()
        X_test.append(data)
        y_test.append(0)
        
'''
Cleaning the text - removing unwanted characters
'''
X_train_sen = []
X_test_sen = []
for doc in X_train:
    new_doc = []
    for sentence in doc.split('\n'):
        sentence = sentence.replace('\t', ' ')
        sentence = re.sub(' +',' ', sentence)
        if len(sentence.split(' ')) > 2:
            new_doc.append(sentence)
    con_sen = ' '.join(new_doc)
    X_train_sen.append(con_sen)

for doc in X_test:
    new_doc = []
    for sentence in doc.split('\n'):
        sentence = sentence.replace('\t', ' ')
        sentence = re.sub(' +',' ', sentence)
        if len(sentence.split(' ')) > 2:
            new_doc.append(sentence)
    con_sen = ' '.join(new_doc)
    X_test_sen.append(con_sen)

    
'''
Assigning dataframe and shuffling the dataframe 
to create a proper distribution for training
'''
train['X'] = X_train_sen
train['y'] = y_train
test['X'] = X_test_sen
test['y'] = y_test

train = train.sample(frac=1)
test = test.sample(frac=1)

X_train_sen = train['X']
X_test_sen = test['X']
y_train = train['y']
y_test = test['y']

def printDetails(feature_name):
    '''
    input: feature name
    output: print details and formating
    '''
    print('---------------------------------')
    print("Feature:", feature_name)

def runClassifier(X_train_ori, y_train, X_test_ori, y_test):
    '''
    input: train, test - X/y
    output: prints the performance accuracy and f1 score
    '''
    clfs = []
    
    clfs.append(('Randomforest',RandomForestClassifier(n_estimators=5000,n_jobs=-1,verbose=True,max_depth=10000)))
    clfs.append(('Extratrees',ExtraTreesClassifier(n_estimators=10000, n_jobs=-1)))
    clfs.append(('GradientBoosting',GradientBoostingClassifier(n_estimators=1000,verbose=1)))
    clfs.append(('AdaBoosting',AdaBoostClassifier(n_estimators=10000)))
    clfs.append(('MLP',MLPClassifier(verbose=True, hidden_layer_sizes=(1024,512,128),max_iter=500)))
    
    for name, clf in clfs:
        clf.fit(X_train_ori, y_train)
        pred = clf.predict(X_test_ori)
        print("classifier:", name, "f1:", f1_score(y_test, pred), "acc:", accuracy_score(y_test, pred))

'''
Generating char TFIDF vectorizer and finding the TFIDF vectors for the sentences
char TFIDF n-gram range=1-4
'''
printDetails('char TFIDF n-gram range=1-4')
vec = TfidfVectorizer(lowercase=False,ngram_range=(1,4), analyzer='char')
X_train_ori = vec.fit_transform(X_train_sen)
X_test_ori = vec.transform(X_test_sen)
runClassifier(X_train_ori, y_train, X_test_ori, y_test)

'''
Generating word TFIDF vectorizer and finding the TFIDF vectors for the sentences
word TFIDF n-gram range=1-4
'''
printDetails('word TFIDF n-gram range=1-4')
vec = TfidfVectorizer(lowercase=False,ngram_range=(1,4), analyzer='word')
X_train_ori = vec.fit_transform(X_train_sen)
X_test_ori = vec.transform(X_test_sen)
runClassifier(X_train_ori, y_train, X_test_ori, y_test)

'''
Loading the multilingual BERT model
'''
model = SentenceTransformer('distiluse-base-multilingual-cased')

'''
Generating multilingual BERT embedding for train and test sentences
'''
X_train_ori = model.encode(X_train_sen, batch_size=32, show_progress_bar=True)
X_test_ori = model.encode(X_test_sen, batch_size=32, show_progress_bar=True)

printDetails('multilingual BERT')
runClassifier(X_train_ori, y_train, X_test_ori, y_test)


'''
Loading urdu Fasttext model
'''
fast = Magnitude("../downloads/fasttext.magnitude")

'''
Generating word embedding and averaging the word embedding for getting sentence embedding
'''
def fasttext(x):
    vectors = []
    for title in tqdm(x):
        vectors.append(np.average(fast.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)

X_train_fast = fasttext(X_train_sen)
X_test_fast = fasttext(X_test_sen)

printDetails('urdu FastText')
runClassifier(X_train_fast, y_train, X_test_fast, y_test)

'''
Building and training word2vec model
'''
model = Word2Vec([text.split(' ') for text in X_train_sen],size=512,min_count=50,iter=30,window=10)
X_train = []
for text in X_train_sen:
    vecs = []
    text = text.split(' ')
    for t in text:
        if t in model.wv.vocab:
            vecs.append(model.wv[t])
    vecs = np.average(np.array(vecs),axis=0)
    X_train.append(vecs)
X_test = []
for text in X_test_sen:
    vecs = []
    text = text.split(' ')
    for t in text:
        if t in model.wv.vocab:
            vecs.append(model.wv[t])
    vecs = np.average(np.array(vecs),axis=0)
    X_test.append(vecs)
X_train_ori = np.array(X_train)
X_test_ori = np.array(X_test)

printDetails('Word2Vec')
runClassifier(X_train_ori, y_train, X_test_ori, y_test)