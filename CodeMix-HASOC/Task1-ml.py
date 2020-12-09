'''
Runs for CodeMix HASOC task1 malayalam
authors: Nitin Nikamanth Appiah Balaji, Bharathi B
'''
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random

'''
Data Set Loading
Train, development and test datasets
'''
train = pd.read_csv('../data/ml-Hasoc-offensive-train.csv', '\t', names=['label', 'text'])
dev = pd.read_csv('../data/ml-Hasoc-offensive-dev.csv', '\t', names=['label', 'text'])
test = pd.read_csv('../HASOC-Dravidian-CodeMix/Task1/ml_mixedscript_Hascoc_offensive_test_without_label.csv', names=['id', 'text'])

print(train['label'].value_counts())

'''
Balancing the dataset classes
Random duplication of lower count classes till both classes have equal number of samples
'''
offensive = train[train['label'] == 'Offensive']
while(list(train['label'].value_counts())[0]>list(train['label'].value_counts())[1]):
    df = pd.DataFrame()
    r = random.randint(0, len(offensive)-1)
    df['text'] = [offensive.iloc[r]['text']]
    df['label'] = [offensive.iloc[r]['label']]
    train = pd.concat([train, df])
    
print(train['label'].value_counts())

'''
Generating char TFIDF vectorization and converting sentences to vectors
char TFIDF ngram range=1-3
'''
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(train['text'])
X_dev = vectorizer.transform(dev['text'])

y_train = train['label']
y_dev = dev['label']

clf = RandomForestClassifier(n_estimators=1000,verbose=True, n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("TFIDF:")
print(classification_report(y_dev, pred))

'''
Generating char count vectorization and converting sentences to vectors
char count ngram range=1-3
'''
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(train['text'])
X_dev = vectorizer.transform(dev['text'])

y_train = train['label']
y_dev = dev['label']

clf = RandomForestClassifier(n_estimators=1000,verbose=True, n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vec:")
print(classification_report(y_dev, pred))

'''
multilingual BERT model loading and embedding generation
'''
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased')
X_train = model.encode(list(train['text']), batch_size=32,show_progress_bar=True)
X_dev = model.encode(list(dev['text']), batch_size=32,show_progress_bar=True)
X_test = model.encode(list(test['text']), batch_size=32, show_progress_bar=True)

y_train = train['label']
y_dev = dev['label']

clf = MLPClassifier(verbose=True, hidden_layer_sizes=(512))
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("BERT:")
print(classification_report(y_dev, pred))
