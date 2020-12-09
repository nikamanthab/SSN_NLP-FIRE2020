import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
import random

train = pd.read_csv('../data/Malayalam_offensive_data_Training-YT.csv')
test = pd.read_csv('../HASOC-Dravidian-CodeMix/Task2/malayalam_hasoc_tanglish_test_without_labels.tsv','\t', names=['ID','Tweets'])

offensive = train[train['Labels'] == 'OFF']
while(list(train['Labels'].value_counts())[0]>list(train['Labels'].value_counts())[1]):
    df = pd.DataFrame()
    r = random.randint(0, len(offensive)-1)
    df['Tweets'] = [offensive.iloc[r]['Tweets']]
    df['Labels'] = [offensive.iloc[r]['Labels']]
    train = pd.concat([train, df])
    
print(train['Labels'].value_counts())

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,5))
X = vectorizer.fit_transform(train['Tweets'])
y = [1 if x=='NOT' else 0 for x in train['Labels']]
# 1 - NOT
# 0 - OFF

clf = RandomForestClassifier(n_estimators=1000,verbose=True, n_jobs=-1)
scores = cross_validate(clf, X, y, cv=4, n_jobs=-1, scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
print("TFIDF:")
print('precision:',np.average(scores['test_precision_macro']))
print('recall:',np.average(scores['test_recall_macro']))
print('f1:',np.average(scores['test_f1_macro']))
print('acc:',np.average(scores['test_accuracy']))

#Count vectorization
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,5))
X = vectorizer.fit_transform(train['Tweets'])
y = [1 if x=='NOT' else 0 for x in train['Labels']]
# 1 - NOT
# 0 - OFF

clf = RandomForestClassifier(n_estimators=1000,verbose=True, n_jobs=-1)
scores = cross_validate(clf, X, y, cv=4, n_jobs=-1, scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
print("Count vectorization")
print('precision:',np.average(scores['test_precision_macro']))
print('recall:',np.average(scores['test_recall_macro']))
print('f1:',np.average(scores['test_f1_macro']))
print('acc:',np.average(scores['test_accuracy']))

#BERT
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased')
X_train = model.encode(list(train['Tweets']), batch_size=32,show_progress_bar=True)
X_test = model.encode(list(test['Tweets']), batch_size=32, show_progress_bar=True)

y_train = train['Labels']

clf = MLPClassifier(verbose=True, hidden_layer_sizes=(512))

scores = cross_validate(clf, X_train, y, cv=4, n_jobs=-1, scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy'])
print("BERT:")
print('precision:',np.average(scores['test_precision_macro']))
print('recall:',np.average(scores['test_recall_macro']))
print('f1:',np.average(scores['test_f1_macro']))
print('acc:',np.average(scores['test_accuracy']))