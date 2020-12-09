'''
Runs for CodeMIx Sentiment Analysis task2
authors: Nitin Nikamanth Appiah Balaji, Bharahti B
'''
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

'''
Loading data-sets
'''
train = pd.read_csv('../codemix-corpus-fire2020/malayalam_train.tsv','\t')
dev = pd.read_csv('../codemix-corpus-fire2020/malayalam_dev.tsv','\t')
test = pd.read_csv('../Dravidian-CodeMix/malayalam_test.csv')

X_train_ori, y_train = train['text'], train['category']
X_dev_ori, y_dev = dev['text'], dev['category']
X_test_ori, y_test = test['text'], dev['category']

'''
Generating char count vectorization and converting sentences to vectors
char count ngram range=1-5
'''
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,5), max_features=50000)
X_train = vectorizer.fit_transform(X_train_ori)
X_dev = vectorizer.transform(X_dev_ori)
X_test = vectorizer.transform(X_test_ori)

clf = BernoulliNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vec + NB:")
# print("f1 score:",f1_score(y_dev, pred, average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))

clf = MLPClassifier(hidden_layer_sizes=(512),max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("Count vec + LR")
# print("f1 score:",f1_score(y_dev, pred, average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))

'''
Generating char TFIDF vectorization and converting sentences to vectors
char TFIDF ngram range=1-5
'''
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,5), max_features=50000)
X_train = vectorizer.fit_transform(X_train_ori)
X_dev = vectorizer.transform(X_dev_ori)
X_test = vectorizer.transform(X_test_ori)

clf = BernoulliNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("TFIDF + NB:")
# print("f1 score:",f1_score(y_dev, pred, average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))

clf = MLPClassifier(hidden_layer_sizes=(512), max_iter=300)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("TFIDF + LR:")
# print("f1 score:",f1_score(y_dev, pred, average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))

'''
multilingual BERT model loading and embedding generation
'''
test = pd.read_csv('../Dravidian-CodeMix/malayalam_test.csv')
model = SentenceTransformer('distiluse-base-multilingual-cased',device='cuda:1')
X_train = model.encode(X_train_ori, batch_size=20,show_progress_bar=True)
X_dev = model.encode(X_dev_ori, batch_size=20, show_progress_bar=True)
X_test = model.encode(X_test_ori, batch_size=20, show_progress_bar=True)

clf = MLPClassifier(hidden_layer_sizes=(512,),max_iter=25)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("BERT + MLP:")
# print("f1 score:",f1_score(y_dev, pred, average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))

'''
Loading Malayalam specific pretrained fastText model
'''
from pymagnitude import *
from nltk import word_tokenize
fast = Magnitude("../downloads/malayalam.magnitude")
def fasttext(x):
    vectors = []
    for title in tqdm(x):
        vectors.append(np.average(fast.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)
X_train = fasttext(train['text'])
X_dev = fasttext(dev['text'])

clf = MLPClassifier(hidden_layer_sizes=(1024,),max_iter=25)
clf.fit(X_train, y_train)
pred = clf.predict(X_dev)
print("FASTTEXT + MLP:")
# print("f1 score:",f1_score(y_dev, pred,average='weighted'))
# print("acc:",accuracy_score(y_dev, pred))
print(classification_report(y_dev, pred))
