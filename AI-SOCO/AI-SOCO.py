'''
Runs for Authorship Identification of SOurce COde (AI-SOCO) 2020 task
authors: Nitin Nikamanth Appiah Balaji, Bharathi B
'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

'''
For limiting the CPU cores utilization
'''
import mkl
mkl.set_num_threads(36)

'''
Loading data - train and dev
shuffling the dataframe rows
'''
train = pd.read_csv('../AI-SOCO-master/data_dir/train.csv').sample(frac=1)
dev = pd.read_csv('../AI-SOCO-master/data_dir/dev.csv')

X_train_id, y_train = train['pid'], train['uid']
X_dev_id, y_dev = dev['pid'], dev['uid']


'''
Reading corpus files and forming array of sentences
'''
X_train_sen = []
X_dev_sen = []
X_test_sen = []
for pid in tqdm(X_train_id):
    with open('../AI-SOCO-master/data_dir/train/'+str(pid)) as file:
        X_train_sen.append(file.read())
for pid in tqdm(X_dev_id):
    with open('../AI-SOCO-master/data_dir/dev/'+str(pid)) as file:
        X_dev_sen.append(file.read())

        
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
    clfs.append(('Naive Bayes', BernoulliNB()))
    clfs.append(('Randomforest',RandomForestClassifier(n_jobs=36,verbose=True,n_estimators=400))
    
    for name, clf in clfs:
        clf.fit(X_train_ori, y_train)
        pred = clf.predict(X_test_ori)
        print("classifier:", name, "f1:", f1_score(y_dev, pred, average='weighted'), "acc:", accuracy_score(y_test, pred))      

    
'''
Fitting char TFIDF vectorizer and converting sentences to vectors
char TFIDF n-gram range=2-5
'''
vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,5))
X_train = vectorizer.fit_transform(X_train_sen)
X_dev = vectorizer.transform(X_dev_sen)
                
printDetails('char TFIDF n-gram range=2-5')
runClassifier(X_train, y_train, X_dev, y_dev)
                
'''
Fitting char count vectorizer and converting sentences to vectors
char count vec n-gram range=2-5
'''
vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,5))
X_train = vectorizer.fit_transform(X_train_sen)
X_dev = vectorizer.transform(X_dev_sen)
                
printDetails('char TFIDF n-gram range=2-5')
runClassifier(X_train, y_train, X_dev, y_dev)

