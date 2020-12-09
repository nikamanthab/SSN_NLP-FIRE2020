'''
LSTM Run for Code Mix Sentiment Analysis task2
authors: Nitin Nikamanth Appiah Balaji, Bharathi B
'''

import mkl
mkl.set_num_threads(36)
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM, SpatialDropout1D, Bidirectional
# from keras import callbacks
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from pymagnitude import *
from nltk import word_tokenize
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

train = pd.read_csv('../codemix-corpus-fire2020/malayalam_train.tsv','\t')
dev = pd.read_csv('../codemix-corpus-fire2020/malayalam_dev.tsv','\t')
test = pd.read_csv('../Dravidian-CodeMix/malayalam_test.csv')

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 150
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = tokenizer.texts_to_sequences(train['text'].values)
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train.shape)

X_dev = tokenizer.texts_to_sequences(dev['text'].values)
X_dev = pad_sequences(X_dev, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_dev.shape)

X_test = tokenizer.texts_to_sequences(test['text'].values)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)


y_train = []
for i in train['category']:
    if i == 'Positive ':
        y_train.append([1,0,0,0,0])
    elif i == 'Negative ':
        y_train.append([0,1,0,0,0])
    elif i == 'Mixed_feelings ':
        y_train.append([0,0,1,0,0])
    elif i == 'unknown_state ':
        y_train.append([0,0,0,1,0])
    elif i == 'not-malayalam ':
        y_train.append([0,0,0,0,1])
# y_train = pd.get_dummies(train['category'][:]).values
y_train = np.array(y_train)
print('Shape of label tensor:', y_train.shape)

y_dev = []
for i in dev['category']:
    if i == 'Positive ':
        y_dev.append([1,0,0,0,0])
    elif i == 'Negative ':
        y_dev.append([0,1,0,0,0])
    elif i == 'Mixed_feelings ':
        y_dev.append([0,0,1,0,0])
    elif i == 'unknown_state ':
        y_dev.append([0,0,0,1,0])
    elif i == 'not-malayalam ':
        y_dev.append([0,0,0,0,1])
y_dev = np.array(y_dev)
# y_dev = pd.get_dummies(dev['category'][:]).values
print('Shape of label tensor:', y_dev.shape)


from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_w(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(1024, dropout=0.5, recurrent_dropout=0.2)))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])

epochs = 5
batch_size = 128

history = model.fit(X_train, y_train, epochs=epochs, 
                    batch_size=batch_size,validation_data=(X_dev,y_dev))#,
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
preds= model.predict_classes(X_dev)
pred_classes = []

for i in preds:
    if i == 0:
        pred_classes.append('Positive ')
    elif i == 1:
        pred_classes.append('Negative ')
    elif i == 2:
        pred_classes.append('Mixed_feelings ')
    elif i == 3:
        pred_classes.append('unknown_state ')
    elif i == 4:
        pred_classes.append('not-malayalam ')
# print(f1_w(np.argmax(y_dev,axis=1), preds))
# print(f1_w(dev['category'], preds))
print(f1_score(dev['category'], pred_classes, average='weighted'))

preds= model.predict_classes(X_test)
pred_classes = []

for i in preds:
    if i == 0:
        pred_classes.append('Positive')
    elif i == 1:
        pred_classes.append('Negative')
    elif i == 2:
        pred_classes.append('Mixed_feelings')
    elif i == 3:
        pred_classes.append('unknown_state')
    elif i == 4:
        pred_classes.append('not-malayalam')
        
len(pred_classes)
test['label'] = pred_classes

test.to_csv('../SSNCSE-NLP/Task2/lstm2.tsv', index=False, sep='\t')
# print(test)
