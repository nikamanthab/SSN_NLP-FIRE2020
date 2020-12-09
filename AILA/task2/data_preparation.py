'''
Data Preparation for AILA 2020 task2
authors: Nitin Nikamanth Appiah Balaji, Bharathi B, Bhuvana J
'''
import pandas as pd
import numpy as np
import os

for file in os.listdir('../../aila20-task2/task-2/'):
    with open('../../aila20-task2/task-2/'+file) as f:
        data = f.read()
        sen = []
        labels = []
        print(file)
        for line in data.split('\n'):
            sp = line.split('\t')
            if len(sp) == 2:
                sen.append(sp[0])
                labels.append(sp[1])
        df = pd.DataFrame()
        df['text'] = sen
        df['labels'] = labels
        df.to_csv('../../tsv/task2-train/'+file.split('.')[0]+'.tsv','\t',index=False)

train = pd.concat([
    pd.read_csv('../../tsv/task2-train/'+i,'\t') for i in os.listdir('../../tsv/task2-train/')
])
print("Value Counts:")
print(train['labels'].value_counts())

train.to_csv('../../tsv/task2-train/train.tsv',sep='\t',index=False)
