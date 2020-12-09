'''
Preprocessing for AILA 2020 task1
authors: Nitin Nikamanth Appiah Balaji, Bharathi B, Bhuvana J
'''
import pandas as pd
import numpy as np
import os

'''
Processing the corpus txt files
Converting into tsv files
'''
with open('../../aila20-task1/relevance_judgements/task1a_rel_judgements.txt') as f:
    data = f.read()
with open('../../aila20-task1/dataset/Query_doc.txt') as query_doc:
    querys = query_doc.read().split('\n')
    
query_df = pd.DataFrame()
query_id = []
query = []
for j in querys[:50]:
    s = j.split('||')
    query_id.append(s[0])
    query.append(s[1])
query_df['query_id'] = query_id
query_df['query'] = query
query_df.to_csv('../../tsv/query.csv', index=False)

total_cases_df = pd.DataFrame()
text = []
document_id = []
for i in os.listdir('../../aila20-task1/dataset/Object_casedocs/'):
    with open('../../aila20-task1/dataset/Object_casedocs/'+i) as f:
        data = f.read()
        doc_id = i.split('.')[0]
        document_id.append(doc_id)
        text.append(data)
total_cases_df['document_id'] = document_id
total_cases_df['text'] = text
total_cases_df.to_csv('../../tsv/cases.csv', index=False)

total_statues_df = pd.DataFrame()
text = []
document_id = []
for i in os.listdir('../../aila20-task1/dataset/Object_statutes/'):
    with open('../../aila20-task1/dataset/Object_statutes/'+i) as f:
        data = f.read()
        doc_id = i.split('.')[0]
        document_id.append(doc_id)
        text.append(data)
total_statues_df['document_id'] = document_id
total_statues_df['text'] = text
total_statues_df.to_csv('../../tsv/statues.csv', index=False)

query_df = pd.DataFrame()
query_id = []
q = []
for no,i in enumerate(querys[:50]):
    query_id.append(i.split('||')[0])
    q.append(i.split('||')[1])
query_df['query_id'] = query_id
query_df['query'] = q

with open('../../aila20-task1/relevance_judgements/task1a_rel_judgements.txt') as f:
    data = f.read()
with open('../../aila20-task1/dataset/Query_doc.txt') as query_doc:
    querys = query_doc.read().split('\n')
    
df = pd.DataFrame()
query_id = []
document_id = []
relevence = []
cases = []
for rec in data.split('\n'):
    query_id.append(rec.split(' ')[0])
    document_id.append(rec.split(' ')[2])
    with open('../../aila20-task1/dataset/Object_casedocs/'+str(rec.split(' ')[2])+'.txt') as case_doc:
        case_text = case_doc.read()
        cases.append(case_text)
    relevence.append(rec.split(' ')[3])
df['query_id'] = query_id
df['document_id'] = document_id
df['relevence'] = relevence
df['case_text'] = cases

df = df.merge(query_df, on='query_id')
print("df relevance value count:")
print(df['relevence'].value_counts())
print("unique query ids:")
print(len(df['query_id'].unique()))
df.to_csv('../../tsv/task1a.csv', index=False)
df = df[['query_id', 'document_id', 'relevence']]
df.to_csv('../../tsv/rel_task1a.csv', index=False)

with open('../../aila20-task1/relevance_judgements/task1b_rel_judgements.txt') as f:
    data = f.read()
    
df = pd.DataFrame()
query_id = []
document_id = []
relevence = []
statues = []
for rec in data.split('\n'):
    query_id.append(rec.split(' ')[0])
    document_id.append(rec.split(' ')[2])
    with open('../../aila20-task1/dataset/Object_statutes/'+str(rec.split(' ')[2])+'.txt') as statue_doc:
        statue_text = statue_doc.read()
        statues.append(statue_text)
    relevence.append(rec.split(' ')[3])
df['query_id'] = query_id
df['document_id'] = document_id
df['relevence'] = relevence
df['statues_text'] = statues

df = df.merge(query_df, on='query_id')
print("df relevance value count:")
print(df['relevence'].value_counts())
print("unique query ids:")
len(df['query_id'].unique())

df.to_csv('../../tsv/task1b.csv', index=False)
df = df[['query_id', 'document_id', 'relevence']]
df.to_csv('../../tsv/rel_task1b.csv', index=False)

