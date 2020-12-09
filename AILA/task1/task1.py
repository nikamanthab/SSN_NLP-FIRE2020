'''
Runs of AILA 2020 document ranking task1
'''

import pandas as pd
import numpy as np
import os
from rank_bm25 import BM25Okapi
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('../../aila20-task1/relevance_judgements/task1a_rel_judgements.txt') as f:
    data = f.read()
with open('../../aila20-test/TestData_release/Task1_test_data.txt') as query_doc:
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
query_df.to_csv('../../tsv/submission_query.csv', index=False)

query = pd.read_csv('../../tsv/submission_query.csv')
cases_csv = pd.read_csv('../../tsv/cases.csv')
statues_csv = pd.read_csv('../../tsv/statues.csv')
rel = pd.read_csv('../../tsv/rel_task1a.csv')
rel2 = pd.read_csv('../../tsv/rel_task1b.csv')

cases = cases_csv
# cases = statues_csv

result = pd.DataFrame()

corpus = cases['text']
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

scores = []
query_ids = []
document_ids = []

count = 0
for text in list(query['query']):
    count+=1
    tokenized_query = text.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    query_ids = np.concatenate([query_ids, ['AILA_TQ'+str(count)]*len(doc_scores)])
    document_ids = np.concatenate([document_ids, cases['document_id']])
    scores = np.concatenate([scores, doc_scores])

result['query_id'] = query_ids
result['document_id'] = document_ids
result['score'] = scores

print(result)

'''
sorting by the scores from BM25 algo
And generating the result files
'''
r = max(result['score'])-min(result['score'])
with open('../../submission/SSNCSE_NLP_task1a_1_revised.txt','wt') as f:
    for i in result['query_id'].unique():
        print(i)
        ds = result[result['query_id'] == i]
        ds = ds.sort_values(by='score', ascending=False)
        for row_id in range(len(ds)):
            inst = ds.iloc[row_id]
            f.write(inst['query_id']+' ')
            f.write('Q0 ')
            f.write(inst['document_id']+' ')
            val = inst['score']/r
            if val>1.:
                val = 1.
            elif val < 0.:
                val = 0.
            f.write(str(val)+' ')
            f.write('SSNCSE_NLP_1')
            f.write('\n')
            
'''
Feature extraction for TFIDF
'''
query_text = query['query']
cases_text = cases['text']
unique_text = np.concatenate([query_text, cases_text])
vec = TfidfVectorizer()
vec.fit_transform(unique_text)

query_vec = vec.transform(query['query'])
cases_vec = vec.transform(cases['text'])

'''
Applying cosine similarity
'''
result = pd.DataFrame()

similarity_scores = []
query_ids = []
document_ids = []

for query_index, qq in enumerate(query_vec[:]):
    print(query_index, end=' ')
    query_id = query.iloc[query_index]['query_id']
    for cases_index, cc in enumerate(cases_vec):
        document_id = cases.iloc[cases_index]['document_id']
        
        query_ids.append(query_id)
        document_ids.append(document_id)
        similarity_scores.append(cosine_similarity(qq, cc)[0][0])

result['query_id'] = query_ids
result['document_id'] = document_ids
result['similarity_score'] = similarity_scores

print(result)

'''
Normalizing the scores and saving the sorted results
'''
r = max(result['similarity_score'])-min(result['similarity_score'])
with open('../../submission/SSNCSE_NLP_task1a_2_revised.txt','wt') as f:
    for i in result['query_id'].unique():
        print(i)
        ds = result[result['query_id'] == i]
        ds = ds.sort_values(by='similarity_score', ascending=False)
        for row_id in range(len(ds)):
            inst = ds.iloc[row_id]
            f.write(inst['query_id']+' ')
            f.write('Q0 ')
            f.write(inst['document_id']+' ')
            val = inst['similarity_score']/r
            if val>1:
                val = 1.
            elif val<0:
                val = 0.
            f.write(str(val)+' ')
            f.write('SSNCSE_NLP_2')
            f.write('\n')
            



