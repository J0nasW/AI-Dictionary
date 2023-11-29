import pandas as pd
import pickle, os
pd.options.mode.chained_assignment = None  # default='warn'
from sentence_transformers import SentenceTransformer
import time

import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm
# register tqdm with pandas
tqdm.pandas()

from helper.keyword_helper import make_aho_automation, neo4j_fetch_data

NUM_CORES = 20
CHUNK_SIZE_CORE = 1000
CHUNK_SIZE_EXTENDED = 100000

KEYWORD_FREQ_RANGE = (4,5000)
COS_THRESHOLD = 0.95

DICT_PATH = "data/dictionaries"

# Create a dict of neo4j credentials
NEO4J_CREDENTIALS = {"url": "bolt://localhost:37687", "user": "neo4j", "password": "neo4jpassword"}

print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print("")

# Get all keywords related to papers - takes around 2 minutes
query = """
MATCH (p:Paper)-[r]->(k:Keyword)
WITH k.keyword AS keyword, p.title AS paper_title, p.id AS paper_id
RETURN keyword, paper_title, paper_id
"""
print("Fetching data...")
paper_keywords = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
print("Done.")
print(f"Number of keywords: {len(paper_keywords)}")

dedupe_keywords_f = (paper_keywords
                     .value_counts(subset=['keyword'])
                     .reset_index(name='frequency')
                     .sort_values(by='frequency', ascending=False)
                     .assign(frequency_normalized=lambda df: df['frequency'] / df['frequency'].max()))

print(f"Got {len(dedupe_keywords_f)} keywords after deduplication.")

print("Loading Sentence Transformer model...")
model_scincl = SentenceTransformer('malteos/SciNCL')
print("Done.")
print("Embedding all keywords...")
dedupe_keywords_f['embedding'] = model_scincl.encode(dedupe_keywords_f['keyword'].tolist(), show_progress_bar=True).tolist()

# Load the core keywords with their embeddings into a df
core_keywords = pd.read_csv('data/dictionaries/core_keywords.csv')
core_keywords['embedding'] = core_keywords['embedding'].apply(lambda x: eval(x))

# Calculating similarity - this step will take a while (around 1h)

core_keywords_cso = core_keywords[core_keywords['source'] == 'cso']
core_keywords_method = core_keywords[core_keywords['source'] == 'method']
core_keywords_task = core_keywords[core_keywords['source'] == 'task']
core_keywords_dataset = core_keywords[core_keywords['source'] == 'dataset']

# Function to divide DataFrame into chunks
def chunk_dataframe(df, chunk_size):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def get_keywords_above_threshold(core_embedding, extended_embeddings, cos_threshold=0.1):
    similarities = cosine_similarity([core_embedding], extended_embeddings)[0]
    return np.where(similarities > cos_threshold)[0]

def batch_process_embeddings(core_embeddings, extended_embeddings_chunk, cos_threshold):
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(lambda embedding: get_keywords_above_threshold(embedding, extended_embeddings_chunk, cos_threshold), core_embeddings))
    return results

def process_keywords_chunk(core_chunk, dedupe_chunks, cos_threshold, source):
    extended_keywords_all = []
    core_embeddings = core_chunk['embedding'].tolist()
    
    for dedupe_chunk in dedupe_chunks:
        dedupe_embeddings = np.array(dedupe_chunk['embedding'].tolist())
        core_chunk['keywords_above_threshold'] = batch_process_embeddings(core_embeddings, dedupe_embeddings, cos_threshold)
        
        core_chunk['keywords_above_threshold'] = core_chunk['keywords_above_threshold'].apply(lambda indices: dedupe_chunk.iloc[indices]['keyword'].tolist())
        core_chunk = core_chunk[['keyword', 'source', 'keywords_above_threshold']]
        core_chunk['keywords_above_threshold'] = core_chunk['keywords_above_threshold'].apply(lambda keywords: list(set(keywords) - set(core_chunk['keyword'].tolist())))
        
        extended_keywords_chunk = pd.DataFrame({keyword for keywords in core_chunk['keywords_above_threshold'] for keyword in keywords}, columns=['keyword'])
        extended_keywords_chunk['source'] = source
        
        extended_keywords_all.append(extended_keywords_chunk)

    if extended_keywords_all:
        return pd.concat(extended_keywords_all)
    else:
        return pd.DataFrame()

# Main processing loop
def process_all_keywords(core_keywords, dedupe_keywords, cos_threshold, source):
    core_chunks = chunk_dataframe(core_keywords, CHUNK_SIZE_CORE)
    dedupe_chunks = chunk_dataframe(dedupe_keywords, CHUNK_SIZE_EXTENDED)
    extended_keywords_all = []

    for core_chunk in core_chunks:
        extended_keywords_chunk = process_keywords_chunk(core_chunk, dedupe_chunks, cos_threshold, source)
        if not extended_keywords_chunk.empty:
            extended_keywords_all.append(extended_keywords_chunk)

    if extended_keywords_all:
        return pd.concat(extended_keywords_all)
    else:
        return pd.DataFrame()

print("Finding similar keywords...")
print("")

print("Processing core keywords...")
extended_keywords_cso = process_all_keywords(core_keywords_cso, dedupe_keywords_f, COS_THRESHOLD, "cso")
print(f"Got {len(extended_keywords_cso)} unique extended cso keywords after processing core keywords ({len(extended_keywords_cso) / len(dedupe_keywords_f) * 100:.2f}%)")

print("Processing method keywords...")
extended_keywords_method = process_all_keywords(core_keywords_method, dedupe_keywords_f, COS_THRESHOLD, "method")
print(f"Got {len(extended_keywords_method)} unique extended method keywords after processing core keywords ({len(extended_keywords_method) / len(dedupe_keywords_f) * 100:.2f}%)")

print("Processing task keywords...")
extended_keywords_task = process_all_keywords(core_keywords_task, dedupe_keywords_f, COS_THRESHOLD, "task")
print(f"Got {len(extended_keywords_task)} unique extended task keywords after processing core keywords ({len(extended_keywords_task) / len(dedupe_keywords_f) * 100:.2f}%)")

print("Processing dataset keywords...")
extended_keywords_dataset = process_all_keywords(core_keywords_dataset, dedupe_keywords_f, COS_THRESHOLD, "dataset")
print(f"Got {len(extended_keywords_dataset)} unique extended dataset keywords after processing core keywords ({len(extended_keywords_dataset) / len(dedupe_keywords_f) * 100:.2f}%)")

print("")
print("Done.")

extended_keywords = pd.concat([extended_keywords_cso, extended_keywords_method, extended_keywords_task, extended_keywords_dataset])
extended_keywords = extended_keywords.reset_index(drop=True)

# Make a new folder in the DICT_PATH for the ahocorasick dumps
if not os.path.exists(DICT_PATH + "/extended_aho_automation"):
    os.mkdir(DICT_PATH + "/extended_aho_automation")
    
extended_keywords_cso_automation = make_aho_automation(extended_keywords_cso['keyword'].tolist())
extended_keywords_cso_automation.save(f"{DICT_PATH}/extended_aho_automation/cso_aho_automation.pkl", pickle.dumps)

extended_keywords_method_automation = make_aho_automation(extended_keywords_method['keyword'].tolist())
extended_keywords_method_automation.save(f"{DICT_PATH}/extended_aho_automation/method_aho_automation.pkl", pickle.dumps)

extended_keywords_task_automation = make_aho_automation(extended_keywords_task['keyword'].tolist())
extended_keywords_task_automation.save(f"{DICT_PATH}/extended_aho_automation/task_aho_automation.pkl", pickle.dumps)

extended_keywords_dataset_automation = make_aho_automation(extended_keywords_dataset['keyword'].tolist())
extended_keywords_dataset_automation.save(f"{DICT_PATH}/extended_aho_automation/dataset_aho_automation.pkl", pickle.dumps)

# Calculate embeddings for all extended keywords
print("Embedding all extended keywords...")
extended_keywords['embedding'] = model_scincl.encode(extended_keywords['keyword'].tolist(), show_progress_bar=True).tolist()
print("Done.")

# Save the extended keywords to a csv
extended_keywords.to_csv('data/dictionaries/extended_keywords.csv', index=False)
print("Done saving.")

print("")
print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")