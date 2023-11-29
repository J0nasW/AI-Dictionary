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
CHUNK_SIZE_EXTENDED = 100000
COS_THRESHOLD = 0.95

DICT_PATH = "data/dictionaries"

# Create a dict of neo4j credentials
NEO4J_CREDENTIALS = {"url": "bolt://localhost:37687", "user": "neo4j", "password": "neo4jpassword"}

print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print("")

print("Loading Sentence Transformer model...")
MODEL_SCINCL = SentenceTransformer('malteos/SciNCL')
print("Done.")
print("")

def get_keywords():
    query = f"""
    MATCH (p:Paper)-[r]->(k:Keyword)
    WITH k.keyword AS keyword, p.title AS paper_title, p.id AS paper_id
    RETURN keyword, paper_title, paper_id
    // LIMIT 10000
    """ 
    
    print("Fetching data from neo4j...")
    paper_keywords = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    if len(paper_keywords) == 0:
        return pd.DataFrame()
    print(f"Number of keywords: {len(paper_keywords)}")
    
    dedupe_keywords_f = (paper_keywords
                     .value_counts(subset=['keyword'])
                     .reset_index(name='frequency')
                     .sort_values(by='frequency', ascending=False)
                     .assign(frequency_normalized=lambda df: df['frequency'] / df['frequency'].max()))

    print(f"Got {len(dedupe_keywords_f)} keywords after deduplication.")

    
    print("Embedding all keywords...")
    dedupe_keywords_f['embedding'] = MODEL_SCINCL.encode(dedupe_keywords_f['keyword'].tolist(), show_progress_bar=True).tolist()
    print("Done.")

    return dedupe_keywords_f

def get_keywords_above_threshold(core_embedding, extended_embeddings, cos_threshold=0.1):
    similarities = cosine_similarity([core_embedding], extended_embeddings)[0]
    return np.where(similarities > cos_threshold)[0]

def batch_process_embeddings(core_embeddings, extended_embeddings_chunk, cos_threshold):
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(lambda embedding: get_keywords_above_threshold(embedding, extended_embeddings_chunk, cos_threshold), core_embeddings))
    return results

# Function to divide DataFrame into chunks
def chunk_dataframe(df, chunk_size):
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def process_all_keywords(core_keywords_df, extended_keywords_df, cos_threshold, source):
    
    # Chunk the extended keywords
    extended_keywords_chunks = chunk_dataframe(extended_keywords_df, CHUNK_SIZE_EXTENDED)
    
    # Iterate over the chunks and process them
    extended_keywords_all = []
    for extended_keywords_chunk in tqdm(extended_keywords_chunks):
        extended_embeddings = np.array(extended_keywords_chunk['embedding'].tolist())
        core_keywords_df['keywords_above_threshold'] = batch_process_embeddings(core_keywords_df['embedding'].tolist(), extended_embeddings, cos_threshold)

        # Accumulate data in a list
        similar_keywords_data = []
        for i, row in core_keywords_df.iterrows():
            keywords_above_threshold = extended_keywords_chunk.iloc[row['keywords_above_threshold']]['keyword'].tolist()
            for keyword in keywords_above_threshold:
                keyword = keyword.lower()
                if keyword != row['keyword']:
                    similar_keywords_data.append({'keyword': keyword, 'source': source, 'core_keyword': row['keyword']})

        # Create DataFrame from accumulated data
        similar_keywords_df = pd.DataFrame(similar_keywords_data)

        # Deduplicate and append to extended_keywords_all
        similar_keywords_df = similar_keywords_df.drop_duplicates()
        extended_keywords_all.append(similar_keywords_df)
        
    # Concatenate all DataFrames outside the loop
    return pd.concat(extended_keywords_all).reset_index(drop=True)
    
# Load the core keywords with their embeddings into a df
print("Loading core keywords...")
# core_keywords = pd.read_csv('data/dictionaries/core_keywords.csv').sample(10000)
core_keywords = pd.read_csv('data/dictionaries/core_keywords.csv')
core_keywords['embedding'] = core_keywords['embedding'].apply(lambda x: eval(x))
core_keywords_cso = core_keywords[core_keywords['source'] == 'cso']
core_keywords_method = core_keywords[core_keywords['source'] == 'method']
core_keywords_task = core_keywords[core_keywords['source'] == 'task']
core_keywords_dataset = core_keywords[core_keywords['source'] == 'dataset']
print("Done.")
print("")

# Loading the extended keywords
extended_keywords_df = get_keywords()

print("Finding similar keywords...")
print("")

print("Processing core keywords...")
extended_keywords_cso = process_all_keywords(core_keywords_cso, extended_keywords_df, COS_THRESHOLD, "cso")
print(f"Got {len(extended_keywords_cso)} unique extended cso keywords after processing core keywords.")

print("Processing method keywords...")
extended_keywords_method = process_all_keywords(core_keywords_method, extended_keywords_df, COS_THRESHOLD, "method")
print(f"Got {len(extended_keywords_method)} unique extended method keywords after processing core keywords.")

print("Processing task keywords...")
extended_keywords_task = process_all_keywords(core_keywords_task, extended_keywords_df, COS_THRESHOLD, "task")
print(f"Got {len(extended_keywords_task)} unique extended task keywords after processing core keywords.")

print("Processing dataset keywords...")
extended_keywords_dataset = process_all_keywords(core_keywords_dataset, extended_keywords_df, COS_THRESHOLD, "dataset")
print(f"Got {len(extended_keywords_dataset)} unique extended dataset keywords after processing core keywords.")

print("")
print("Done.")

# Subtract duplicates
print("")
print("Subtracting duplicates and combine dataframes...")
extended_keywords_cso = extended_keywords_cso.drop_duplicates()
extended_keywords_method = extended_keywords_method.drop_duplicates()
extended_keywords_task = extended_keywords_task.drop_duplicates()
extended_keywords_dataset = extended_keywords_dataset.drop_duplicates()

extended_keywords_method = extended_keywords_method[~extended_keywords_method['keyword'].isin(extended_keywords_cso['keyword'])]
extended_keywords_task = extended_keywords_task[~extended_keywords_task['keyword'].isin(extended_keywords_cso['keyword'])]
extended_keywords_task = extended_keywords_task[~extended_keywords_task['keyword'].isin(extended_keywords_method['keyword'])]
extended_keywords_dataset = extended_keywords_dataset[~extended_keywords_dataset['keyword'].isin(extended_keywords_cso['keyword'])]
extended_keywords_dataset = extended_keywords_dataset[~extended_keywords_dataset['keyword'].isin(extended_keywords_method['keyword'])]
extended_keywords_dataset = extended_keywords_dataset[~extended_keywords_dataset['keyword'].isin(extended_keywords_task['keyword'])]
print(f"Got {len(extended_keywords_cso)} unique extended cso keywords after subtracting duplicates.")
print(f"Got {len(extended_keywords_method)} unique extended method keywords after subtracting duplicates.")
print(f"Got {len(extended_keywords_task)} unique extended task keywords after subtracting duplicates.")
print(f"Got {len(extended_keywords_dataset)} unique extended dataset keywords after subtracting duplicates.")
extended_keywords = pd.concat([extended_keywords_cso, extended_keywords_method, extended_keywords_task, extended_keywords_dataset])
extended_keywords = extended_keywords.reset_index(drop=True)
print(f"Got {len(extended_keywords)} unique extended keywords.")
print("Done.")
print("")

print("Saving extended keywords...")
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

# Save the extended keywords to a csv
extended_keywords.to_csv('data/dictionaries/extended_keywords.csv', index=False)
print("Done saving.")

print("")
print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")