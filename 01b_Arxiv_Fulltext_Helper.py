from pdfminer.high_level import extract_text
import os
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import cleantext
import hashlib
from collections import Counter

from neo4j import GraphDatabase

import multiprocessing

BATCH_SIZE = 1000
NUM_CORES = 20

MIN_WORD_LENGTH = 30
MIN_TEXT_LENGTH = 100

PATH_TO_ARXIV_PDFS = "/mnt/hdd02/PubCrawl/tmp/arxiv_pdf/" # Path to the folder with the arxiv pdfs
PATH_TO_NEO4J_IMPORT = "data/neo4j/"

# Define your neo4j connection parameters here:
NEO4J_URL = "bolt://localhost:37687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jpassword"
driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_data(query):
  with driver.session() as session:
    result = session.run(query)
    return pd.DataFrame([r.values() for r in result], columns=result.keys())

# Get all openalex_ids from papers for a negative list
query = """
MATCH (p:Paper)
RETURN p.arxiv_id as arxiv_id, p.id as paper_id
"""
print("Fetching data...")
paper_arxiv_ids = fetch_data(query)
print("Done.")
print(f"Got {len(paper_arxiv_ids)} PwC Paper arXiv ID pairs.")

# Make a list with all arxiv pdf names in the tmp/arxiv_pdf folder
pdf_list = []
for file in os.listdir(PATH_TO_ARXIV_PDFS):
    if file.endswith('.pdf'):
        pdf_list.append(file)
    else:
        continue
print(f"Got {len(pdf_list)} pdfs in list.")
    
pdf_df = pd.DataFrame(pdf_list, columns=['pdf_name'])
pdf_df['arxiv_id_file'] = pdf_df['pdf_name'].str.replace('.pdf', '', regex=True)
pdf_df['arxiv_id_file'] = pdf_df['arxiv_id_file'].str.replace('v.*', '', regex=True)

# Match the paper_arxiv_ids with the arxiv_id_file to get the paper_id
pdf_df = pdf_df.merge(paper_arxiv_ids, how='left', left_on='arxiv_id_file', right_on='arxiv_id')
# Drop the arxiv_id_file column
pdf_df.drop(columns=['arxiv_id_file'], inplace=True)
# Drop the rows where paper_id is null
pdf_df.dropna(subset=['paper_id'], inplace=True)
# Remove duplicates
pdf_df.drop_duplicates(subset=['paper_id'], inplace=True)
# Reset index
pdf_df.reset_index(drop=True, inplace=True)
print(f"Got {len(pdf_df)} pdfs with paper_id.")

def process_pdf(pdf_name):
    try:
        text = extract_text(PATH_TO_ARXIV_PDFS + pdf_name)
        text = cleantext.clean(text, clean_all=True, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True, stp_lang='english')
        words = text.split()
        processed_words = [process_word(word) for word in words if word]
        text = ' '.join(processed_words).strip()
        # Remove unnecessary whitespaces
        text = re.sub(r'[^a-z\s]+', ' ', text).strip()
        # Find the index of 'abstract' or 'introduction'
        index = max(text.find('abstract'), text.find('introduction'))
        # If 'abstract' or 'introduction' was found, remove everything before it and itself
        if index > 0:
            # Remove everything before 'abstract' or 'introduction' and also remove 'abstract' or 'introduction'
            text = text[index + len('introduction'):]
        # Check if the text is long enough and if all words are shorter than 30 characters
        if check_word_conditions(text.split()):
            return text
        else:
            # If the text is not long enough or if there are words longer than 30 characters, return None
            return None
    except Exception as e:
        print(f"Error processing file {pdf_name}: {e}")
        return None

def check_word_conditions(words):
    return len(words) > MIN_TEXT_LENGTH and all(len(word) <= MIN_WORD_LENGTH for word in words)

def process_word(word):
    if len(word) <= 2 or len(word) >= 30:
        return ''
    return word
    
# Split dataframe into chunks
num_batches = int(np.ceil(len(pdf_df) / BATCH_SIZE))

fulltexts_df = pd.DataFrame(columns=['paper_id', 'pdf_text'])

i = 1

for batch in tqdm(np.array_split(pdf_df, num_batches)):
    with multiprocessing.Pool(NUM_CORES) as pool:
        batch_results = list(tqdm(pool.imap(process_pdf, batch['pdf_name']), total=len(batch['pdf_name'])))
    batch['pdf_text'] = batch_results
    # Remove rows that have None as pdf_text
    batch.dropna(subset=['pdf_text'], inplace=True)    
    # fulltexts_df = pd.concat([fulltexts_df, batch[['paper_id', 'pdf_text']]], ignore_index=True)
    # Save batch as csv
    batch[['paper_id', 'pdf_text']].to_csv(f'data/fulltexts_temp/fulltexts_{i}_{num_batches}.csv', index=False, header=True)
    # Empty cache
    del batch
    i += 1
    # if i > 2:
    #     break
    # break
    
# # Merge pdf_df with fulltexts_df
# Load all csv files in data/fulltexts_temp/ into one dataframe
fulltext_csv_df = pd.concat([pd.read_csv(f'data/fulltexts_temp/{file}') for file in os.listdir('data/fulltexts_temp/')], ignore_index=True)
# Merge pdf_df with fulltext_csv_df
pdf_df = pdf_df.merge(fulltext_csv_df, how='left', on='paper_id')

# Make neo4j import CSVs

# Make fulltext ids with hashlib like this: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16))
pdf_df["id"] = pdf_df["arxiv_id"].apply(lambda x: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)))

# Nodes
fulltext_node_csv = pdf_df[['paper_id', 'pdf_text']].copy()
fulltext_node_csv[':LABEL'] = 'Fulltext'
fulltext_node_csv['paper_id'] = fulltext_node_csv['paper_id'].astype(str)
fulltext_node_csv.rename(columns={'paper_id': ':id:ID', 'pdf_text': 'text'}, inplace=True)
fulltext_node_csv.to_csv(f'{PATH_TO_NEO4J_IMPORT}fulltexts.csv', index=False, header=True)

# Relationships
fulltext_rel_csv = pdf_df[['paper_id', 'id']].copy()
fulltext_rel_csv[':TYPE'] = 'HAS_FULLTEXT'
fulltext_rel_csv['paper_id'] = fulltext_rel_csv['paper_id'].astype(str)
fulltext_rel_csv['id'] = fulltext_rel_csv['id'].astype(str)
fulltext_rel_csv.rename(columns={'paper_id': ':START_ID', 'id': ':END_ID'}, inplace=True)
fulltext_rel_csv.to_csv(f'{PATH_TO_NEO4J_IMPORT}papers_fulltexts.csv', index=False, header=True)