import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import logging
logging.basicConfig(level=logging.ERROR)
from sqlalchemy import create_engine, URL, text
from tqdm.auto import tqdm
# register tqdm with pandas
tqdm.pandas()
import ahocorasick
import pickle, re, os, time

from helper.keyword_helper import get_clean_keywords, neo4j_fetch_data, make_aho_automation

print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print("")

OALEX_SAMPLE_LIMIT = 100000
# OALEX_SAMPLE_FRAC = 0.01
AHO_LENGTH = 3 # Paper Abstract has to contain AHO_LENGTH or more core keywords to be labeled as ai and deleted from neg sample

url_object = URL.create(
    drivername='postgresql+psycopg2', 
    username='tie',
    password='TIE%2023!tuhh',
    host='134.28.58.100',
    # host='tie-workstation.tail6716.ts.net',
    # host='localhost',
    port=45432,
    database='openalex_db',
)
engine = create_engine(url_object)
engine.connect()

# Create a dict of neo4j credentials
NEO4J_CREDENTIALS = {"url": "bolt://localhost:37687", "user": "neo4j", "password": "neo4jpassword"}

# Get a Sample of around 0.01% of the data
# sql_query = f'''
#     SELECT *
#     FROM openalex.works
#     TABLESAMPLE SYSTEM ({OALEX_SAMPLE_FRAC})
#     WHERE abstract_inverted_index IS NOT NULL
# '''
sql_query = f'''
    SELECT *
    FROM openalex.works TABLESAMPLE BERNOULLI(10)
    WHERE abstract_inverted_index IS NOT NULL
    LIMIT {OALEX_SAMPLE_LIMIT}
'''

with engine.connect() as conn:
    print("Fetching data from Postgres...")
    openalex = pd.read_sql(sql=text(sql_query), con=conn)
    print(f"Got {len(openalex)} OpenAlex Works")
    
# Get all openalex_ids from papers for a negative list
query = """
MATCH (p:Paper)
WHERE p.id_openalex IS NOT NULL
RETURN p.id_openalex as openalex_id
"""
print("Fetching data from Neo4j...")
pwc_oalex_ids = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
print("Done.")
print(f"Got {len(pwc_oalex_ids)} OpenAlex IDs from PwC Papers.")

# Remove all openalex rows, where the id is in the pwc_oalex_ids df
print(f"Number of openalex sample abstracts before removing PwC papers: {len(openalex)}")
openalex_id_filtered = openalex[~openalex['id'].isin(pwc_oalex_ids['openalex_id'])]
print(f"Number of openalex sample abstracts after removing PwC papers: {len(openalex_id_filtered)} ({round(len(openalex_id_filtered)/len(openalex), 3) * 100}% remain)")

# Some cleaning
openalex_sample = openalex_id_filtered.copy()
openalex_sample['abstract_inverted_index'] = openalex_sample['abstract_inverted_index'].apply(lambda x: x['InvertedIndex'])
openalex_sample['abstract'] = [" ".join(list(d.keys())) if d else None for d in openalex_sample['abstract_inverted_index']]
openalex_sample.drop(columns=['abstract_inverted_index'], inplace=True)
openalex_sample['abstract'] = (openalex_sample['abstract']
                                .str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
                                .str.replace(r"abstract", "", flags=re.IGNORECASE)
                                .str.strip()
                                .str.replace(r"\s+", " ", regex=True)
                                .astype(str)
                                .str.lower()
                                )
openalex_sample['abstract'] = openalex_sample['abstract'].apply(lambda x: x if len(x.split()) > 10 else None)
openalex_sample = openalex_sample[openalex_sample['abstract'].notna()]
openalex_sample.reset_index(drop=True, inplace=True)
print(f"Number of rows after cleaning: {len(openalex_sample)} ({round((len(openalex_sample))/len(openalex_id_filtered), 2) * 100}% remain)")

# Load the aho automation for the core keywords
print("Loading aho automation...")
cso_aho_automation = ahocorasick.load('data/dictionaries/core_aho_automation/cso_aho_automation.pkl', pickle.loads)

openalex_sample_aho = openalex_sample.copy()
# Apply the automation on the abstracts and make a new column with all the results
print("Applying aho automation...")
openalex_sample_aho['aho_results'] = openalex_sample_aho['abstract'].progress_apply(lambda x: list(cso_aho_automation.iter_long(x)))
# Extract only the keywords, not the positions
openalex_sample_aho['aho_results'] = openalex_sample_aho['aho_results'].apply(lambda x: [y[1][1] for y in x])
# Make a new column aho_length with the length of the results
openalex_sample_aho['aho_length'] = openalex_sample_aho['aho_results'].apply(lambda x: len(x))
# Sort by aho_length descending
openalex_sample_aho.sort_values(by='aho_length', ascending=False, inplace=True)
# Reset the index
openalex_sample_aho.reset_index(drop=True, inplace=True)

openalex_sample_aho_cut = openalex_sample_aho.copy()
# Remove all rows with an aho_length of x or more
print(f"Found all core ai keywords in abstracts, will now remove all rows with an aho_length of {AHO_LENGTH} or more.")
openalex_sample_aho_cut = openalex_sample_aho_cut[openalex_sample_aho_cut['aho_length'] < AHO_LENGTH]
print(f"Number of rows after removing all rows with an aho_length of {AHO_LENGTH} or more: {len(openalex_sample_aho_cut)} ({round(len(openalex_sample_aho_cut)/len(openalex_sample), 3) * 100}% remain)")

# Get the keywords (this will take some time)
negative_keywords_df = get_clean_keywords(openalex_sample_aho_cut, ["title", "abstract"])

if not os.path.exists('data/dictionaries/'):
    os.makedirs('data/dictionaries/')

# Only keep the columns "title", "abstract", "title_keywords", "abstract_keywords", "id"
negative_keywords_df = negative_keywords_df[["title", "abstract", "title_keywords", "abstract_keywords", "id"]]
negative_keywords_df.reset_index(drop=True, inplace=True)
# Save the csv
print("Saving the negative keywords csv...")
negative_keywords_df.to_csv(f"data/dictionaries/negative_keywords_papers.csv", index=False)

paper_title_keywords = [keyword for keywords in negative_keywords_df.title_keywords.tolist() 
                        for keyword, score in keywords]
print(f"Got {len(paper_title_keywords)} keywords for negative paper titles.")
paper_abs_keywords = [keyword for keywords in negative_keywords_df.abstract_keywords.tolist() 
                        for keyword, score in keywords]
print(f"Got {len(paper_abs_keywords)} keywords for negative paper abstracts.")

all_paper_keywords = paper_title_keywords + paper_abs_keywords
print(f"TOTAL: {len(all_paper_keywords)} keywords for papers titles and abstracts.")

all_paper_keywords_dedupe = (pd.DataFrame(all_paper_keywords, columns=['keyword'])
                               .groupby('keyword')
                               .size()
                               .reset_index(name='frequency')
                               .sort_values(by='frequency', ascending=False)
                               .assign(frequency_normalized=lambda df: df['frequency'] / df['frequency'].max()))
all_paper_keywords_dedupe = all_paper_keywords_dedupe.reset_index(drop=True)
print(f"UNIQUE: {len(all_paper_keywords_dedupe)} unique keywords for paper titles and abstracts.")

# Save the all_paper_keywords_dedupe df
print("Saving the all_paper_keywords_dedupe df...")
all_paper_keywords_dedupe.to_csv(f"data/dictionaries/negative_keywords.csv", index=False)

print("")
print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")