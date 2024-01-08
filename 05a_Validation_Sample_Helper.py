from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
# register tqdm with pandas
tqdm.pandas()
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, URL, text
import time, os, pickle
# Initialize spacy nlp pipeline
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map  # Import from tqdm
import os

from helper.keyword_helper import neo4j_fetch_data, clean_abstracts, get_clean_keywords

CREATE_SAMPLES = False
NEGATIVE_SAMPLE_LIMIT = 2000000
SAMPLE_SIZE = None
YAKE_SCORE_THRESHOLD = 0.1 # Use it with YAKEscore(x) <= YAKE_SCORE_THRESHOLD

CREATE_TFIDF = False

GENERATE_KEYWORDS_NEGATIVE_SAMPLE = False
GENERATE_KEYWORDS_POSITIVE_SAMPLE = True

SAMPLE_PATH = "data/samples"
MODEL_PATH = "data/models"
DICT_PATH = "data/dictionaries"
NEO4J_CREDENTIALS = {"url": "bolt://localhost:37687", "user": "neo4j", "password": "neo4jpassword"}

# Check if SAMPLE_PATH, MODEL_PATH and DICT_PATH exist. If not, create them.
if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)

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

# Function to process a single row
def process_row(text):
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc])
    no_stopwords = ' '.join([word for word in lemmatized.split() if word not in nlp.Defaults.stop_words])
    return no_stopwords.lower()

# Main function to clean samples
def clean_samples(df, column_name):
    df[column_name] = df[column_name].astype(str)
    # Process in parallel using process_map from tqdm
    num_cores = cpu_count()
    df[column_name] = process_map(process_row, df[column_name], max_workers=num_cores, chunksize=CHUNKSIZE)
    return df

SAMPLE_KEYPHRASES = None

CHUNKSIZE = 1000
ITERATIONS = 500000
USE_PREPARED_SAMPLES = True
USE_PREPARED_VECTORIZER = False
USE_PREPARED_LOGREG = True

VISUALIZE_TFIDF = False
VISUALIZE_TFIDF_WITH_REG = True

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

print("")
print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print("")

if CREATE_SAMPLES:
    
    print("Creating positive sample...")
    query = """
    MATCH (t:Task)
    WHERE t.name IS NOT NULL AND t.description IS NOT NULL
    RETURN t.id as task_id, t.name as task_name, t.description as task_description
    """
    print("Fetching data...")
    tasks = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    tasks['text'] = tasks['task_name'] + " " + tasks['task_description']
    print(f"Got {len(tasks)} tasks.")

    query = """
    MATCH (m:Method)
    WHERE m.name IS NOT NULL AND m.description IS NOT NULL
    RETURN m.id as method_id, m.name as method_name, m.description as method_description
    """
    print("Fetching data...")
    methods = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    methods['text'] = methods['method_name'] + " " + methods['method_description']
    print(f"Got {len(methods)} methods.")
    
    query = """
    MATCH (p:Paper)
    WHERE p.id_openalex IS NOT NULL
    RETURN p.id_openalex AS openalex_id
    """
    print("Fetching data...")
    pwc_oalex_ids = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    print(f"Got {len(pwc_oalex_ids)} pwc openalex ids.")

    # Get Abstracts
    query2 = f"""
    MATCH (p:Paper)
    WHERE p.abstract IS NOT NULL
    RETURN p.id_openalex AS openalex_id, p.abstract AS abstract
    """
    print("Fetching data...")
    ai_papers = neo4j_fetch_data(query2, NEO4J_CREDENTIALS)
    print("Done.")
    print(f"Got {len(ai_papers)} papers.")
    print("Done.")

    # Make a new dataframe called positive_sample that contains only the AI abstracts in the column text
    positive_sample = pd.DataFrame(ai_papers['abstract'])
    positive_sample = positive_sample.rename(columns={'abstract': 'text'})
    # append the tasks and methods to the positive sample with the column text. Use concat and make sure that the text is the first column "text"
    positive_sample = pd.concat([positive_sample, tasks[['text']], methods[['text']]], ignore_index=True, axis=0)
    # Make text column string
    positive_sample['text'] = positive_sample['text'].astype(str)
    # Remove all rows that have a text shorter than 10 characters
    positive_sample = positive_sample[positive_sample['text'].str.len() > 20]

    positive_sample = clean_samples(positive_sample[['text']], 'text')

    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} papers...")
        positive_sample = positive_sample.sample(n=SAMPLE_SIZE, random_state=42)
        print("Done.")
    
    sql_query = f'''
        SELECT id, title, abstract, abstract_inverted_index
        FROM openalex.works TABLESAMPLE BERNOULLI(10)
        WHERE abstract_inverted_index IS NOT NULL
        LIMIT {NEGATIVE_SAMPLE_LIMIT + (NEGATIVE_SAMPLE_LIMIT * 0.5)}
    '''

    sql_query2 = f'''
        SELECT wc.work_id
        FROM openalex.works_concepts wc
        JOIN openalex.concepts c ON wc.concept_id = c.id
        WHERE c.display_name = 'Artificial intelligence' OR c.display_name = 'Machine learning' OR c.display_name = 'Natural language processing' AND wc.score > 0.2
    '''

    with engine.connect() as conn:
        print("Fetching abstracts from OpenAlex Postgres...")
        openalex = pd.read_sql(sql=text(sql_query), con=conn)
        print(f"Got {len(openalex)} OpenAlex Works")
        print("Fetching OpenAlex Work IDs with AI for filtering purposes...")
        openalex_ids_with_AI = pd.read_sql(sql=text(sql_query2), con=conn)
        print(f"Got {len(openalex_ids_with_AI)} OpenAlex Work IDs with AI")
        
    # Remove all openalex rows where the id is in the pwc_oalex df
    openalex_id_filtered = openalex[~openalex['id'].isin(pwc_oalex_ids['openalex_id'])]
    print(f"Got {len(openalex_id_filtered)} OpenAlex Works after removing PwC papers")
    
    # Remove all openalex rows where the id is in the openalex_ids_with_AI df
    openalex_id_filtered = openalex_id_filtered[~openalex_id_filtered['id'].isin(openalex_ids_with_AI['work_id'])]
    print(f"Got {len(openalex_id_filtered)} OpenAlex Works after removing papers with AI")
    
    print("Cleaning abstracts from OpenAlex...")
    openalex_sample = clean_abstracts(openalex_id_filtered.copy(), "abstract", openalex_inv_index=True, lang='en')
    openalex_sample.reset_index(drop=True, inplace=True)
    openalex_sample.drop(columns=['abstract_inverted_index', 'abstract_lang'], inplace=True)
    
    print(f"Got {len(openalex_id_filtered)} OpenAlex Works after removing PwC papers")
    print(f"Number of rows after cleaning: {len(openalex_sample)} ({round((len(openalex_sample))/len(openalex_id_filtered), 2) * 100}% remain)")

    non_ai_papers = openalex_sample

    negative_sample = pd.DataFrame(non_ai_papers['abstract'])
    negative_sample = negative_sample.rename(columns={'abstract': 'text'})
    negative_sample = clean_samples(negative_sample, 'text')
    negative_sample = negative_sample[['text']]
    
    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} papers...")
        positive_sample = positive_sample.sample(n=SAMPLE_SIZE, random_state=42)
        negative_sample = negative_sample.sample(n=SAMPLE_SIZE, random_state=42)
        print("Done.")

    # Save the dataframes as parquet files
    print("Saving data...")
    positive_sample.to_parquet(os.path.join(SAMPLE_PATH, "positive_sample.parquet"))
    negative_sample.to_parquet(os.path.join(SAMPLE_PATH, "negative_sample.parquet"))
    print("Done.")

if CREATE_TFIDF:
    if CREATE_SAMPLES == False:
        print("Loading data...")
        positive_sample = pd.read_parquet(os.path.join(SAMPLE_PATH, "positive_sample.parquet"))
        negative_sample = pd.read_parquet(os.path.join(SAMPLE_PATH, "negative_sample.parquet"))
        print("Done.")
        
    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} papers...")
        positive_sample = positive_sample.sample(n=SAMPLE_SIZE, random_state=42)
        negative_sample = negative_sample.sample(n=SAMPLE_SIZE, random_state=42)
        print("Done.")
    
    # Replace these with your actual data
    positive_sample_list = positive_sample['text'].tolist()
    negative_sample_list = negative_sample['text'].tolist()
    print(f"Got {len(positive_sample_list)} positive samples and {len(negative_sample_list)} negative samples.")
        
    # Combining datasets and creating labels
    abstracts = positive_sample_list + negative_sample_list
    labels = [1] * len(positive_sample_list) + [0] * len(negative_sample_list)  # 1 for AI, 0 for non-AI
        
    # Get Keywords
    print("Fetching keywords...")
    keywords = pd.read_csv(DICT_PATH + "/dictionary.csv")
    # Deduplication
    keywords = keywords.drop_duplicates(subset=['keyword'])
    # Make keyword column string
    keywords['keyword'] = keywords['keyword'].astype(str)
    print("Done.")

    if SAMPLE_KEYPHRASES:
        print(f"Sampling {SAMPLE_KEYPHRASES} keywords...")
        keywords = keywords.sample(n=SAMPLE_KEYPHRASES, random_state=42)
        print("Done.")

    # key phrases
    key_phrases = keywords['keyword'].tolist()

    # Vectorization with tqdm for progress tracking
    print("Vectorizing abstracts...")
    vectorizer = TfidfVectorizer(
        vocabulary=key_phrases,
        ngram_range=(1, 4),
        # max_features=10000,
        max_df=0.95,
        min_df=2,
        stop_words='english',
        use_idf=True,
        smooth_idf=True,
        norm='l2',
        decode_error='replace',
        strip_accents='unicode',
        lowercase=True,
        # sublinear_tf=True
    )
    X = vectorizer.fit_transform(tqdm(abstracts, desc="Vectorizing abstracts"))
    pickle.dump(X, open(os.path.join(MODEL_PATH, "X.pkl"), 'wb'))
    pickle.dump(vectorizer, open(os.path.join(MODEL_PATH, "vectorizer.pkl"), 'wb'))
    print("Done.")
    
if GENERATE_KEYWORDS_NEGATIVE_SAMPLE:
    if CREATE_SAMPLES == False:
        print("Loading data...")
        negative_sample = pd.read_parquet(os.path.join(SAMPLE_PATH, "negative_sample.parquet"))
        print("Done.")
        
    if SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} papers...")
        negative_sample = negative_sample.sample(n=SAMPLE_SIZE, random_state=42)
        print("Done.")
        
    negative_keywords = get_clean_keywords(negative_sample, ['text'])
    
    # Make a new column "keywords" that contains only the keywords as a list and filter out keywords with a score higher or equal than YAKE_SCORE_THRESHOLD
    negative_keywords['keywords'] = negative_keywords['text_keywords'].apply(lambda x: [keyword for keyword, score in x if score <= YAKE_SCORE_THRESHOLD])
    
    # Make a new column "keywords string" that concatenates all keywords in the list to a string with spaces in between
    # negative_keywords['keywords_string'] = negative_keywords['keywords'].apply(lambda x: " ".join(x).strip())
    
    # Drop the column "text_keywords"
    negative_keywords.drop(columns=['text_keywords'], inplace=True)
    
    # Save the dataframe as parquet file
    print("Saving data...")
    negative_keywords.to_parquet(os.path.join(SAMPLE_PATH, "negative_keywords.parquet"))
    print("Done.")
    
if GENERATE_KEYWORDS_POSITIVE_SAMPLE:
    # TASKS
    query = f"""
    MATCH (t:Task)-[r:HAS_KEYWORD]->(k:Keyword)
    WHERE toFloat(r.score) <= {YAKE_SCORE_THRESHOLD}
    RETURN t.id AS id, collect(k.keyword) AS keywords
    """
    print("Fetching data...")
    tasks = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    print(f"Got {len(tasks)} task keywords.")

    # METHODS
    query = f"""
    MATCH (m:Method)-[r:HAS_KEYWORD]->(k:Keyword)
    WHERE toFloat(r.score) <= {YAKE_SCORE_THRESHOLD}
    RETURN m.id AS id, collect(k.keyword) AS keywords
    """
    print("Fetching data...")
    methods = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    print(f"Got {len(methods)} method keywords.")

    # PAPERS
    query = f"""
    MATCH (p:Paper)-[r:HAS_KEYWORD]->(k:Keyword)
    WHERE toFloat(r.score) <= {YAKE_SCORE_THRESHOLD}
    RETURN p.id AS id, collect(k.keyword) AS keywords
    """
    print("Fetching data...")
    papers = neo4j_fetch_data(query, NEO4J_CREDENTIALS)
    print("Done.")
    print(f"Got {len(papers)} paper keywords.")

    # Combine
    positive_sample = pd.concat([tasks, methods, papers], ignore_index=True)
    # Drop id column
    positive_sample.drop(columns=["id"], inplace=True)
    
    # Save the dataframe as parquet file
    print("Saving data...")
    positive_sample.to_parquet(os.path.join(SAMPLE_PATH, "positive_keywords.parquet"))
    print("Done.")

print("")
print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")