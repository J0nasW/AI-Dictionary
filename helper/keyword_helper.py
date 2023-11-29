from neo4j import GraphDatabase
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import logging
logging.basicConfig(level=logging.ERROR)
from nltk.stem import WordNetLemmatizer
import spacy
# nlp = spacy.load("en_core_web_trf") # For better accuracy
nlp = spacy.load("en_core_web_sm") # For better efficiency
from tqdm.auto import tqdm
tqdm.pandas()
import yake
import ahocorasick
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.optimize import curve_fit

from sentence_transformers import SentenceTransformer

from transformers import pipeline
import torch

from openTSNE import TSNE
import openTSNE
from hdbscan import HDBSCAN

def neo4j_fetch_data(query, credentials):
    with GraphDatabase.driver(credentials["url"], auth=(credentials["user"], credentials["password"])) as driver:
        with driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())
    

def initialize_sentence_transformer(model_name):
    return SentenceTransformer(model_name)

def get_sentence_embeddings(df, column_name, model_name):
    sentence_transformer = initialize_sentence_transformer(model_name)
    sentences = [sentence if sentence is not None else '' for sentence in df[column_name].tolist()]
    return sentence_transformer.encode(sentences, show_progress_bar=True).tolist()

def initialize_yake():
    # Specify custom parameters for YAKE
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.25
    deduplication_algo = "seqm"
    windowSize = 5
    numOfKeywords = 15
    return yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                    features=None)
    
def worker_initializer():
    global yake_extractor, lemmatizer
    yake_extractor = initialize_yake()
    lemmatizer = WordNetLemmatizer()

def filter_keyword_length(keyword_list, length=3):
    if keyword_list is None:
        return []
    return [[keyword, score] for keyword, score in keyword_list if keyword and len(keyword) > length]

def extract_keywords_helper(args):
    return extract_keywords(*args)

def extract_keywords(row, column_name):
    try:
        if row[column_name] is None:
            return [], []
        # Extract keywords using YAKE
        unfiltered_keywords = yake_extractor.extract_keywords(row[column_name])
        doc = nlp(row[column_name])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except:
        return [], []
    
def make_aho_automation(keyword_list):
    # Create an automaton
    automaton = ahocorasick.Automaton()
    for index, keyword in enumerate(keyword_list):
        automaton.add_word(keyword, (index, keyword))
    automaton.make_automaton()
    return automaton

def get_clean_keywords(df, column_names):
    
    # Set up multiprocessing
    num_cores = max(1, int(0.8 * cpu_count()))  # Use only 80% of CPU cores
    
    # Process in batches to reduce memory footprint
    batch_size = 10000  # Adjust based on your RAM and dataframe size
    num_batches = int(np.ceil(len(df) / batch_size))
    
    # pool = Pool(num_cores)
    with Pool(num_cores, initializer=worker_initializer) as pool:

        for column_name in column_names:
            print(f"Generating keywords for {column_name}...")

            all_results = []
            for batch in tqdm(np.array_split(df, num_batches)):
                batch_results = pool.map(extract_keywords_helper, [(row, column_name) for _, row in batch.iterrows()])
                all_results.extend(batch_results)
            
            df[f"{column_name}_keywords"] = [filtered_keywords for unfiltered_keywords, filtered_keywords in all_results]
            print(f"Extracted keywords for {column_name}")

            # Remove all keywords with less than 4 characters
            df[f"{column_name}_keywords"] = df[f"{column_name}_keywords"].apply(filter_keyword_length, length=3)
            print(f"Filtered out keywords with less than 3 characters.")
            print(f"Ended up with {df[f'{column_name}_keywords'].apply(len).sum()} keywords.")

        pool.close()
        pool.join()
    
    return df

def process_keywords_for_papers(filtered_keywords, keyword_freq_range):
    # nlp = spacy.load("en_core_web_sm")
    # lemmatizer = WordNetLemmatizer()

    # def is_noun(keyword):
    #     return any(token.pos_ == 'NOUN' for token in nlp(keyword))

    # filtered_keywords = filtered_keywords[~filtered_keywords['keyword'].str.contains(r'[A-Z]{2,4}')]
    # filtered_keywords['keyword'] = filtered_keywords['keyword'].apply(lambda keyword: keyword.lower())
    # filtered_keywords = filtered_keywords.dropna()
    # filtered_keywords['keyword'] = filtered_keywords['keyword'].apply(lambda keyword: keyword if not keyword.startswith('(') and not keyword.endswith(')') else None)
    # filtered_keywords = filtered_keywords.dropna()

    # filtered_keywords['keyword'] = filtered_keywords['keyword'].progress_apply(lambda keyword: lemmatizer.lemmatize(keyword) if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word) for word in keyword.split()]))

    dedupe_keywords = filtered_keywords.drop_duplicates(subset=['keyword'])
    dedupe_keywords['frequency'] = filtered_keywords.groupby('keyword')['keyword'].transform('count')
    dedupe_keywords = dedupe_keywords.sort_values(by=['frequency'], ascending=False)
    dedupe_keywords = dedupe_keywords.reset_index(drop=True)

    aggregated_data = filtered_keywords.groupby('keyword').agg(
        paper_ids=pd.NamedAgg(column='paper_id', aggfunc=list),
    ).reset_index()

    dedupe_keywords = dedupe_keywords.merge(aggregated_data, on='keyword', how='left')
    dedupe_keywords['paper_counts'] = dedupe_keywords['paper_ids'].apply(lambda x: len(x))

    dedupe_keywords_f = dedupe_keywords[(dedupe_keywords['frequency'] >= keyword_freq_range[0]) & (dedupe_keywords['frequency'] <= keyword_freq_range[1])]

    # dedupe_keywords_f['keyword'] = dedupe_keywords_f['keyword'].progress_apply(lambda keyword: keyword if is_noun(keyword) else None)
    # dedupe_keywords_f = dedupe_keywords_f.dropna()

    dedupe_keywords_f = dedupe_keywords_f.drop_duplicates(subset=['keyword'])
    dedupe_keywords_f = dedupe_keywords_f.reset_index(drop=True)

    return dedupe_keywords_f

def get_tsne_coordinates(x):
    keyword_embeddings_affinities = openTSNE.affinity.Multiscale(
        x,
        perplexities=[50, 200],
        metric="cosine",
        n_jobs=8,
        random_state=3,
        verbose=True,
    )
    init = openTSNE.initialization.pca(
        x,
        random_state=42,
        verbose=True,
    )
    keyword_embeddings_tsne_2d = openTSNE.TSNE(
        n_jobs=8,
        verbose = True,
        ).fit(
        affinities=keyword_embeddings_affinities,
        initialization=init,
    )
    return keyword_embeddings_tsne_2d

def representation_generator(keywords):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = "google/flan-t5-large"
    generator = pipeline('text2text-generation', model=model, device=device)
    # prompt = "I have a general topic described by the following keywords: [KEYWORDS]. Based on these keywords, what is this topic about? Be specific, precise and short! Don't use names or numbers!"
    prompt = "Based on the following keywords, come up with a topic name that is specific and precise: [KEYWORDS]"
    keyword_string = ', '.join(keywords)[:2000]
    representation = generator(
        prompt.replace('[KEYWORDS]', keyword_string),
        max_length=10,
        do_sample=True,
        temperature=0.9
    )[0]['generated_text']
    return representation