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
import unicodedata

import re

from helper.wordtrie_builder import WordTrie

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
    numOfKeywords = 10
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
        row[column_name] = unicodedata.normalize("NFKD", row[column_name]).encode('ASCII', 'ignore').decode('utf-8')
        row[column_name] = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", row[column_name])
        row[column_name] = re.sub(r"https?:\/\/\S+", "", row[column_name])
        row[column_name] = re.sub(r"[^a-zA-Z- ]", " ", row[column_name]).lower().strip()
        unfiltered_keywords = yake_extractor.extract_keywords(row[column_name]) # Extract keywords using YAKE
        doc = nlp(row[column_name])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
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

def representation_generator(keyword_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    model = "google/flan-t5-large"
    generator = pipeline('text2text-generation', model=model, device=device)
    # prompt = "I have a general topic described by the following keywords: [KEYWORDS]. Based on these keywords, what is this topic about? Be specific, precise and short! Don't use names or numbers!"
    prompt = "Based on the following keywords, come up with a topic name that is specific and precise: [KEYWORDS]"
    representation_list = []
    for keywords in tqdm(keyword_list, desc="Generating representations..."):
        keyword_string = ', '.join(keywords)[:2000]
        prompt = prompt.replace('[KEYWORDS]', keyword_string)
        representation_list.append(representation_generator_helper(keyword_string, generator, prompt))
    return representation_list

def representation_generator_helper(keyword_string, generator, prompt):
    representation = generator(
        prompt,
        max_length=10,
        do_sample=True,
        temperature=0.9
    )[0]['generated_text']
    return representation

# Perform cleaning operations
def clean_abstracts(df, column_name, openalex_inv_index=False, lang='en'):
    if openalex_inv_index:
        # Extract and join keys from abstract_inverted_index
        print(f"Extracting {column_name} from abstract_inverted_index...")
        df[column_name] = df[f'{column_name}_inverted_index'].progress_apply(lambda x: " ".join(x['InvertedIndex'].keys()) if x else None)
    
    # Regular expressions and string operations
    df[column_name] = df[column_name].str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)\
                                      .str.replace(r"abstract", "", flags=re.IGNORECASE)\
                                      .str.strip()\
                                      .str.replace(r"\s+", " ", regex=True)\
                                      .str.lower()
    
    # Filter by abstract length
    df = df[df[column_name].str.split().str.len() > 10]
        
    if lang:
        try:
            # Detect language and keep only English abstracts
            import gcld3
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
            print(f"Detecting language {lang}...")
            # Sample output 
            # print(detector.FindLanguage(text="This is a test").language)
            df[f'{column_name}_lang'] = df[column_name].progress_apply(lambda x: detector.FindLanguage(text=x).language if pd.notna(x) else None)
            df = df[df[f'{column_name}_lang'] == lang]
        except:
            print("Language detection failed. Keeping all abstracts.")
            pass

    # Calculate abstract length by counting the words
    df[f'{column_name}_length'] = df[column_name].str.split().str.len()

    return df[df[f'{column_name}_length'] > 10]

def make_wordtrie(keyword_list, id_list):    
    if keyword_list is None:
        return None
    # Make a wordtrie from the dict_df
    wordtrie = WordTrie(word_filter=True, text_filter=True, show_progress_bar=True, weights=False)
    wordtrie.add_bulk(keyword_list, id_list)
    return wordtrie

def make_weight_wordtrie(keyword_list, id_list, weights_list):    
    if keyword_list is None:
        return None
    # Make a wordtrie from the dict_df
    wordtrie = WordTrie(word_filter=True, text_filter=True, show_progress_bar=True, weights=True)
    wordtrie.add_bulk(keyword_list, id_list, weights_list)
    return wordtrie