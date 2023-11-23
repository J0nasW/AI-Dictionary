import pandas as pd
import time
import hashlib
from sqlalchemy import create_engine, URL, text
from tqdm import tqdm

from helper.keyword_helper import get_clean_keywords, get_sentence_embeddings

KEYWORDS = True
EMBEDDINGS = True
EMBEDDING_MODEL = "malteos/scincl"

OPENALEX_IDS = True
OPENALEX_AUTHOR_IDS = True

OPENALEX_WORKS = True
OPENALEX_AUTHORS = True
OPENALEX_INSTITUTIONS = True

OPENALEX_CITATIONS = True

PROCESSED_JSON_PATH = "data/pwc_processed_json/"
NEO4J_PATH = "data/neo4j/"
TEST_HEAD = None # Set to None to use all rows

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

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

if __name__ == "__main__":
    print("---")
    print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("")
    print(f"Reading json files from {PROCESSED_JSON_PATH}...")
    papers = pd.read_json(PROCESSED_JSON_PATH + "papers.json", dtype={"id": str})
    methods = pd.read_json(PROCESSED_JSON_PATH + "methods.json", dtype={"id": str})
    datasets = pd.read_json(PROCESSED_JSON_PATH + "datasets.json", dtype={"id": str})
    areas = pd.read_json(PROCESSED_JSON_PATH + "areas.json", dtype={"id_md5": str})
    tasks = pd.read_json(PROCESSED_JSON_PATH + "tasks.json", dtype={"area_id_md5": str, "id_md5": str})
    print("Done reading json files.")

    if TEST_HEAD is not None:
        print("")
        print(f"Cutting files to {TEST_HEAD} rows for testing...")
        papers = papers.head(TEST_HEAD)
        methods = methods.head(TEST_HEAD)
        datasets = datasets.head(TEST_HEAD)
        areas = areas.head(TEST_HEAD)
        tasks = tasks.head(TEST_HEAD)
        
    if KEYWORDS:
        try:
            print("")
            print("Extracting keywords...")
            timestamp = time.time()
            
            papers = get_clean_keywords(papers, ["abstract"])
            methods = get_clean_keywords(methods, ["description"])
            tasks = get_clean_keywords(tasks, ["description"])
            datasets = get_clean_keywords(datasets, ["description"])
            
            print("")
            print(f"Done extracting keywords. Time elapsed: {format_time(time.time() - timestamp)}")
            print("")
        except Exception as e:
            print("Error calculating keywords.")
            print(e)
        
    if EMBEDDINGS:
        try:
            print("")
            print("[EMBEDDINGS]")
            print("Calculating embeddings...")
            timestamp = time.time()
            timestamp_master = timestamp

            papers["title_embedding"] = get_sentence_embeddings(papers, "title", EMBEDDING_MODEL)
            papers["abstract_embedding"] = get_sentence_embeddings(papers, "abstract", EMBEDDING_MODEL)
            print(f"Done calculating paper embeddings (title, abstract). Time elapsed: {format_time(time.time() - timestamp)}")
            timestamp = time.time()

            methods["name_embedding"] = get_sentence_embeddings(methods, "name", EMBEDDING_MODEL)
            methods["description_embedding"] = get_sentence_embeddings(methods, "description", EMBEDDING_MODEL)
            print(f"Done calculating method embeddings (name, description). Time elapsed: {format_time(time.time() - timestamp)}")
            timestamp = time.time()

            tasks["name_embedding"] = get_sentence_embeddings(tasks, "name", EMBEDDING_MODEL)
            tasks["description_embedding"] = get_sentence_embeddings(tasks, "description", EMBEDDING_MODEL)
            print(f"Done calculating task embeddings (name, description). Time elapsed: {format_time(time.time() - timestamp)}")
            timestamp = time.time()

            datasets["name_embedding"] = get_sentence_embeddings(datasets, "name", EMBEDDING_MODEL)
            datasets["description_embedding"] = get_sentence_embeddings(datasets, "description", EMBEDDING_MODEL)
            print(f"Done calculating dataset embeddings (name, description). Time elapsed: {format_time(time.time() - timestamp)}")
            timestamp = time.time()

            areas["name_embedding"] = get_sentence_embeddings(areas, "name", EMBEDDING_MODEL)
            print(f"Done calculating area embeddings (name, description). Time elapsed: {format_time(time.time() - timestamp)}")
            print("")
            
            print(f"Done calculating embeddings. Total time elapsed: {format_time(time.time() - timestamp_master)}")
            print("")
        except Exception as e:
            print("Error calculating embeddings.")
            print(e)
        
    if OPENALEX_IDS:
        try:
            print("Fetching all paper-related OpenAlex IDs...")
            with engine.connect() as conn:
                timestamp = time.time()
                query = """
                    SELECT *
                    FROM openalex.works_best_oa_locations
                    WHERE landing_page_url LIKE '%%arxiv%%';
                """
                openalex_ids = pd.read_sql(sql=text(query), con=conn)
                print(f"Got {len(openalex_ids)} OpenAlex IDs.")
                openalex_ids["arxiv_id"] = openalex_ids["landing_page_url"].apply(lambda x: x[x.rfind("/") + 1:])
                merged_df = papers.merge(openalex_ids[['arxiv_id', 'work_id']], on='arxiv_id', how='left')
                papers = merged_df.rename(columns={'work_id': 'openalex_id'})
                print(f"Got {papers['openalex_id'].notnull().sum()} OpenAlex IDs from the arXiv IDs ({(papers['openalex_id'].notnull().sum() / papers['arxiv_id'].notnull().sum()) * 100}%).")
            print(f"Done fetching all arxiv related OpenAlex IDs. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching openalex_ids from OpenAlex.")
            print(e)
        
    if OPENALEX_WORKS:
        try:
            print("Fetching works from OpenAlex...")
            with engine.connect() as conn:
                timestamp = time.time()
                openalex_ids_single = pd.DataFrame(papers["openalex_id"].unique(), columns=["oalex_id"])        
                print(f"Got {len(openalex_ids_single)} OpenAlex IDs ({(len(openalex_ids_single) / len(papers)) * 100}%).")
                openalex_ids_single.to_sql("temp_table_works", con=conn, schema="openalex", if_exists="replace", index=False)
                query = """
                    SELECT *
                    FROM openalex.works
                    JOIN openalex.temp_table_works ON openalex.works.id = openalex.temp_table_works.oalex_id;
                """
                result = pd.read_sql(sql=text(query), con=conn)
                columns_to_drop = [result.columns[-1], "abstract_inverted_index", "display_name", "title", "abstract", "abstract_embedding"]
                result = result.drop(columns=columns_to_drop)
                papers = papers.merge(result, left_on="openalex_id", right_on="id", suffixes=("", "_openalex"))
                papers["id"] = papers["id"].astype(str)
                # result = result.drop(columns=["id_openalex"])
                print(f"Got {len(result)} works from OpenAlex ({(len(result) / len(openalex_ids_single)) * 100}%).")
            print(f"Done fetching works from OpenAlex. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching works from OpenAlex.")
            print(e)
        
    if OPENALEX_AUTHOR_IDS:
        try:
            print("Fetching author information for each OpenAlex paper...")
            with engine.connect() as conn:
                timestamp = time.time()
                openalex_ids_single = pd.DataFrame(papers["openalex_id"].unique(), columns=["oalex_id"])
                print(f"Got {len(openalex_ids_single)} OpenAlex IDs ({(len(openalex_ids_single) / len(papers)) * 100}%).")
                openalex_ids_single.to_sql("temp_table_authors", con=conn, schema="openalex", if_exists="replace", index=False)
                query = """
                    SELECT *
                    FROM openalex.works_authorships
                    JOIN openalex.temp_table_authors ON openalex.works_authorships.work_id = openalex.temp_table_authors.oalex_id;
                """
                result = pd.read_sql(sql=text(query), con=conn)
                columns_to_drop = [result.columns[-1], "raw_affiliation_string"]
                result = result.drop(columns=columns_to_drop)
                
                # Neo4J Edges
                work_author_edges_csv = result[["work_id", "author_id", "author_position", "institution_id"]]
                work_author_edges_csv = work_author_edges_csv.merge(papers[["openalex_id", "id"]], left_on="work_id", right_on="openalex_id")
                work_author_edges_csv = work_author_edges_csv.drop(columns=["work_id", "openalex_id"])
                work_author_edges_csv["author_id"] = ("author/" + work_author_edges_csv["author_id"]).apply(lambda x: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)))
                work_author_edges_csv = work_author_edges_csv.rename(columns={"author_id": ":START_ID", "id": ":END_ID",})
                work_author_edges_csv[":START_ID"] = work_author_edges_csv[":START_ID"].astype(str)
                work_author_edges_csv[":END_ID"] = work_author_edges_csv[":END_ID"].astype(str)
                work_author_edges_csv[":TYPE"] = "AUTHORS"
                work_author_edges_csv.to_csv(NEO4J_PATH + "authors_papers.csv", index=False)
                print(f"Saved {len(work_author_edges_csv)} authorship edges to {NEO4J_PATH}authors_papers.csv.")
                
                result = result.groupby("work_id").apply(lambda x: x.to_dict(orient="records")).reset_index(name="authorships")
                papers = papers.merge(result, left_on="openalex_id", right_on="work_id", suffixes=("", "_openalex_authors"))
                papers["id"] = papers["id"].astype(str)
                print(f"Got {len(result)} authorships from OpenAlex ({(len(result) / len(openalex_ids_single)) * 100}%).")
            print(f"Done fetching author information for each OpenAlex paper. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching author_ids from OpenAlex.")
            print(e)
        
    if OPENALEX_AUTHORS:
        try:
            print("Fetching authors from OpenAlex...")
            with engine.connect() as conn:
                # Get all author ids from the papers dataframe
                timestamp = time.time()
                # Fetch all author_ids from the papers dataframe stored in the column authorships as list of dicts
                author_ids_single = pd.DataFrame([author["author_id"] for authorships in papers["authorships"] for author in authorships], columns=["author_id"])
                print(f"Got {len(author_ids_single)} OpenAlex Author IDs.")
                author_ids_single.to_sql("temp_table_authors", con=conn, schema="openalex", if_exists="replace", index=False)
                query = """
                    SELECT *
                    FROM openalex.authors
                    JOIN openalex.temp_table_authors ON openalex.authors.id = openalex.temp_table_authors.author_id;
                """
                result = pd.read_sql(sql=text(query), con=conn)
                columns_to_drop = [result.columns[-1], "display_name_alternatives"]
                authors = result.drop(columns=columns_to_drop)
                authors["id_md5"] = ("author/" + authors["id"]).apply(lambda x: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)))
                authors.to_json(PROCESSED_JSON_PATH + "authors.json", orient="records")
                
                # Neo4J Nodes
                authors_csv = authors.rename(columns={"id": "openalex_id"})
                authors_csv = authors_csv.rename(columns={"id_md5": "id:ID"})
                authors_csv = authors_csv[["id:ID"] + [col for col in authors_csv.columns if col != "id:ID"]]
                # Make a label for the authors
                authors_csv[":LABEL"] = "Author"
                authors_csv.to_csv(NEO4J_PATH + "authors.csv", index=False)
                
                print(f"Got {len(authors)} authors from OpenAlex.")
            print(f"Done fetching authors from OpenAlex. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching authors from OpenAlex.")
            print(e)
        
    if OPENALEX_INSTITUTIONS:
        try:
            print("Fetching institution information for each author...")
            with engine.connect() as conn:
                timestamp = time.time()
                # Fetch all institution_ids from the authors dataframe stored in the column last_known_institution
                institution_ids_single = pd.DataFrame(authors["last_known_institution"].unique(), columns=["institution_id"])
                print(f"Got {len(institution_ids_single)} OpenAlex Institution IDs.")
                institution_ids_single.to_sql("temp_table_institutions", con=conn, schema="openalex", if_exists="replace", index=False)
                query = """
                    SELECT *
                    FROM openalex.institutions
                    JOIN openalex.temp_table_institutions ON openalex.institutions.id = openalex.temp_table_institutions.institution_id;
                """
                result = pd.read_sql(sql=text(query), con=conn)
                columns_to_drop = [result.columns[-1]]
                institutions = result.drop(columns=columns_to_drop)
                institutions["id_md5"] = ("institution/" + institutions["id"]).apply(lambda x: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)))
                institutions.to_json(PROCESSED_JSON_PATH + "institutions.json", orient="records")
                
                # Neo4J Nodes
                institutions_csv = institutions.rename(columns={"id": "openalex_id"})
                institutions_csv = institutions_csv.rename(columns={"id_md5": "id:ID"})
                institutions_csv = institutions_csv[["id:ID"] + [col for col in institutions_csv.columns if col != "id:ID"]]
                institutions_csv[":LABEL"] = "Institution"
                institutions_csv.to_csv(NEO4J_PATH + "institutions.csv", index=False)
                
                # Neo4J Edges
                author_institution_edges_csv = authors[["id_md5", "last_known_institution"]].rename(columns={"id_md5": ":START_ID", "last_known_institution": ":END_ID"})
                author_institution_edges_csv[":START_ID"] = author_institution_edges_csv[":START_ID"].astype(str)
                author_institution_edges_csv[":END_ID"] = author_institution_edges_csv[":END_ID"].astype(str)
                author_institution_edges_csv[":END_ID"] = ("institution/" + author_institution_edges_csv[":END_ID"]).apply(lambda x: str(int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)))
                author_institution_edges_csv[":TYPE"] = "AFFILIATED_WITH"
                author_institution_edges_csv.to_csv(NEO4J_PATH + "authors_institutions.csv", index=False)
                        
                print(f"Got {len(institutions)} institutions from OpenAlex.")
            print(f"Done fetching institution information for each author. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching institution information for each author.")
            print(e)
        
    if OPENALEX_CITATIONS:
        try:
            print("Fetching citations from OpenAlex...")
            with engine.connect() as conn:
                timestamp = time.time()
                openalex_ids_single = pd.DataFrame(papers["openalex_id"].unique(), columns=["oalex_id"])
                print(f"Got {len(openalex_ids_single)} OpenAlex IDs ({(len(openalex_ids_single) / len(papers)) * 100}%).")
                openalex_ids_single.to_sql("temp_table_citations", con=conn, schema="openalex", if_exists="replace", index=False)
                query = """
                    SELECT *
                    FROM openalex.works_referenced_works
                    JOIN openalex.temp_table_citations ON openalex.works_referenced_works.work_id = openalex.temp_table_citations.oalex_id;
                """
                result = pd.read_sql(sql=text(query), con=conn)
                print(f"Got {len(result)} citations to {len(openalex_ids_single)} Works from OpenAlex.")
                columns_to_drop = [result.columns[-1]]
                result = result.drop(columns=columns_to_drop)
                
                # In the result df, drop all rows, where the referenced_work_id is not in the openalex_ids_single df
                result_filtered = result[result["referenced_work_id"].isin(openalex_ids_single["oalex_id"])]
                result_filtered = result_filtered.reset_index(drop=True)
                print(f"Got {len(result_filtered)} citations after dropping citations outside this dataset ({(len(result_filtered) / len(result)) * 100}%).")
                
                # Neo4J Edges
                work_citation_edges_csv = result_filtered[["work_id", "referenced_work_id"]]
                # Get the paper ids from the papers dataframe by comparing the work_id and referenced_work_id with the openalex_id in the papers dataframe
                work_citation_edges_csv = work_citation_edges_csv.merge(papers[["openalex_id", "id"]], left_on="work_id", right_on="openalex_id")
                work_citation_edges_csv = work_citation_edges_csv.rename(columns={"id": ":START_ID"})
                work_citation_edges_csv = work_citation_edges_csv.drop(columns=["openalex_id"])
                # Also translate the work_id to the paper id by merging with the papers dataframe
                work_citation_edges_csv = work_citation_edges_csv.merge(papers[["openalex_id", "id"]], left_on="referenced_work_id", right_on="openalex_id")
                work_citation_edges_csv = work_citation_edges_csv.rename(columns={"id": ":END_ID"})
                work_citation_edges_csv = work_citation_edges_csv.drop(columns=["openalex_id"])
                # Drop all unnecessary columns, only keep the md5 ids
                work_citation_edges_csv[":START_ID"] = work_citation_edges_csv[":START_ID"].astype(str)
                work_citation_edges_csv[":END_ID"] = work_citation_edges_csv[":END_ID"].astype(str)
                work_citation_edges_csv[":TYPE"] = "CITES"
                work_citation_edges_csv = work_citation_edges_csv.drop(columns=["work_id", "referenced_work_id"])
                work_citation_edges_csv.to_csv(NEO4J_PATH + "papers_citations.csv", index=False)
                
                result = result.groupby("work_id").apply(lambda x: x.to_dict(orient="records")).reset_index(name="citations")
                # For the merge with the papers dataframe, only list the referenced_work_id in the citations column
                result["citations"] = result["citations"].apply(lambda x: [citation["referenced_work_id"] for citation in x])
                papers = papers.merge(result, left_on="openalex_id", right_on="work_id", suffixes=("", "_openalex_citations"))
                print("Result query after merge")
                
                print(f"Got {len(work_citation_edges_csv)} citations from OpenAlex.")
            print(f"Done fetching citations from OpenAlex. Total time elapsed: {format_time(time.time() - timestamp)}")
        except Exception as e:
            print("Error fetching citations from OpenAlex.")
            print(e)
         
    print("Saving to json files...")
    papers.to_json(PROCESSED_JSON_PATH + "papers_processed.json", orient="records")
    methods.to_json(PROCESSED_JSON_PATH + "methods_processed.json", orient="records")
    tasks.to_json(PROCESSED_JSON_PATH + "tasks_processed.json", orient="records")
    datasets.to_json(PROCESSED_JSON_PATH + "datasets_processed.json", orient="records")
    areas.to_json(PROCESSED_JSON_PATH + "areas_processed.json", orient="records")
    print("Done saving to json files.")
    print("")
    
    print(f"END TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("---")
    print("Goodbye!")