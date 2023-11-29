import pandas as pd

def modify_csv_header(input_file, output_file, original_header, new_header):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Read the first line (header)
        header = infile.readline()
        
        # Replace the original header with the new header
        modified_header = header.replace(original_header, new_header)
        outfile.write(modified_header)

        # Copy the rest of the lines
        for line in infile:
            outfile.write(line)

modify_csv_header("data/neo4j/papers.csv", "data/neo4j2/papers.csv", "id:ID", "id:ID")
modify_csv_header("data/neo4j/authors.csv", "data/neo4j2/authors.csv", "id:ID", "id:ID(authors)")
modify_csv_header("data/neo4j/institutions.csv", "data/neo4j2/institutions.csv", "id:ID", "id:ID(institutions)")
modify_csv_header("data/neo4j/datasets.csv", "data/neo4j2/datasets.csv", "id:ID", "id:ID(datasets)")
modify_csv_header("data/neo4j/fulltexts.csv", "data/neo4j2/fulltexts.csv", "id:ID", "id:ID(fulltexts)")
modify_csv_header("data/neo4j/methods.csv", "data/neo4j2/methods.csv", "id:ID", "id:ID(methods)")
modify_csv_header("data/neo4j/repos.csv", "data/neo4j2/repos.csv", "id:ID", "id:ID(repos)")
modify_csv_header("data/neo4j/areas.csv", "data/neo4j2/areas.csv", "id:ID", "id:ID(areas)")
modify_csv_header("data/neo4j/tasks.csv", "data/neo4j2/tasks.csv", "id:ID", "id:ID(tasks)")
modify_csv_header("data/neo4j/keywords.csv", "data/neo4j2/keywords.csv", "id:ID", "id:ID(keywords)")

modify_csv_header("data/neo4j/authors_institutions.csv", "data/neo4j2/authors_institutions.csv", ":START_ID", ":START_ID(authors)")
modify_csv_header("data/neo4j/authors_institutions.csv", "data/neo4j2/authors_institutions.csv", ":END_ID", ":END_ID(institutions)")

modify_csv_header("data/neo4j/authors_papers.csv", "data/neo4j2/authors_papers.csv", ":START_ID", ":START_ID(authors)")
modify_csv_header("data/neo4j/authors_papers.csv", "data/neo4j2/authors_papers.csv", ":END_ID", ":END_ID")

modify_csv_header("data/neo4j/datasets_keywords.csv", "data/neo4j2/datasets_keywords.csv", ":START_ID", ":START_ID(datasets)")
modify_csv_header("data/neo4j/datasets_keywords.csv", "data/neo4j2/datasets_keywords.csv", ":END_ID", ":END_ID(keywords)")

modify_csv_header("data/neo4j/datasets_tasks.csv", "data/neo4j2/datasets_tasks.csv", ":START_ID", ":START_ID(datasets)")
modify_csv_header("data/neo4j/datasets_tasks.csv", "data/neo4j2/datasets_tasks.csv", ":END_ID", ":END_ID(tasks)")

modify_csv_header("data/neo4j/methods_keywords.csv", "data/neo4j2/methods_keywords.csv", ":START_ID", ":START_ID(methods)")
modify_csv_header("data/neo4j/methods_keywords.csv", "data/neo4j2/methods_keywords.csv", ":END_ID", ":END_ID(keywords)")

modify_csv_header("data/neo4j/papers_keywords.csv", "data/neo4j2/papers_keywords.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_keywords.csv", "data/neo4j2/papers_keywords.csv", ":END_ID", ":END_ID(keywords)")

modify_csv_header("data/neo4j/papers_fulltexts.csv", "data/neo4j2/papers_fulltexts.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_fulltexts.csv", "data/neo4j2/papers_fulltexts.csv", ":END_ID", ":END_ID(fulltexts)")

modify_csv_header("data/neo4j/papers_repos.csv", "data/neo4j2/papers_repos.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_repos.csv", "data/neo4j2/papers_repos.csv", ":END_ID", ":END_ID(repos)")

modify_csv_header("data/neo4j/papers_tasks.csv", "data/neo4j2/papers_tasks.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_tasks.csv", "data/neo4j2/papers_tasks.csv", ":END_ID", ":END_ID(tasks)")

modify_csv_header("data/neo4j/papers_methods.csv", "data/neo4j2/papers_methods.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_methods.csv", "data/neo4j2/papers_methods.csv", ":END_ID", ":END_ID(methods)")

modify_csv_header("data/neo4j/tasks_areas.csv", "data/neo4j2/tasks_areas.csv", ":START_ID", ":START_ID(tasks)")
modify_csv_header("data/neo4j/tasks_areas.csv", "data/neo4j2/tasks_areas.csv", ":END_ID", ":END_ID(areas)")

modify_csv_header("data/neo4j/tasks_keywords.csv", "data/neo4j2/tasks_keywords.csv", ":START_ID", ":START_ID(tasks)")
modify_csv_header("data/neo4j/tasks_keywords.csv", "data/neo4j2/tasks_keywords.csv", ":END_ID", ":END_ID(keywords)")

modify_csv_header("data/neo4j/papers_citations.csv", "data/neo4j2/papers_citations.csv", ":START_ID", ":START_ID")
modify_csv_header("data/neo4j/papers_citations.csv", "data/neo4j2/papers_citations.csv", ":END_ID", ":END_ID")
