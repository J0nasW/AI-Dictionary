# The AI Dictionary: A Text-Based Tool to Identify and Measure Technology Innovation

This repository is the official codebase for the paper "The AI Dictionary: The Foundation for a Text-Based Tool to Identify and Measure Technology Innovation". The paper explores the rapidly evolving landscape of Artificial Intelligence (AI) as a General Purpose Technology and its dual role in driving and sustaining innovation across various domains. It introduces the AI Dictionary, a novel tool designed to measure technological innovation using text-based methods.

The AI Dictionary is a comprehensive that describe the whole domain of AI. It serves both the research community and industry domains, aiding in the identification of radical innovations and uncovering applications of AI in new domains. We hope to establish a foundational methodology for a new and innovation measurement tool.

The AI Dictionary has been uploaded as a Dataset to HuggingFace and can be accessed [here](https://huggingface.co/datasets/J0nasW/AI-Dictionary).

The repository contains several Jupyter notebooks and Python scripts that detail the process of creating and validating the AI Dictionary. Here is a brief overview of each:

- `01_Data_Gathering.ipynb`: This notebook functions as a data acquisition tool, gathering data from various sources like PapersWithCode and CSO.

- `01a_Database_Generation_Helper.py` and `01b_Arxiv_Fulltext_Helper.py`: These scripts assist in generating helper files for the graph database and perform long function calls. It is advised to use `tmux` for execution.

- `02_Database_Generation.ipynb`: This notebook is used to gather all files from previous steps and generate importable CSVs for a neo4j graph database.

- `03_Core_Dictionary.ipynb`: This notebook can be executed once the graph database is set up. It generates the core AI Dictionary.

- `03a_Extended_Dictionary_Prep.py` and `03b_Extended_Dictionary_Visualizer.py`: These scripts assist in preparing and visualizing the extended AI Dictionary.

- `04_Extended_Dictionary.ipynb`: This notebook extends the core AI Dictionary with additional key phrases from paper abstracts taken from the graph database. Please execute the helper scripts before executing this notebook.

- `05_Validation.ipynb`: This notebook validates the AI Dictionary using regression analysis.

- `05a_Validation_Sample_Helper.py` and `05b_Log_Regression_UI.py`: These scripts assist in validating the AI Dictionary. Please execute `05a_Validation_Sample_Helper.py` to generate a validation sample before executing `05_Validation.ipynb`.

- `06_Data_Visualization.ipynb`: This notebook visualizes the data from the AI Dictionary.

- `07_Comparison.ipynb`: HIGHLY EXPERIMENTAL: This notebook compares the AI Dictionary to document corpora from various sources.

The repository also contains a `helper` folder that contains several helper scripts and files.