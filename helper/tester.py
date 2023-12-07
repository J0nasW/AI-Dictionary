from wordtrie_builder import WordTrie
import pandas as pd

trie = WordTrie(word_filter=True, text_filter=True, show_progress_bar=True, weights=True)

word_list = [
    "apple",
    "apples",
    "app",
    "application",
    "apply",
    "applying",
    "appreciate",
    "appreciation",
    "appreciate your",
    "of apples.",
    "from text",
    "FCA",
    "fca",
    "(FCA)",
    "controlled vocabularies",
]
word_list_index = [i for i in range(len(word_list))]
# Make a word_list_weights with random weights between 0 and 1 for each word
import random
word_list_weights = [random.random() for i in range(len(word_list))]

core_keywords = pd.read_csv("./data/dictionaries/negative_keywords.csv")
core_keyword_list = core_keywords["keyword"].tolist()
core_keyword_index = [i for i in range(len(core_keyword_list))]
core_keywords_weights = [random.random() for i in range(len(core_keyword_list))]

test_text = "We introduce an end-to-end methodology (from text processing to querying a knowledge graph) for the sake of knowledge extraction from text corpora with a focus on a list of vocabularies of interest. We propose a pipeline that incorporates Natural Language Processing (NLP), Formal Concept Analysis (FCA), and Ontology Engineering techniques to build an ontology from textual data. We then extract the knowledge about controlled vocabularies by querying that knowledge graph, i.e., the engineered ontology. We demonstrate the significance of the proposed methodology by using it for knowledge extraction from a text corpus that consists of 800 news articles and reports about companies and products in the IT and pharmaceutical domain, where the focus is on a given list of 250 controlled vocabularies."

print(f"Adding {len(core_keyword_list)} keywords to trie...")
    
trie.add_bulk(core_keyword_list, core_keyword_index, core_keywords_weights)
# trie.add_bulk(word_list, word_list_index, word_list_weights)
    
print(f"Node Count: {trie.count_nodes()}")
print("")

results = trie.search(test_text, return_nodes=True)
print(f"Results: {results}")
# Calculate the mean weights of the keywords
words = trie.search(test_text, only_return_words=True)
print(f"Words found: {words}")
print(f"Weights found: {[result[2] for result in results if result is not None]}")
weight = [weight_indiv[2] for weight_indiv in results if weight_indiv is not None]
print(f"Manual calc of mean weight: {sum(weight) / len(weight)}")
print(f"Manual calc of sum weight: {sum(weight)}")
print("")
print(f"Automatic mean of weights: {trie.search_mean(test_text)}")
print(f"Automatic sum of weights: {trie.total_weight_of_found_words(test_text)}")
print("")
print(f"Absolute weight: {trie.search_absolute(test_text)}")
print(f"Ratio: {trie.search_ratio(test_text)}")
print("")
print(f"Aggregate search info: {trie.aggregate_search_info(test_text)}")
# trie.visualize()

