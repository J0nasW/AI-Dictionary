# WordTrie Implementation (inspired by https://github.com/mhowison/wordtrie)
import json, re
from nltk.stem import WordNetLemmatizer

_RESERVED_KEY = '#'  # Reserved value key

def _ensure_valid_key(word):
    """ Ensure the reserved key is not misused. """
    return f"{_RESERVED_KEY}{word}" if word.startswith(_RESERVED_KEY) else word

def _split_if_string(words):
    """ Split words if it's a string. """
    return words.split() if isinstance(words, str) else words

def _replace_value(old, new):
    """ Replace old value with new value. """
    return new

def worker_initializer():
    global lemmatizer
    lemmatizer = WordNetLemmatizer()

def filter_string(text):
    try:
        # Convert to string and make it lower case
        text = str(text)
        
        # Lemmatize the text
        text = ' '.join([lemmatizer.lemmatize(word).lower() for word in text.split()])

        # Combine regular expressions:
        # - Keep only alphanumeric characters and spaces
        # - Filter out Russian, Chinese characters, and numbers
        pattern = r"[^a-zA-Z0-9 ]|[\u0400-\u04FF]+|[\u4E00-\u9FFF]+|\d+|-"
        text = re.sub(pattern, " ", text)

        # Strip leading and trailing whitespaces
        text = text.lower().strip()

        return text
    
    except:
        raise ValueError("The text cannot be processed in the filter step.")

class WordTrie:
    def __init__(self, weights=False, word_filter=False, text_filter=False, show_progress_bar=False):
        self.root = {}
        worker_initializer()
        self.word_filter = word_filter
        self.text_filter = text_filter
        self.show_progress_bar = show_progress_bar
        self.weights = weights
        if self.show_progress_bar:
            from tqdm import tqdm
            self.tqdm = tqdm

    def add(self, word, value, weight=None, aggregator=_replace_value):
        # Modify to accept weight
        if self.weights and weight is None:
            raise ValueError("Weight is required when weights are enabled.")
        if self.word_filter:
            word = filter_string(word)
        node = self.root
        for word in map(_ensure_valid_key, _split_if_string(word)):
            node = node.setdefault(word, {})
        node_data = {'value': aggregator(node.get(_RESERVED_KEY, {}).get('value'), value)}
        if self.weights:
            node_data['weight'] = weight
        node[_RESERVED_KEY] = node_data
        
    def add_bulk(self, words_list, value_list, weight_list=None):
        # Add a list of words to the trie, with an optional list of values.
        if len(words_list) != len(value_list):
            raise ValueError("The length of words_list and value_list must be the same.")
        if not all(isinstance(word, str) for word in words_list):
            raise ValueError("The words_list must contain only strings.")
        # print(f"Weights: {self.weights}")
        # print(f"Length of words_list: {len(words_list)}")
        # print(f"Length of value_list: {len(value_list)}")
        # print(f"Length of weight_list: {len(weight_list)}")
        if self.weights == True:
            print("Weights are enabled.")
            if weight_list is None or len(words_list) != len(weight_list):
                raise ValueError("Weight list is required and must be the same length as words_list when weights are enabled.")
            for words, value, weight in self.tqdm(zip(words_list, value_list, weight_list), total=len(words_list)):
                self.add(words, value, weight)
        else:
            print("Weights are disabled.")
            for words, value in self.tqdm(zip(words_list, value_list), total=len(words_list)):
                self.add(words, value)
            
    def length(self):
        # Return the number of nodes in the trie.
        return self.count_nodes()

    def match(self, words):
        # Return the value of the node that matches the longest prefix of words.
        node = self.root
        for word in map(_ensure_valid_key, _split_if_string(words)):
            if word not in node:
                return None
            node = node[word]
        return node.get(_RESERVED_KEY)

    # def search(self, text, return_nodes=False):
    #     # Return a list of values that match the words.
    #     if self.text_filter:
    #         filter_string(text)   
    #     node, match, values = self.root, [], []
    #     for word in map(_ensure_valid_key, _split_if_string(text)):
    #         if word in node:
    #             node = node[word]
    #             match.append(word)
    #         else:
    #             self._process_match(node, match, values, return_nodes)
    #             node, match = self.root.get(word, self.root), [word] if word in self.root else []

    #     self._process_match(node, match, values, return_nodes)
    #     return values
    
    def search(self, text, return_nodes=False, only_return_words=False):
        # Return a list of values that match the words or just the words if only_return_words is True.
        if self.text_filter:
            text = filter_string(text)

        node, match, values, found_words = self.root, [], [], []
        for word in map(_ensure_valid_key, _split_if_string(text)):
            if word in node:
                node = node[word]
                match.append(word)
            else:
                if only_return_words:
                    if match and _RESERVED_KEY in node:
                        found_word = ' '.join(match)
                        found_words.append(found_word)
                else:
                    self._process_match(node, match, values, return_nodes)
                node, match = self.root.get(word, self.root), [word] if word in self.root else []

        if only_return_words:
            if match and _RESERVED_KEY in node:
                found_word = ''.join(match)
                found_words.append(found_word)
            return found_words
        else:
            self._process_match(node, match, values, return_nodes)
            return values
    
    def search_absolute(self, text):
        # Returns the absolute count of matches
        cleaned_text = filter_string(text) if self.text_filter else text
        count = 0
        node, match = self.root, []
        for word in map(_ensure_valid_key, _split_if_string(cleaned_text)):
            if word in node:
                node = node[word]
                match.append(word)
            else:
                if self._is_match(node, match):
                    count += 1
                node, match = self.root.get(word, self.root), [word] if word in self.root else []

        if self._is_match(node, match):
            count += 1

        return count
    
    def search_mean(self, text):
        # Returns the mean weight of matches
        if not self.weights:
            raise ValueError("Mean weight calculation is only available when weights are enabled.")
        cleaned_text = filter_string(text) if self.text_filter else text
        total_weight, count = 0, 0
        node, match = self.root, []
        for word in map(_ensure_valid_key, _split_if_string(cleaned_text)):
            if word in node:
                node = node[word]
                match.append(word)
            else:
                weight = self._process_match_for_weight(node, match)
                if weight is not None:
                    total_weight += weight
                    count += 1
                node, match = self.root.get(word, self.root), [word] if word in self.root else []

        weight = self._process_match_for_weight(node, match)
        if weight is not None:
            total_weight += weight
            count += 1

        return total_weight / count if count > 0 else 0
    
    def total_weight_of_found_words(self, text):
        if not self.weights:
            raise ValueError("Total weight calculation is only available when weights are enabled.")

        total_weight = 0
        node, match = self.root, []
        for word in map(_ensure_valid_key, _split_if_string(text)):
            if word in node:
                node = node[word]
                match.append(word)
            else:
                weight = self._process_match_for_weight(node, match)
                if weight is not None:
                    total_weight += weight
                node, match = self.root.get(word, self.root), [word] if word in self.root else []

        weight = self._process_match_for_weight(node, match)
        if weight is not None:
            total_weight += weight

        return total_weight
    
    def search_ratio(self, text):
        # Returns the ratio of matches to total words
        cleaned_text = filter_string(text) if self.text_filter else text
        words = _split_if_string(cleaned_text)

        # Count total words and matches
        total_words = len(words)
        match_count = 0
        for word in words:
            match_count += 1 if self.match(word) is not None else 0

        # Calculate and return the ratio
        return match_count / total_words if total_words > 0 else 0
    
    def aggregate_search_info(self, text):
        '''
        Returns a list of the following:
        - Absolute count of matches
        - Ratio of matches to total words
        - (Optional) Mean weight of matches
        - (Optional) Total weight of matches
        '''
        cleaned_text = filter_string(text) if self.text_filter else text
        words = _split_if_string(cleaned_text)

        # Initialize counters and accumulators
        absolute_count = 0
        total_words = len(words)
        total_weight = 0
        weight_count = 0

        for word in words:
            node, match = self.root, []
            for char in map(_ensure_valid_key, _split_if_string(word)):
                if char in node:
                    node = node[char]
                    match.append(char)
                else:
                    break

            if match and _RESERVED_KEY in node:
                absolute_count += 1
                if self.weights:
                    weight = node[_RESERVED_KEY].get('weight')
                    if weight is not None:
                        total_weight += weight
                        weight_count += 1

        result = [absolute_count, absolute_count / total_words if total_words > 0 else 0]
        
        if self.weights:
            mean_weight = total_weight / weight_count if weight_count > 0 else 0
            result.extend([mean_weight, total_weight])

        return result
        

    def _process_match(self, node, match, values, return_nodes):
        # Process a match. If return_nodes is True, return the node and the match.
        if match and _RESERVED_KEY in node:
            match_data = node[_RESERVED_KEY]
            result = (match, match_data['value'])
            if self.weights:
                result += (match_data['weight'],)
            values.append(result if return_nodes else match_data['value'])
            
    def _process_match_for_weight(self, node, match):
        # Helper function to process a match and return its weight
        if match and _RESERVED_KEY in node:
            match_data = node[_RESERVED_KEY]
            return match_data.get('weight')
        return None
    
    def _is_match(self, node, match):
        # Helper function to check if a match is valid
        return match and _RESERVED_KEY in node

    def to_json(self, filename, indent=2):
        # Save the trie to a JSON file. 
        with open(filename, "w") as f:
            json.dump(self.root, f, indent=indent)

    def from_json(self, filename):
        with open(filename) as f:
            self.root = json.load(f)
            
    def count_nodes(self, node=None):
        """ Count the number of nodes in the trie. """
        if node is None:
            node = self.root

        count = 1  # Count current node
        for child in node:
            if child != _RESERVED_KEY and isinstance(node[child], dict):
                count += self.count_nodes(node[child])
        return count
            
    def visualize(self, node=None, indent="", last=True):
        """ Visualize the trie structure. """
        if node is None:
            node = self.root

        if isinstance(node, dict):
            children = list(node.keys())
            for i, child in enumerate(children):
                prefix = "└── " if last else "├── "
                print(f"{indent}{prefix}{child}" if child != _RESERVED_KEY else f"{indent}{prefix}[VALUE: {node[child]}]")

                # Only call visualize on child if it is a dictionary
                if isinstance(node[child], dict):
                    if i < len(children) - 1:
                        self.visualize(node[child], indent + "│   ", False)
                    else:
                        self.visualize(node[child], indent + "    ", True)