import difflib

import nltk
import networkx as nx
from rouge import Rouge
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from termcolor import colored
import string
import re


# You may need to download 'punkt' and 'stopwords' from nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')

def similarity_matrix(sentences):
    # Create an empty similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentences[i].reshape(1, -1), sentences[j].reshape(1, -1))[0,0]
    return sim_mat

def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    # Convert sparse matrix to dense matrix
    vectors = vectorizer.toarray()
    return vectors

def text_rank(text, n):
    # Tokenize the text and remove stopwords
    sentences = sent_tokenize(text)
    sentences = [' '.join(w for w in word_tokenize(sentence) if w not in stop_words) for sentence in sentences]

    # Convert sentences to vectors
    vectors = vectorize_sentences(sentences)

    # Create a similarity matrix
    sim_mat = similarity_matrix(vectors)

    # Use the similarity matrix to create a graph
    nx_graph = nx.from_numpy_array(sim_mat)

    # Apply the PageRank algorithm to the graph
    scores = nx.pagerank(nx_graph)

    # Sort the sentences by score and return the n best
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    best_sentences = [s for score, s in ranked_sentences[:n]]

    return ' '.join(best_sentences)

def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data

# Specify the file path
file_path = "./dataset/business/001.txt"

# Read the text from the file
text = read_text_from_file(file_path)

# Generate the summary
summary = text_rank(text, 2)

print(summary)

# Specify the file path for saving the summary
output_path = "./res.txt"

# Save the summary to the file
with open(output_path, 'w') as file:
    file.write(summary)


def read_reference_summary(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


# Specify the file path
reference_summary_path = "./reference res.txt"

# Read the reference summary from the file
reference_summary = read_reference_summary(reference_summary_path)

def calculate_rouge_scores(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

# Calculate ROUGE scores
rouge_scores = calculate_rouge_scores(summary, reference_summary)
print(rouge_scores)


def normalize(s):
    # Lowercase, remove punctuation, and remove extra whitespaces
    return re.sub(' +', ' ', s.lower().translate(str.maketrans('', '', string.punctuation)).strip())

def highlight_matches(reference, summary, output_html):
    # Normalize the texts
    reference_norm = normalize(reference)
    summary_norm = normalize(summary)

    # Tokenize the texts
    reference_tokens = word_tokenize(reference_norm)
    summary_tokens = word_tokenize(summary_norm)

    # Use difflib to find common words
    matcher = difflib.SequenceMatcher(None, reference_tokens, summary_tokens)
    common_words = set([reference_tokens[match.a] for match in matcher.get_matching_blocks() if match.size > 0])

    # Create HTML for reference and summary
    def generate_html(raw_text, norm_text, highlight_words):
        html = ""
        for raw_word, norm_word in zip(word_tokenize(raw_text), word_tokenize(norm_text)):
            if norm_word in highlight_words:
                html += '<span style="background-color: #FFFF00">' + raw_word + '</span> '
            else:
                html += raw_word + ' '
        return html

    reference_html = generate_html(reference, reference_norm, common_words)
    summary_html = generate_html(summary, summary_norm, common_words)

    html = """
    <h1>Reference</h1>
    <p>{}</p>
    <h1>Summary</h1>
    <p>{}</p>
    """.format(reference_html, summary_html)

    # Save the HTML to a file
    with open(output_html, 'w') as f:
        f.write(html)

    print(f"HTML saved to {output_html}")

highlight_matches(reference_summary, summary, "highlight_matches.html")
