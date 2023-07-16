import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords

class TextRankClassifier:

    def __init__(self):
        pass

    def text_rank(self, file_path, n):
        nltk.download('punkt')
        nltk.download('stopwords')

        stop_words = stopwords.words('english')

        # Read the text from the file
        text = self._read_text_from_file(file_path)

        # Tokenize the text and remove stopwords
        sentences = sent_tokenize(text)
        sentences = [' '.join(w for w in word_tokenize(sentence) if w not in stop_words) for sentence in sentences]

        # Convert sentences to vectors
        vectors = self._vectorize_sentences(sentences)

        # Create a similarity matrix
        sim_mat = self._similarity_matrix(vectors)

        # Use the similarity matrix to create a graph
        nx_graph = nx.from_numpy_array(sim_mat)

        # Apply the PageRank algorithm to the graph
        scores = nx.pagerank(nx_graph)

        # Sort the sentences by score and return the n best
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        best_sentences = [s for score, s in ranked_sentences[:n]]

        return ' '.join(best_sentences)

    def _vectorize_sentences(self, sentences):
        vectorizer = TfidfVectorizer().fit_transform(sentences)
        # Convert sparse matrix to dense matrix
        vectors = vectorizer.toarray()
        return vectors

    def _similarity_matrix(self, sentences):
        # Create an empty similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentences[i].reshape(1, -1), sentences[j].reshape(1, -1))[0, 0]
        return sim_mat

    def _read_text_from_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read().replace('\n', '')
        return data
