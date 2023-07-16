import torch
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import nltk
from nltk.corpus import stopwords

class BERTSummarizer:
    def __init__(self, pretrained_model_name='bert-base-uncased', num_clusters=3):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name)
        self.model.eval()
        self.num_clusters = num_clusters
        nltk.download('punkt')
        nltk.download('stopwords')

    def summarize(self, text, num_sentences=3):
        stop_words = stopwords.words('english')
        sentences = sent_tokenize(text)
        sentences = [' '.join(w for w in word_tokenize(sentence) if w not in stop_words) for sentence in sentences]
        sentence_vectors = self._get_sentence_vectors(sentences)

        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(sentence_vectors)

        avg = []
        for j in range(self.num_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_vectors)
        ordering = sorted(range(self.num_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering][:num_sentences])
        return summary

    def _get_sentence_vectors(self, sentences):
        vectors = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            vectors.append(embeddings.numpy().flatten())
        return np.array(vectors)