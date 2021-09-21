import csv
import math
from decimal import Decimal


class TF_IDF(object):

    def __init__(self, dataFile: str):
        # Ingesting file.
        with open(dataFile, 'r') as csv_file:
            dict_reader: csv.DictReader = csv.DictReader(csv_file)
            self.documents: dict = {}
            for row in dict_reader:
                word_occurrences: dict = {}
                doc_id: str = row['id']
                doc_terms: list = row['description'].split(' ')
                self._count_words(word_occurrences, doc_terms)
                self.documents[doc_id] = word_occurrences

    def _count_words(self, word_dict: dict, words: list):
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    def tfidf(self, Q, k):
        ...

    def relevance(self, d, Q):
        ...

    def tf(self, d, t) -> float:
        """
            d: document id
            t: target term
            returns log(1 + (n(d,t)/n(d)))
        """
        document_terms: dict = self.documents[d]
        # Calculating n(d)
        n_d: int = 0
        for key in document_terms:
            n_d += document_terms[key]
        # Setting n(d, t)
        ndt: int = 0
        if t in document_terms:
            ndt = document_terms[t]
        # fraction_val: Decimal = Decimal(ndt) / Decimal(n_d)
        fraction_val = ndt / n_d
        final: float = math.log((1 + fraction_val), 10)
        return final