import csv
import math
from decimal import Decimal


class TF_IDF(object):

    def __init__(self, dataFile: str):
        # Ingesting file.
        with open(dataFile, 'r') as csv_file:
            dict_reader: csv.DictReader = csv.DictReader(csv_file)
            self.documents: dict = {}  # dict of dicts
            for row in dict_reader:
                word_occurrences: dict = {}
                doc_id: str = row['id']
                doc_terms: list = row['description'].split(' ')
                self._count_words(word_occurrences, doc_terms)
                self.documents[doc_id] = word_occurrences
                if '' in word_occurrences:
                    word_occurrences.pop('')

    def _count_words(self, word_dict: dict, words: list):
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    def _contained_by(self, term: str) -> int:
        """ Counts documents which contain the given term """
        count: int = 0
        for key in self.documents:
            if term in self.documents[key]:
                count += 1
        return count

    def relevance(self, d: str, Q: str) -> Decimal:
        """
        Sum of TF(d, t)/n(t) across all terms in the query

        n(t) = number of documents containing t

        :param d:  document id
        :param Q:  query string
        """
        sum: Decimal = Decimal(0)
        terms: list = Q.split(' ')
        for term in terms:
            tf = self.tf(d, term)
            nt: int = self._contained_by(term)
            if nt > 0:
                sum += (Decimal(tf) / Decimal(nt))
        return sum

    def tf(self, d, t) -> Decimal:
        """
            d: document id
            t: target term
            returns log(1 + (n(d,t)/n(d)))
        """
        document_terms: dict = self.documents[d]
        # Calculating n(d): number of terms in a document
        n_d: int = 0
        for key in document_terms:
            n_d += document_terms[key]
        # Setting n(d, t): number of times term t occurs in document d
        ndt: int = 0
        if t in document_terms:
            ndt = document_terms[t]
        # fraction_val: Decimal = Decimal(ndt) / Decimal(n_d)
        fraction_val: Decimal = Decimal(ndt) / Decimal(n_d)
        final: Decimal = Decimal(math.log((1 + fraction_val)))
        return final

    def tfidf(self, Q: str, k: int):
        """
            Q: query term
            k: number of results to list
        """
        results: list = []
        for key in self.documents:
            score: Decimal = self.relevance(key, Q)
            if score > 0:
                results.append((key, score))
        results.sort(key=lambda pair: pair[1])
        results.reverse()
        return results[:5]



