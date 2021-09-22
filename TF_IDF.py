import csv
import math
from decimal import Decimal


class TF_IDF(object):

    """ TF_IDF functionality as described in class. """
    def __init__(self, dataFile: str):
        # Ingesting file.
        self.global_index: dict = {}
        self.document_tcount: dict = {}
        with open(dataFile, 'r') as csv_file:
            dict_reader: csv.DictReader = csv.DictReader(csv_file)
            # Iterating through documents
            for row in dict_reader:
                temp_dict: dict = {}
                doc_id: str = row['id']
                terms: list = row['description'].split(' ')
                if '' in terms:
                    terms.remove('')
                # Tallying terms
                total_count: int = 0
                for term in terms:
                    if term in temp_dict:
                        temp_dict[term] += 1
                    else:
                        temp_dict[term] = 1
                    total_count += 1
                # Logging document info
                for key in temp_dict:
                    self._update_index(doc_id, key, temp_dict[key])
                self.document_tcount[doc_id] = total_count

    def _update_index(self,
                      doc_id: str,
                      term: str,
                      occurences: int):
        """ __init__ helper function. """
        if term in self.global_index:
            # Managing posting order
            posting_list: list = self.global_index[term]
            for i, docpair in enumerate(posting_list):
                if occurences >= docpair[1]:
                    self.global_index[term].insert(i, (doc_id, occurences))
                    break
                if docpair is self.global_index[term][-1]:
                    # New posting goes at the end.
                    self.global_index[term].append((doc_id, occurences))
        else:
            self.global_index[term] = [(doc_id, occurences)]

    def _relevant_docids(self, Q: str):
        """ tfidf helper function.
            Returns a list of document IDs containing one or more terms in Q
        """
        terms = Q.split(' ')
        relevant_ids: list = []
        for term in terms:
            if term in self.global_index:
                term_posts: list = self.global_index[term]
                relevant_ids += [docpair[0] for docpair in term_posts]
        return relevant_ids

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
            nt: int = len(self.global_index[term])
            if nt > 0:
                sum += (Decimal(tf) / Decimal(nt))
        return sum

    def tf(self, d, t) -> Decimal:
        """
            d: document id
            t: target term
            returns log(1 + (n(d,t)/n(d)))
        """

        if t not in self.global_index:
            return Decimal(0)
        # Setting n(d): number of terms in a document
        n_d: int = self.document_tcount[d]
        # Setting n(d, t): number of times term t occurs in document d
        ndt: int = 0
        docpairs: list = self.global_index[t]
        for pair in docpairs:
            if pair[0] == d:
                ndt = pair[1]
                break
        # Math now
        fraction_val: Decimal = Decimal(ndt) / Decimal(n_d)
        final: Decimal = Decimal(math.log((1 + fraction_val)))
        return final

    def tfidf(self, Q: str, k: int):
        """
            Q: query term
            k: number of results to list
        """
        results: list = []
        relevant_docids = self._relevant_docids(Q)
        for docid in relevant_docids:
            score: Decimal = self.relevance(docid, Q)
            if score > 0:
                results.append((docid, score))
        results.sort(key=lambda pair: pair[1])
        results.reverse()
        return results[:k]

