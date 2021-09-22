import csv
from decimal import Decimal
import math


class BM_25(object):

    def __init__(self, dataFile: str):
        """
        Need: 
            SUM(t in Q) of:  
            
            N: Number of documents in the corpus
            df_t: The number of documents containing term t
            k_1: Typically set to 1.2
            f_t: Frequency of that term in the document
            b: typically set to .75
            |d|: length of the document
            k_2: Typically between 0 and 1000 (much larger than k_1
            qf_t: the frequency of the term t in query Q. 
            
            
            1. Number of documents in the corpus
            2. Number of documents containing term t 
            3. k_1: Set to 1.2
            4. f_t: frequency of term in the document
            5. b: typically set to .75
            6. |d| length of the document
        
        """

        # Ingesting file.
        self.global_index: dict = {}
        self.document_tcount: dict = {}  # aka document length in words.
        self.n: int = 0
        self.k1: Decimal = Decimal(1.2)
        self.k2: Decimal = Decimal(500)
        self.b: Decimal = Decimal(.75)
        self.avg_d: Decimal = Decimal(0)  # Calculated after ingesting documents
        with open(dataFile, 'r') as csv_file:
            dict_reader: csv.DictReader = csv.DictReader(csv_file)
            # Iterating through documents
            for row in dict_reader:
                self.n += 1
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
        # Calculating average document length
        total_length: int = 0
        document_count: int = 0
        for key in self.document_tcount:
            document_count += 1
            total_length += self.document_tcount[key]
        self.avg_d = Decimal(total_length) / Decimal(document_count)

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

    def _first_algoterm(self, term: str) -> Decimal:
        """
            ln( (N - dft + .5)/(dft + .5) )
        """

        dft: int = 0
        if term in self.global_index:
            term_posting: list = self.global_index[term]
            dft = sum([1 for docpair in term_posting])
        fraction: Decimal = Decimal((self.n - dft + .5) / (dft + .5))
        return Decimal(math.log(fraction))

    def _second_algoterm(self, term: str, doc_id: int) -> Decimal:
        """
        ((k1 + 1) * ft) / ( k1*(1-b) + (b * (|d|/avg(|d|)) + ft

        :param term:
        :return:
        """
        # Finding the given term/document in the postings
        ft: Decimal = Decimal(0)
        docpairs: list = self.global_index[term]
        for pair in docpairs:
            if pair[0] == doc_id:
                ft = pair[1]
                break
        d: Decimal = Decimal(self.document_tcount[doc_id])
        numerator: Decimal = (1 + self.k1) * Decimal(ft)
        # Staging pieces of the denominator
        denom_1: Decimal = self.k1 * (1 - self.b)
        denom_2: Decimal = self.b * (d / self.avg_d)
        denom_3: Decimal = ft
        # Putting it all together
        final: Decimal = numerator / (denom_1 + denom_2 + denom_3)
        return final

    def _third_algoterm(self, term: str, query: str) -> Decimal:
        """ ((k2 + 1) * qft) / (k2 + qft)"""

        qterms: list = query.split(' ')
        qft: int = sum([1 for qterm in qterms if qterm == term])
        numerator: Decimal = (1 + self.k2) * Decimal(qft)
        demonimator: Decimal = self.k2 + Decimal(qft)
        return numerator / demonimator

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
        return set(relevant_ids)

    def _score(self, doc_id: int, query: str) -> Decimal:
        """ Runs the BM25 scoring algorithm against the given document """
        terms: list = query.split(' ')
        sum: Decimal = Decimal(0)
        # Performing summation
        for term in terms:
            if term not in self.global_index:
                continue
            first_term: Decimal = self._first_algoterm(term)
            second_term: Decimal = self._second_algoterm(term, doc_id)
            third_term: Decimal = self._third_algoterm(term, query)
            sum += (first_term * second_term * third_term)
        return sum

    def bm25(self, query: str, k: int):
        relevant_docids: set = self._relevant_docids(query)
        results: list = []
        for docid in relevant_docids:
            score = self._score(docid, query)
            if score > 0:
                results.append((docid, score))

        results.sort(key=lambda pair: pair[1])
        results.reverse()
        return results[:k]

