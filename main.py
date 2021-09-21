import TF_IDF

if __name__ == "__main__":

    t = TF_IDF.TF_IDF('wine.csv')

    ret = t.tfidf('tremendous', 5)
    for tup in ret:
        print(tup)
