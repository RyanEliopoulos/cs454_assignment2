import TF_IDF

if __name__ == "__main__":
    t = TF_IDF.TF_IDF('wine.csv')

    print(t.tf('0', 'tremendous'))