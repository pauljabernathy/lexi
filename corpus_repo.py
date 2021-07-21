import pickle

class CorpusRepo:

    def __init__(self):
        with open("fyj/fyj_lemma_map.pkl", 'rb') as f:
            self.fyj_lm = pickle.load(f)

        with open('fyj/fyj.pkl', 'rb') as f:
            self.fyj = pickle.load(f)

        self.fyj_sentences = list(self.fyj.sents)
