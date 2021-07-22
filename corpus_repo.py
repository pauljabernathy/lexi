import pickle


class CorpusRepo:

    def __init__(self):
        with open("fyj/fyj_lm_md.pkl", 'rb') as f:
            self.fyj_lm = pickle.load(f)

        with open('fyj/fyj_md.pkl', 'rb') as f:
            self.fyj = pickle.load(f)

        self.fyj_sentences = list(self.fyj.sents)
