import pickle


class CorpusRepo:

    def __init__(self):
        with open("fyj/fyj_lm_md.pkl", 'rb') as f:
            self.fyj_lm = pickle.load(f)

        with open('fyj/fyj_sentences_md.pkl', 'rb') as f:
            self.fyj_sentences = pickle.load(f)