import unittest
import vector_utils as vu
import spacy
import time
import pandas as pd
import numpy as np
import pickle
import constants


class CosineSimilarityTest(unittest.TestCase):

    def test_cosine_similarity(self):
        self.assertEqual(1, vu.cosine_similarity((1, 1), (2, 2)))


class WordVectorsTest(unittest.TestCase):

    def setUp(self):
        print("->setup")
        self.en = spacy.load('en_core_web_md')
        vocab = list(self.en.vocab.strings)
        self.en_words = set(v.lower() for v in vocab)

        self.es = spacy.load('es_core_news_md')
        vocab = list(self.es.vocab.strings)
        self.es_words = set(v.lower() for v in vocab)
        print("<-setup")

    @unittest.skip("time")
    def test_closest_word_vector_en(self):
        print("->test_closest_word_vector_en")
        #df1 = vu.find_closest_word_vectors(self.en.vocab['afghanistan'].vector, self.en_words, self.en.vocab)#.head(37)
        #df2 = vu.find_closest_word_vectors(self.en.vocab['tiger'].vector, self.en_words, self.en.vocab)#.head(37)
        #df3 = vu.find_closest_word_vectors(self.en.vocab['cat'].vector, self.en_words, self.en.vocab)#.head(37)
        '''
        df1 = vu.find_closest_word_vectors("i'll", self.en_words, self.en)
        print(df1)
        df1 = vu.find_closest_word_vectors('afghanistan', self.en_words, self.en)#.head(37)
        df2 = vu.find_closest_word_vectors('tiger', self.en_words, self.en)#.head(37)
        df3 = vu.find_closest_word_vectors('cat', self.en_words, self.en)#.head(37)
        print(df1.head(37))
        print(df2.head(37))
        print(df3.head(37))
        '''
        pass

    #def test_closest_word_vector_en(self):
        n = 25
        df1 = vu.find_closest_word_vectors("i'll", self.en_words, self.en)
        print(df1.head(n))
        df2 = vu.find_closest_word_vectors("you'll", self.en_words, self.en)
        print(df2.head(n))
        df3 = vu.find_closest_word_vectors("we'll", self.en_words, self.en)
        print(df3.head(n))
        df4 = vu.find_closest_word_vectors("aren't", self.en_words, self.en)
        print(df4.head(n))
        df5 = vu.find_closest_word_vectors("i'd", self.en_words, self.en)
        print(df5.head(n))
        df6 = vu.find_closest_word_vectors("i'm", self.en_words, self.en)
        print(df6.head(n))
        print("<-test_closest_word_vector_en")

    @unittest.skip("")
    def test_closest_word_vector_es(self):
        print("\n\n----")
        #df1 = vu.find_closest_word_vectors(self.en.vocab['afghanistan'].vector, self.es_words, self.es.vocab)#.head(37)
        #df2 = vu.find_closest_word_vectors(self.en.vocab['tigre'].vector, self.es_words, self.es.vocab)#.head(37)
        #df3 = vu.find_closest_word_vectors(self.en.vocab['gato'].vector, self.es_words, self.es.vocab)#.head(37)

        df1 = vu.find_closest_word_vectors('afghanistan', self.es_words, self.es)#.head(37)
        print(df1.head(37))
        #print(df2.head(37))
        #print(df3.head(37))

    def test_closest_word_vector_series(self):
        print("\n\n----")
        df1 = vu.find_closest_word_vectors('afghanistan', self.en_words, self.en)#.head(37)

        df2 = vu.find_closest_word_vectors_series('afghanistan', self.en_words, self.en.vocab)
        print(df1.head())
        print(df2.head())
        #self.assertEqual(df1, df2)
        self.assertTrue(df1.equals(df2))

    def test_find_closest_word_vector_matrix(self):
        '''with open('../word_stats_pkls/matrix_13686.pkl', 'rb') as f:
            matrix = pickle.load(f)
        with open('../word_stats_pkls/word_docs_top_13686.pkl', 'rb') as f:
            word_docs = pickle.load(f)
        words_list = [w.text for w in word_docs]
        matrix_df = pd.DataFrame(data=matrix, index=words_list, columns=words_list)'''
        source = "news"
        with open(f"../word_stats_pkls/{source}_matrix_df.pkl", 'rb') as f:
            matrix_df = pickle.load(f)
        result1 = vu.find_closest_word_vectors_from_matrix('afghanistan', matrix_df)
        print(result1.iloc[:17])
        #df1 = vu.find_closest_word_vectors('afghanistan', self.en_words, self.en.vocab)
        #print(df1.head(17))
        # The results from the two find_closest_word_vectors functions are different because
        # find_closest_word_vectors() goes through all the words in the Spacy vocab,
        # while find_closest_word_vectors_from_matrix() only uses whatever is in the df it loads, which currently is
        # only 13686 words, the ones that account for the top 95% of words used in the data set.
        # self.assertEqual(df1[constants.SIMILARITY][:100], result1.values[:100])
        # self.assertEqual(df1[constants.WORD][:100], result1.index[:100])

        # Look for a word that is not there.
        empty_result = vu.find_closest_word_vectors_from_matrix("weasel", matrix_df)
        self.assertTrue(([constants.WORD, constants.POS, constants.SIMILARITY] == empty_result.columns).all())
        self.assertEqual((0, 3), empty_result.shape)

        word_list = ['afghanistan', 'tiger', 'car', 'chef', 'skinny', 'weasel', 'wood', 'mechanic']
        start = time.time()
        for word in word_list:
            vu.find_closest_word_vectors_from_matrix(word, matrix_df)
        stop = time.time()
        print(stop - start)

    @unittest.skip("")
    def test_find_closest_word_vector_matrix_perf(self):
        with open("../word_stats_pkls/twitter_matrix_13686_df.pkl", 'rb') as f:
            matrix_df = pickle.load(f)
        with open('../word_stats_pkls/word_docs_top_13686.pkl', 'rb') as f:
            word_docs = pickle.load(f)
        words_list = [w.text for w in word_docs]
        start = time.time()
        for word in words_list:
            vu.find_closest_word_vectors_from_matrix(word, matrix_df)
        stop = time.time()
        print(f"{stop - start} seconds")

    @unittest.skip("move to a perf test file")
    def test_perf_comparison_word_vectors(self):
        word_list = ['afghanistan', 'tiger', 'car', 'chef', 'skinny', 'weasel', 'wood', 'mechanic']
        word_list = ['afghanistan', 'tiger', 'car']
        with open("../word_stats_pkls/twitter_matrix_13686_df.pkl", 'rb') as f:
            matrix_df = pickle.load(f)
        for_times = []
        apply_times = []
        matrix_times = []
        for word in word_list:
            for_start = time.time()
            vu.find_closest_word_vectors(word, self.en_words, self.en)
            for_stop = time.time()
            for_time = for_stop - for_start
            for_times.append(for_time)

            apply_start = time.time()
            vu.find_closest_word_vectors_series(word, self.en_words, self.en.vocab)
            apply_stop = time.time()
            apply_time = apply_stop - apply_start
            apply_times.append(apply_time)

            matrix_start = time.time()
            vu.find_closest_word_vectors_from_matrix(word, matrix_df)
            matrix_stop = time.time()
            matrix_time = matrix_stop - matrix_start
            matrix_times.append(matrix_time)

            print(f"word: {word};  \nfor: {for_time};  apply: {apply_time};  matrix: {matrix_time}")

        print(for_times)
        print(apply_times)
        print(matrix_times)
        import seaborn as sns
        sns.lineplot(for_times)
        sns.lineplot(apply_times)
        sns.lineplot(matrix_times)
        import matplotlib.pyplot as plt
        plt.show()

    def test_make_word_similarity_matrix(self):
        stats = pd.read_csv("../en_US_twitter_stats.csv")
        stats.columns = ['word', 'count', 'fraction', 'cum_sum', 'cum_frac']
        with open("../word_stats_pkls/twitter_word_docs_top_13686.pkl", 'rb') as f:
            spacy_words = pickle.load(f)
        start = time.time()
        n = 100
        matrix = vu.make_word_similarity_matrix(stats.word, self.en, n, spacy_words)
        end = time.time()
        print("created the matrix in", (end - start), " time")
        #print(matrix)
        #with open("z3.pkl", "wb") as f:
        #    pickle.dump(matrix, f)

    @unittest.skip('only make this matrix ad hoc')
    def test_actually_make_word_similarity_matrix(self):
        # stats = pd.read_csv("../en_US_twitter_stats.csv")
        stats = pd.read_csv("../en_US_news_stats.csv")
        stats.columns = ['word', 'count', 'fraction', 'cum_sum', 'cum_frac']
        start = time.time()
        n = 13686  # 13686 -> 95% of all words usage
        n = min(stats[stats.cum_frac > .95].index)
        matrix = vu.make_word_similarity_matrix(stats.word, self.en, n)
        end = time.time()
        print("created the matrix in", (end - start), " time")
        #print(matrix)
        file_name = f"../word_stats_pkls/news_matrix_{n}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(matrix, f)

    def test_make_word_similarity_df_from_matrix(self):
        matrix = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        word_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        word_list = ['cat', 'dog', 'otter', 'give', 'take', 'run', 'hot', 'cold', 'unknown']
        start = time.time()
        df = vu.make_word_similarity_df_from_matrix(matrix, word_list, self.en)
        end = time.time()
        print("created the df in", (end - start), " time")
        self.assertEqual((9, 10), df.shape)
        self.assertTrue((df.pos.values == ['NOUN', 'NOUN', 'NOUN', 'VERB', 'VERB', 'VERB', 'ADJ', 'ADJ', 'ADJ']).all())

    @unittest.skip('only make this matrix ad hoc')
    def test_actually_make_word_similarity_df(self):
        matrix_file_fp = "../word_stats_pkls/twitter_matrix_13686.pkl"
        matrix_file_fp = "../word_stats_pkls/news_matrix_17355.pkl"
        with open(matrix_file_fp, 'rb') as f:
            matrix = pickle.load(f)

        word_docs_file_fp = "../word_stats_pkls/news_top_17355.pkl"
        with open(word_docs_file_fp, "rb") as f:
            word_list = pickle.load(f)
        # word_list = [w.text for w in word_list]

        df = vu.make_word_similarity_df_from_matrix(matrix, word_list, self.en)
        self.assertTrue("pos" in list(df.columns))
        df_file_fp = "../word_stats_pkls/twitter_matrix_df.pkl"
        df_file_fp = "../word_stats_pkls/news_matrix_df.pkl"
        with open(df_file_fp, "wb") as f:
            pickle.dump(df, f)

    def initialize_matrix_df(self):
        if not hasattr(self, "twitter_matrix_df") or not self.matrix_df:
            with open("../word_stats_pkls/twitter_matrix_13686_df.pkl", 'rb') as f:
                self.matrix_df = pickle.load(f)


if __name__ == '__main__':
    unittest.main()
