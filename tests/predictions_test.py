import unittest
import pandas as pd
import time
import basic_operations as bo
import prediction as prd
import constants
import spacy
import pickle
from functools import partial
import timeit

PKLS_DIR = "../word_stats_pkls/"


class PerfTest(unittest.TestCase):

    def test_list_iteration_vs_series_apply(self):
        file_name = '../../../courses/data_science_capstone/en_US/moby_dick_no_header.txt'
        file_name = '../../../courses/data_science_capstone/en_US/twitter_train.txt'

        start = time.time()
        with open(file_name, 'r', encoding='UTF-8') as f:
            file_text = f.read()
        #word_stats_df = bo.find_word_stats(file_text)
        sentences = bo.tokenize_by_sentence(file_text)
        #sentence_lengths = bo.find_sentence_lengths_hist(sentences)
        two_grams = bo.find_n_grams_list_of_lists(sentences, 2)
        three_grams = bo.find_n_grams_list_of_lists(sentences, 3)
        #end = time.time()

        two_grams_pd = pd.Series(two_grams)
        three_grams_pd = pd.Series(three_grams)
        two_target = ['im', 'happy']
        three_target = ['in', 'the', 'fridge']

        # print(len(two_grams))
        # print(len(three_grams))
        # print(len(two_grams_pd))
        # print(len(three_grams_pd))

        start_list = time.time()
        for item in two_grams:
            self.do_something(item, two_target)
        end_list = time.time()

        start_series = time.time()
        two_grams_pd.apply(lambda item: self.do_something(item, two_target))
        end_series = time.time()
        print("list:", end_list - start_list)
        print("series:", end_series - start_series)

        f = partial(self.do_something, target=three_target)
        start_partial = time.time()
        three_grams_pd.apply(f)
        end_partial = time.time()
        print("partial:", end_partial - start_partial)

        start_list = time.time()
        for item in three_grams:
            self.do_something(item, three_target)
        end_list = time.time()

        start_series = time.time()
        three_grams_pd.apply(lambda item: self.do_something(item, three_target))
        end_series = time.time()
        print("list:", end_list - start_list)
        print("series:", end_series - start_series)

        f = partial(self.do_something, target=three_target)
        start_partial = time.time()
        three_grams_pd.apply(f)
        end_partial = time.time()

        print("partial:", end_partial - start_partial)

    def do_something(self, item, target):
        #print(item)
        if item == target:
            #print('found it')
            return True
        else:
            return False

    def test_value_counts_list_vs_string(self):
        file_name = '../../../courses/data_science_capstone/en_US/moby_dick_no_header.txt'
        file_name = '../../../courses/data_science_capstone/en_US/twitter_test_3.txt'

        start = time.time()
        print(start)
        with open(file_name, 'r', encoding='UTF-8') as f:
            file_text = f.read()
        # word_stats_df = find_word_stats(file_text)
        sentences = bo.tokenize_by_sentence(file_text)
        # sentence_lengths = find_sentence_lengths_hist(sentences)
        two_grams = bo.find_n_grams_list_of_lists(sentences, 2)
        two_grams_series = pd.Series(two_grams)
        j = two_grams_series.apply(",".join)

        before_string = time.time()
        j.value_counts()
        after_string = time.time()
        string_time = after_string - before_string
        print("string:", string_time)

        before_list = time.time()
        two_grams_hist = two_grams_series.value_counts()
        after_list = time.time()
        list_time = after_list - before_list
        print("list:", list_time)

        '''
        example output
        1634895587.5742245
        string: 0.018950939178466797
        list: 63.788137435913086
        => I think it is safe to conclude that performance is faster when you are doing value_counts on strings rather
        than lists, even when two two things contain the same information.
        '''

    def load_n_grams(self):
        if not hasattr(self, "four_grams_hist") or self.four_grams_hist is None:
            self.four_grams_hist = pd.read_csv("../en_US.twitter.txt_4_grams.csv")
        if not hasattr(self, "five_grams_hist") or self.five_grams_hist is None:
            #self.five_grams_hist = pd.read_csv("../en_US.twitter.txt_5_grams.csv")
            pass
        if not hasattr(self, "six_grams_hist") or self.six_grams_hist is None:
            pass
            #self.six_grams_hist = pd.read_csv("../en_US.twitter.txt_6_grams.csv")

    @unittest.skip("move to a perf file file")
    def test_n_gram_matches_times(self):
        self.load_n_grams()
        # list_of_hists = [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist]
        result = []
        tokens = ['date', 'at', 'the']

        result1 = prd.match_n_grams_one_hist_original(tokens, self.four_grams_hist)
        result2 = prd.match_n_grams_one_hist_2(tokens, self.four_grams_hist)
        #result3 = prd.match_n_grams_one_hist_3(tokens, self.four_grams_hist)
        #result4 = prd.match_n_grams_one_hist_4(tokens, self.four_grams_hist)

        # indices = [4915655, 5704679, 9227004, 9479448, 9992499, 11504314, 13198614, 13295528, 14050567, 15405597]
        indices = [2096383, 3646317, 3687529, 6987272, 7082085, 7869870, 11451942, 12106969, 12469418, 15648995]

        indices_result = prd.match_n_grams_by_index(indices, self.four_grams_hist)
        self.assertTrue(result2.equals(indices_result))
        self.assertTrue(result1.equals(indices_result))

        n = 10
        # tokens = ['date', 'at', 'the']
        print("original")
        start_original = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_original(tokens, self.four_grams_hist)
        end_original = time.time()
        print("time of original function", end_original - start_original)

        start_with_index = time.time()
        for i in range(n):
            prd.match_n_grams_by_index(indices, self.four_grams_hist)
        end_with_index = time.time()
        print("using the index", end_with_index - start_with_index)

        print("match_n_grams_one_hist_2")
        start_2 = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_2(tokens, self.four_grams_hist)
        end_2 = time.time()
        print("time 2", end_2 - start_2)

        print("match_n_grams_one_hist_2_2")
        start_2_2 = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_2_2(tokens, self.four_grams_hist)
            pass
        end_2_2 = time.time()
        print("time 2_2", end_2_2 - start_2_2)

        print("match_n_grams_one_hist_2_3")
        start_2_3 = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_2_3(tokens, self.four_grams_hist)
            pass
        end_2_3 = time.time()
        print("time 2_3", end_2_3 - start_2_3)

        print("match_n_grams_one_hist_3")
        start_3 = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_3(tokens, self.four_grams_hist)
        end_3 = time.time()
        print("time 3", end_3 - start_3)

        print("match_n_grams_one_hist_4")
        start_4 = time.time()
        for i in range(n):
            prd.match_n_grams_one_hist_4(tokens, self.four_grams_hist)
        end_4 = time.time()
        print("time 4", end_4 - start_4)


        """
        original
        time of original function 58.02697563171387
        using the index 0.0019838809967041016
        match_n_grams_one_hist_2
        time 2 50.94853067398071
        match_n_grams_one_hist_2_2
        time 2_2 50.930670976638794
        match_n_grams_one_hist_2_3
        time 2_3 74.7573893070221
        match_n_grams_one_hist_3
        time 3 52.255473136901855
        match_n_grams_one_hist_4
        time 4 56.84636116027832
        """
        # => The functions that use np.vectorize() are not must faster than the original.  A little, yes,
        # but still not ware we are looking for.  In fact, 3 was slower, at least in this run.
        # => Using the index is much faster by orders of magnitude.


    # The below tests are for testing the performance of a map vs a data frame.

    def get_from_map(self, map, search_term):
        return map[search_term]

    def get_from_df(self, df, search_term):
        return df.loc[search_term]

    def test_access_df_and_map(self):
        with open(PKLS_DIR + "four_grams_prefix_df.pkl", "rb") as f:
            prefix_df = pickle.load(f)

        with open(PKLS_DIR + "four_grams_prefix_map_twitter.pkl", "rb") as f:
            prefix_map = pickle.load(f)

        map_partial = partial(self.get_from_map, map=prefix_map, search_term="thanks for the")
        df_partial = partial(self.get_from_df, df=prefix_df, search_term="thanks for the")
        n = 100
        print("about the try the map access")
        map_results = timeit.timeit(map_partial, number=n)
        print(map_results)

        print("about the try the data frame access")
        df_results = timeit.timeit(df_partial, number=n)
        print(df_results)

        """
        about the try the map access
        4.619999999988522e-05
        about the try the data frame access
        5.462824399999995
        I guess this means that it is faster to get stuff from a map than a data frame.
        """

    unittest.skip("")
    def test_performance_collect_word_vectors_with_ngram_index(self):
        """
        tests whether the function that matches ngrams using the index, match_n_grams_by_index(), gives the same
        result as the original function to collect ngrams
        :return:
        """
        self.load_n_grams()
        source = "twitter"
        with open(PKLS_DIR + f"four_grams_prefix_map_{source}.pkl", "rb") as f:
            prefix_map = pickle.load(f)

        text = "thanks for the"
        indices = prefix_map[text]
        index_partial = partial(prd.match_n_grams_by_index, indices=indices, n_grams_hist=self.four_grams_hist)
        baseline_partial = partial(prd.match_n_grams_one_hist_original, tokens=text.split(" "),
                                   ngrams_hist=self.four_grams_hist)

        n = 10
        print("about to time original")
        baseline_result = timeit.timeit(baseline_partial, number=n)
        print("about to time with indices")
        result_from_indices = timeit.timeit(index_partial, number=n)

        print("original", baseline_result)
        print("indices", result_from_indices)
        """
        Process finished with exit code 0
        about to time original
        about to time with indices
        original 61.5503849
        indices 0.005324700000002736
        => I think I can conclude it is faster to use the indices stored somewhere previously
        """

        # From the above two tests, concluding that it is faster to get stuff from a map that from a data frame,
        # even by index.  So we if are using an index for getting stuff from the data frame, it is better to store
        # than in a map than in some other data frame.


class PredictFromNGramsTest(unittest.TestCase):

    def setUp(self) -> None:
        file_name = '../../../courses/data_science_capstone/en_US/twitter_test_3.txt'

        # start = time.time()
        # print(start)
        with open(file_name, 'r', encoding='UTF-8') as f:
            file_text = f.read()
        # word_stats_df = find_word_stats(file_text)
        self.sentences = bo.tokenize_by_sentence(file_text)
        # sentence_lengths = find_sentence_lengths_hist(sentences)
        #two_grams = bo.find_n_grams_list_of_lists(sentences, 2)
        #two_grams_series = pd.Series(two_grams)
        # self.load_n_grams()
        self.en = spacy.load("en_core_web_md")
        self.vocab = list(self.en.vocab.strings)
        self.en_words = set(v.lower() for v in self.vocab)

    def load_test_sentences(self):
        sentences = [
            "one two three",
            "one two three four",
            "one two three four five",
            "one two three four five six",
            "thanks for the coke",
            "thanks for the beer",
            "say thanks for the beer",
            "no thanks for the beer",
            "say no thanks for the beer",
            "really no thanks for the beer",
            "i say thanks for the beer",
            "please say thanks for the beer",
            "definitely say thanks for the beer",
            "thanks for the beer that you gave me",
            "thanks for the beer mug",
            "why did I not",
            "why did I not cry",
            "why did I insult that big guy"
        ]
        self.test_sentences = [sentence.split(" ") for sentence in sentences]
        return self.test_sentences

    # TODO:  Decide if these will just get the test data or actually assign the self.whatever.
    def load_test_n_grams(self):
        if not hasattr(self, "test_sentences") or self.test_sentences is None:
            self.test_sentences = self.load_test_sentences()
        if not hasattr(self, "test_four_grams_hist") or self.test_four_grams_hist is None:
            four_grams = bo.find_n_grams_list_of_lists(self.test_sentences, 4)
            self.test_four_grams_hist = bo.convert_n_grams_to_hist_df(four_grams)
            self.test_four_grams_hist = prd.add_source_target_columns(self.test_four_grams_hist)

        if not hasattr(self, "test_five_grams_hist") or self.test_five_grams_hist is None:
            self.test_five_grams_hist = bo.convert_n_grams_to_hist_df(bo.find_n_grams_list_of_lists(self.test_sentences, 5))
            self.test_five_grams_hist = prd.add_source_target_columns(self.test_five_grams_hist)

        if not hasattr(self, "test_six_grams_hist") or self.test_six_grams_hist is None:
            self.test_six_grams_hist = bo.convert_n_grams_to_hist_df(bo.find_n_grams_list_of_lists(self.test_sentences, 6))
            self.test_six_grams_hist = prd.add_source_target_columns(self.test_six_grams_hist)

        self.four_grams_hist = self.test_four_grams_hist
        self.five_grams_hist = self.test_five_grams_hist
        self.six_grams_hist = self.test_six_grams_hist
        self.test_histograms = [self.test_four_grams_hist, self.test_five_grams_hist, self.test_six_grams_hist]
        return self.test_histograms

    def load_test_matrix_df(self):
        # OK, for now just just the existing one, since it contains most words we need and I don't know that we
        # really need to test it right now.
        self.test_matrix_df = self.load_real_matrix_df()
        return self.matrix_df

    def load_test_prefix_maps(self):
        self.test_prefix_maps = []
        for hist in self.test_histograms:
            current_prefix_map = bo.create_prefix_map(hist)
            self.test_prefix_maps.append(current_prefix_map)
        return self.test_prefix_maps

    # TODO:  Have self.real_X, not jus self.X, so we don't keep loading if we run the whole test class and it runs
    # functions that use both the test and real data.
    def load_real_n_grams(self, source="twitter"):
        if not hasattr(self, "real_four_grams_hist") or self.real_four_grams_hist is None:
            self.real_four_grams_hist = pd.read_csv(f"../en_US.{source}.txt_4_grams.csv")
            self.real_four_grams_hist = prd.add_source_target_columns(self.real_four_grams_hist)
        if not hasattr(self, "real_five_grams_hist") or self.real_five_grams_hist is None:
            self.real_five_grams_hist = pd.read_csv(f"../en_US.{source}.txt_5_grams.csv")
            self.real_five_grams_hist = prd.add_source_target_columns(self.real_five_grams_hist)
        if not hasattr(self, "real_six_grams_hist") or self.real_six_grams_hist is None:
            self.real_six_grams_hist = pd.read_csv(f"../en_US.{source}.txt_6_grams.csv")
            self.real_six_grams_hist = prd.add_source_target_columns(self.real_six_grams_hist)
        self.four_grams_hist = self.real_four_grams_hist
        self.five_grams_hist = self.real_five_grams_hist
        self.six_grams_hist = self.real_six_grams_hist

    def load_real_matrix_df(self, source="twitter"):
        matrix_file_fp = f"../word_stats_pkls/matrix_13686_df.pkl"
        matrix_file_fp = f"../word_stats_pkls/news_matrix_df.pkl"
        matrix_file_fp = f"../word_stats_pkls/{source}_matrix_df.pkl"
        if not hasattr(self, "matrix_df") or not self.matrix_df:
            with open(matrix_file_fp, 'rb') as f:
                self.matrix_df = pickle.load(f)
        return self.matrix_df

    def load_real_prefix_maps(self, source="twitter"):
        #source = "news"
        if not hasattr(self, "four_grams_prefix_map") or self.four_grams_prefix_map is None:
            with open(PKLS_DIR + f"four_grams_prefix_map_{source}.pkl", "rb") as f:
                self.four_grams_prefix_map = pickle.load(f)

        if not hasattr(self, "five_grams_prefix_map") or self.five_grams_prefix_map is None:
            with open(PKLS_DIR + f"five_grams_prefix_map_{source}.pkl", "rb") as f:
                self.five_grams_prefix_map = pickle.load(f)

        if not hasattr(self, "six_grams_prefix_map") or self.six_grams_prefix_map is None:
            with open(PKLS_DIR + f"six_grams_prefix_map_{source}.pkl", "rb") as f:
                self.six_grams_prefix_map = pickle.load(f)
        return self.four_grams_prefix_map, self.five_grams_prefix_map, self.six_grams_prefix_map

    def use_test_data(self):
        ngram_hists = self.load_test_n_grams()
        self.four_grams_hist = ngram_hists[0]
        self.five_grams_hist = ngram_hists[1]
        self.six_grams_hist = ngram_hists[2]
        self.matrix_df = self.load_test_matrix_df()
        maps = self.load_test_prefix_maps()
        self.four_grams_prefix_map = maps[0]
        self.five_grams_prefix_map = maps[1]
        self.six_grams_prefix_map = maps[2]

    def use_real_data(self):
        self.load_real_n_grams()
        self.load_real_matrix_df()
        self.load_real_prefix_maps()

    def check_prediction(self, result, query_text):
        """
        Doesn't actually prove correctness.  But does show that everything returned matches the query string and that
        the results are in order from highest count to lowest count.
        :param result:
        :param query_text:
        :return:
        """
        result.gram.apply(lambda g: self.assertTrue(g.startswith(query_text)))
        for i in range(result.shape[0] - 1):
            self.assertGreaterEqual(result['count'].iloc[i], result['count'].iloc[i + 1])

    # TODO:  Theoretically, it is best to have the tests use values coded here or a file that is checked into the
    #  repo.
    def test_match_3_grams(self):
        #self.load_n_grams()
        three_grams = bo.find_n_grams_list_of_lists(self.sentences, 3)
        hist = bo.convert_n_grams_to_hist_df(three_grams)
        query_list = ['thank', 'you']
        result = prd.match_n_grams_one_hist_original(query_list, hist)
        print(result)
        self.check_prediction(result, constants.N_GRAM_SEPARATOR.join(query_list))

    def test_match_4_grams(self):
        #four_grams = bo.find_n_grams_list_of_lists(self.sentences, 4)
        #self.load_real_n_grams()
        [four_grams_hist, five_grams_hist, six_grams_hist] = self.load_test_n_grams()
        #hist = bo.convert_n_grams_to_hist_df(self.four_grams_hist)
        hist = self.four_grams_hist
        query_list = ['thank', 'you', 'for']
        result = prd.match_n_grams_one_hist_original(query_list, hist)
        #print(result)
        self.check_prediction(result, constants.N_GRAM_SEPARATOR.join(query_list))

        '''result = prd.match_n_grams_one_hist("a case of".split(" "), hist)
        print(result)
        result = prd.match_n_grams_one_hist("would mean the".split(" "), hist)
        print(result)
        result = prd.match_n_grams_one_hist("make me the".split(" "), hist)
        print(result)
        result = prd.match_n_grams_one_hist("struggling but the".split(" "), hist)
        print(result)'''
        result = prd.match_n_grams_one_hist_original("thanks for the".split(" "), hist)
        print(result.head())
        print(result.shape)

    def test_match_ngrams_by_index_agrees_with_original(self):
        """
        tests whether the function that matches ngrams using the index, match_n_grams_by_index(), gives the same
        result as the original function to collect ngrams
        :return:
        """
        # TODO:  An actual unit test using small test data, that does not depend on "real word" data, in addition to the
        # "real word" data below.

        self.load_real_n_grams()
        with open(PKLS_DIR + "four_grams_prefix_map_twitter.pkl", "rb") as f:
            prefix_map = pickle.load(f)

        text = "thanks for the"
        indices = prefix_map[text]
        result_from_indices = prd.match_n_grams_by_index(indices, self.four_grams_hist)
        baseline_result = prd.match_n_grams_one_hist_original(text.split(" "), self.four_grams_hist)
        self.assertTrue(baseline_result.equals(result_from_indices))

    def test_match_ngrams_with_index(self):
        self.use_test_data()

        # Try with a blank list of indices; should give an empty DataFrame.
        indices = []
        result = prd.match_n_grams_by_index(indices, self.four_grams_hist)
        self.assertEqual(0, result.shape[0])

        indices = [0, 9]
        result = prd.match_n_grams_by_index(indices, self.four_grams_hist)
        self.assertEqual(2, result.shape[0])

    def test_collect_n_gram_matches(self):
        four_grams = bo.find_n_grams_list_of_lists(self.sentences, 4)
        five_grams = bo.find_n_grams_list_of_lists(self.sentences, 5)
        six_grams = bo.find_n_grams_list_of_lists(self.sentences, 6)
        four_grams_hist = bo.convert_n_grams_to_hist_df(four_grams)
        five_grams_hist = bo.convert_n_grams_to_hist_df(five_grams)
        six_grams_hist = bo.convert_n_grams_to_hist_df(six_grams)
        prd.collect_n_grams_matches_original("ill dust them off and be on my", [four_grams_hist, five_grams_hist,
                                                                                six_grams_hist])

    def test_collect_n_grams_indices(self):
        #self.load_real_n_grams()
        list_of_histograms = self.load_test_n_grams()

        with open(PKLS_DIR + "four_grams_prefix_map_twitter.pkl", "rb") as f:
            prefix_map = pickle.load(f)

        text = "thanks for the"
        tokens = ['thanks', 'for', 'the']
        #list_of_histograms = [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist]
        #list_of_histograms = [self.four_grams_hist]
        #list_of_prefix_maps = [prefix_map]
        list_of_prefix_maps = self.load_test_prefix_maps()

        # Test that it is the same as the old
        new = prd.collect_n_grams_matches_indices(text, list_of_histograms, list_of_prefix_maps)
        baseline = prd.collect_n_grams_matches_original(tokens, list_of_histograms)
        # print(baseline[0].head())
        # print(new[0].head())
        self.assertTrue(baseline[0].equals(new[0].head(25)))   # Currently, the original way only returns 25 entries,
        # the new way all of them.

        # Test a phrase that is not in the test set; should return a list of empty DataFrames
        text = "go on a romantic date"
        result = prd.collect_n_grams_matches_indices(text, list_of_histograms, list_of_prefix_maps)
        self.assertEqual(3, len(result))
        for i in range(3):
            self.assertEqual(0, result[i].shape[0])

        # Now a phrase that is there.
        text = "may i say thanks for the"
        result = prd.collect_n_grams_matches_indices(text, list_of_histograms, list_of_prefix_maps)
        self.assertEqual(3, len(result))
        for i in range(3):
            self.assertGreater(result[i].shape[0], 0)

    def test_collect_word_vector_associations(self):
        """
        won't test this extremely vigorously right now because of the time it would take to do the calculations
        => just check the size, dimensions, etc.
        TODO:  Maybe a more rigorous test some time
        """
        text = "tiger roman"
        self.load_real_matrix_df()
        tokens = ['tiger', 'roman']
        result = prd.collect_word_vector_associations(tokens, self.matrix_df)
        self.assertEqual(8, result.shape[1])
        self.assertTrue(constants.WORD in list(result.columns))
        self.assertFalse("word" in list(result.columns))
        self.assertTrue("tiger" in list(result.columns))
        self.assertTrue("roman" in list(result.columns))
        self.assertEqual((self.matrix_df.shape[0] - 2), result.shape[0])
        # It removes the word you are searching for and any other match of 1.0 similarity, so the result size is 2 less.

        tokens = ['its', 'about', 'one', 'word', 'quality', 'cosgrove']
        result = prd.collect_word_vector_associations(tokens, self.matrix_df)
        self.assertEqual(8, result.shape[1])
        self.assertTrue("word" in list(result.columns))
        self.assertFalse("tiger" in list(result.columns))
        self.assertTrue("quality" in list(result.columns))

    def test_use_collect_word_vector_associations(self):
        phrases = [
            "Go on a romantic date at the",
            "I'll dust them off and be on my",
            "haven't seen it in quite some",
            "Louis will push his long wet hair out of his eyes with his little",
            "the good times and keep the faith during the",
            "If this isn't the cutest thing you've ever seen, then you must be",
        ]
        self.load_real_matrix_df()
        for phrase in phrases:
            tokens = phrase.lower().split(" ")
            result = prd.collect_word_vector_associations(tokens, self.matrix_df)
            print(f"\n\n\n\n----\n{phrase}")
            print(result.head(20))

    def test_get_top_results(self):
        tokens = ['tiger', 'roman']
        self.load_real_matrix_df()
        wv = prd.collect_word_vector_associations(tokens, self.matrix_df)
        top_results = prd.get_top_results(wv, self.en, 10, pos="NOUN")
        self.assertEqual(10, top_results.shape[0])
        self.assertEqual(wv.shape[1], top_results.shape[1])

        top_results = prd.get_top_results(wv, self.en, 17, pos="VERB")
        self.assertEqual(17, top_results.shape[0])
        self.assertEqual(wv.shape[1], top_results.shape[1])

    def test_get_top_results_blank_result(self):
        self.load_real_matrix_df()
        tokens = ['i', 'am', 'not', 'there', 'i', 'do', 'not']
        wv = prd.collect_word_vector_associations(tokens, self.matrix_df)
        top_results = prd.get_top_results(wv, self.en, 10, pos="NOUN")

    @unittest.skip("takes a long time and this function is deprecated anyway")
    def test_predict_from_word_vectors(self):
        text = "Bills game: Offense still struggling but the"
        text = "Go on a romantic date at the"
        text = "Louis will push his long wet hair out of his eyes with his little"
        text = "Be grateful for the good times and keep the faith during the"
        text = "the good times and keep the faith during the"
        tokens = text.split(" ")
        tokens = bo.tokenize_string(text)
        # en = spacy.load('en_core_web_md')
        # vocab = list(en.vocab.strings)
        # en_words = set(v.lower() for v in vocab)
        result = prd.predict_from_word_vectors(tokens, self.en_words, self.en, self.en, "NOUN")
        print(result)
        # TODO:  Actually test something.

    def test_predict_from_word_vectors_matrix(self):
        text = "Bills game: Offense still struggling but the"
        text = "Go on a romantic date at the"
        # with open("../word_stats_pkls/matrix_13686_df.pkl", 'rb') as f:
        #    matrix_df = pickle.load(f)

        self.load_real_matrix_df()
        tokens = bo.tokenize_string(text)
        start = time.time()
        result1 = prd.predict_from_word_vectors_matrix(tokens, self.matrix_df, self.en)
        stop = time.time()
        self.assertTrue(constants.WORD in result1.columns)
        self.assertEqual(constants.DEFAULT_TOP_ASSOCIATIONS, result1.shape[0])
        print(result1)
        print(f"{stop - start} seconds")

        start = time.time()
        result2 = prd.predict_from_word_vectors_matrix(tokens, self.matrix_df, self.en, top_number=17)
        stop = time.time()
        self.assertTrue(constants.WORD in result2.columns)
        self.assertEqual(17, result2.shape[0])
        self.assertTrue(result2.iloc[:constants.DEFAULT_TOP_ASSOCIATIONS].equals(result1))
        print(result2)
        print(f"{stop - start} seconds")

        text = "lachs was one of the few physicians who would go to visit protective"
        tokens = bo.tokenize_string(text)
        start = time.time()
        result3 = prd.predict_from_word_vectors_matrix(tokens, self.matrix_df, self.en, top_number=17)
        stop = time.time()
        print(f"{stop - start} seconds")

        tokens = ['i', 'am', 'not', 'there', 'i', 'do', 'not']
        target = "sleep" # from the training set
        start = time.time()
        result4 = prd.predict_from_word_vectors_matrix(tokens, self.matrix_df, self.en, top_number=17)
        stop = time.time()
        print(f"{stop - start} seconds")

    @unittest.skip("In the past they gave the same result, and this takes too long.")
    def test_compare_predict_functions(self):
        # text = "Bills game: Offense still struggling but the"
        # with open("../word_stats_pkls/matrix_13686_df.pkl", 'rb') as f:
        #    matrix_df = pickle.load(f)
        self.load_real_matrix_df()
        self.compare_predict_functions_one_phrase("Bills game: Offense still struggling but the", self.matrix_df)
        self.compare_predict_functions_one_phrase("Go on a romantic date at the", self.matrix_df)

    def compare_predict_functions_one_phrase(self, text, matrix_df):
        print(f"comparing {text}")
        tokens = bo.tokenize_string(text)
        start1 = time.time()
        result1 = prd.predict_from_word_vectors(tokens, self.en_words, self.en, self.en, "NOUN")
        stop1 = time.time()
        print(result1)
        print(f"{stop1 - start1} seconds")

        start_matrix = time.time()
        matrix_result = prd.predict_from_word_vectors_matrix(tokens, matrix_df, self.en)
        stop_matrix = time.time()
        print(matrix_result)
        print(f"{stop_matrix - start_matrix} seconds")

    def test_predict_from_ngrams_and_vectors(self):
        pass
        a = 'a'
        print(a)
        self.use_test_data()

        # a phrase that is not in the prefix max
        phrase = "go on a romantic date"
        histograms = [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist]
        prefix_maps = [self.four_grams_prefix_map, self.five_grams_prefix_map, self.six_grams_prefix_map]
        result1 = prd.predict(phrase, histograms, prefix_maps, self.matrix_df, self.en)

        # a phrase that is in the prefix max
        phrase = "may i say thanks for the"
        histograms = [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist]
        prefix_maps = [self.four_grams_prefix_map, self.five_grams_prefix_map, self.six_grams_prefix_map]
        result2 = prd.predict(phrase, histograms, prefix_maps, self.matrix_df, self.en)
        print(result1)
        print(result2)

    @unittest.skip("not now due to it's slowness")
    def test_predict_from_ngrams_and_vectors_actual_data(self):
    #def test_predict_from_ngrams_and_vectors(self):
        phrases = [
            #"a pound of bacon, a bouquet, and a case of",
            #"It would mean the",
            #"can you follow me and make me the",
            #"Bills game: Offense still struggling but the",
            "Go on a romantic date at the",
            "I'll dust them off and be on my",
            "haven't seen it in quite some",
            "Louis will push his long wet hair out of his eyes with his little",
            "the good times and keep the faith during the",
            "If this isn't the cutest thing you've ever seen, then you must be",
        ]
        self.load_real_n_grams()
        self.load_real_matrix_df()
        self.load_real_prefix_maps()
        for phrase in phrases:
            print("\n\n----", phrase)

            # tokens = bo.tokenize_string(phrase)
            """prd.collect_n_grams_matches(phrase, [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist])
            result = prd.predict_from_word_vectors(tokens, en_words, en.vocab, en, "NOUN")
            print(result)"""
            #prd.predict(tokens, [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist],
            #            en_words, en.vocab, en, "NOUN")
            histograms = [self.four_grams_hist, self.five_grams_hist, self.six_grams_hist]
            prefix_maps = [self.four_grams_prefix_map, self.five_grams_prefix_map, self.six_grams_prefix_map]
            print(phrase)
            current_result = prd.predict(phrase, histograms, prefix_maps, self.matrix_df, self.en)
            print(current_result)

    def test_do_one_prediction_test(self):
        histograms = self.load_test_n_grams()
        prefix_maps = self.load_test_prefix_maps()
        matrix_df = self.load_test_matrix_df()
        sentences = self.load_test_sentences()

        # Something that is not there
        '''
        result1 = prd.do_one_prediction_test("this is what happens when you try to", "fly", histograms, prefix_maps,
                                             matrix_df, self.en)
        self.assertEqual(2, len(result1))
        self.assertEqual(-1, result1[0])
        self.assertEqual("neither", result1[1])
        #'''

        # Something that should be there in the ngrams
        '''
        result2 = prd.do_one_prediction_test("one two three", "four", histograms, prefix_maps,
                                             matrix_df, self.en, threshold=1)
        self.assertEqual(2, len(result2))
        self.assertEqual(0, result2[0])
        self.assertEqual("ngram", result2[1])
        #'''

        # Something that is not there in the ngrams but should be in the word vector results
        # ?

        # neither
        '''
        result4 = prd.do_one_prediction_test("wet ocean ice", "water", histograms, prefix_maps,
                                             matrix_df, self.en, threshold=1)
        print(result4)
        self.assertEqual(2, len(result4))
        self.assertEqual(1, result4[0])
        self.assertEqual("vector", result4[1])
        '''

        result5 = prd.do_one_prediction_test("wet ocean ice", "water", histograms, prefix_maps,
                                             matrix_df, self.en, threshold=1)
        print(result5)
        self.assertEqual(2, len(result5))
        #self.assertEqual(1, result5[0])
        #self.assertEqual("vector", result5[1])


if __name__ == '__main__':
    unittest.main()
