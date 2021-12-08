import unittest
from basic_operations import *
import pickle

# TODO:  For each of these, test some potentially problematic input.
# Empty strings, anything else.  For the coursera project, each file will have
# plenty of words and sentences, but for long term robustness, come up with
# more test cases.


class CleansingTest(unittest.TestCase):

    def test_cleanse(self):
        text = "you're nice.  maybe; I don't think I want to be a robot!"
        result = cleanse(text)
        self.assertEqual("youre nice maybe i dont think i want to be a robot", result)

        text = "You're nice.  maybe; I don't think I want to bE a Robot!"
        self.assertEqual("youre nice maybe i dont think i want to be a robot", cleanse(text))

        text = "You're nice.  maybe; \nI don't think I want to bE a Robot!"
        self.assertEqual("youre nice maybe i dont think i want to be a robot", cleanse(text))

    def test_cleanse_apostrophe(self):
        text = "what's up you're hot"
        self.assertEqual("whats up youre hot", cleanse(text))

    def test_cleanse_with_punct_without_space(self):
        text = "hey,what's up.NOt much here."
        self.assertEqual("hey whats up not much here", cleanse(text))

    def test_cleanse_file(self):
        with open(r'C:\Users\paulj_1e1uzlz\courses\data_science_capstone\en_US/twitter_test_2.txt', 'r',
                  encoding='UTF-8') as f:
            test_2 = f.read()
            result = cleanse(test_2)
            # print(result)
            tokens = tokenize_string(result)
            print(tokens)


class TokenizeTest(unittest.TestCase):

    def test_basic_tokenize(self):
        text = "youre nice maybe i dont think i want to be a robot"
        result = tokenize_string(text)
        # print(result)
        self.assertEqual(['youre', 'nice', 'maybe', 'i', 'dont', 'think', 'i', 'want', 'to', 'be', 'a', 'robot'],
                         result)

    def test_tokenize_with_some_punct_and_upper(self):
        text = "You're nice.  maybe; I don't think I want to bE a Robot!"
        result = tokenize_string(text)
        # print(result)
        self.assertEqual(['youre', 'nice', 'maybe', 'i', 'dont', 'think', 'i', 'want', 'to', 'be', 'a', 'robot'],
                         result)

    def test_tokenize_blank_input(self):
        result = tokenize_string('')
        self.assertEqual([''], result)
        # At the moment it is returning [''] for blank input, due to the way split() works.  I don't like it but
        # am not sure what the best option really is.
        # TODO:  Have a strategy for dealing with blank input for each function.

    def test_split_to_sentences(self):
        text = "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is."
        sentences = split_to_sentences(text)
        # print(sentences)
        self.assertEquals(["One sentence", "Two sentences", "To be or not to be", "Whatever",
                          "The problem is that I don't even know what a sentence is"], sentences)
        # TODO:  might should have that last item be removed.  But that would involve going through and removing all
        # blank items, because they could be at places other than the very end.

    def test_tokenize_by_sentence(self):
        text = "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is."
        tokens = tokenize_by_sentence(text)
        # print(tokens)
        expected_result = [['one', 'sentence'], ['two', 'sentences'], ['to', 'be', 'or', 'not', 'to', 'be'],
                           ['whatever'], ['the', 'problem', 'is', 'that', 'i', 'dont', 'even', 'know', 'what', 'a',
                                          'sentence', 'is']]
        self.assertEqual(expected_result, tokens)

    def test_tokenize_stuff_with_commas(self):
        # something I found in the file previously
        text = "sanctified,spiritual,superhuman,superlative,supernatural,transcendental,"
        result = tokenize_string(text)
        print(result)
        self.assertEqual(['sanctified', 'spiritual', 'superhuman', 'superlative', 'supernatural', 'transcendental'],
                         result)


class StatsTest(unittest.TestCase):

    def test_basic_word_stats(self):
        text = "You're nice.  maybe; I don't think I want to bE a Robot!"
        stats = find_word_stats(text)
        print(stats)

    def test_sentence_length_hist(self):
        text = "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is."
        sentences = tokenize_by_sentence(text)
        hist = find_sentence_lengths_hist(sentences)
        expected_result = pd.Series([2, 2, 6, 1, 12]).value_counts().sort_index()
        print(hist)
        self.assertTrue((hist == expected_result).all())


class NGramsTest(unittest.TestCase):

    def test_bi_grams(self):
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        bi_grams = find_n_grams_1_d_list(input, 2)
        # print(bi_grams)
        self.assertEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], bi_grams)

    def test_tri_grams(self):
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tri_grams = find_n_grams_1_d_list(input, 3)
        # print(tri_grams)
        self.assertEqual([ [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9] ],
                          tri_grams)

    def test_less_intelligent_n(self):
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], find_n_grams_1_d_list(input, 1))
        self.assertEqual([], find_n_grams_1_d_list(input, 0))
        self.assertEquals([], find_n_grams_1_d_list(input, -1))

        input = [1]
        self.assertEqual([[1]], find_n_grams_1_d_list(input, 1))  # should work
        self.assertEqual([], find_n_grams_1_d_list(input, 2))  # should work

    def test_ngrams_list_of_lists(self):
        input = [['one', 'sentence'], ['two', 'sentences'], ['to', 'be', 'or', 'not', 'to', 'be'],
                 ['whatever'], ['the', 'problem', 'is', 'that', 'i', 'dont', 'even', 'know', 'what', 'a',
                                'sentence', 'is']]
        expected_result = [['one', 'sentence'], ['two', 'sentences'], ['to', 'be'], ['be', 'or'], ['or', 'not'],
                           ['not', 'to'], ['to', 'be'], ['the', 'problem'], ['problem', 'is'], ['is', 'that'],
                           ['that', 'i'], ['i', 'dont'], ['dont', 'even'], ['even', 'know'], ['know', 'what'],
                           ['what', 'a'], ['a', 'sentence'], ['sentence', 'is']]
        result = find_n_grams_list_of_lists(input, 2)
        # print(result)
        self.assertEqual(expected_result, result)

    def test_ngrams_list_of_lists_blank_input(self):
        self.assertEqual([], find_n_grams_list_of_lists([], 2))
        self.assertEqual([], find_n_grams_list_of_lists([], 3))

        self.assertEqual([], find_n_grams_list_of_lists(None, 2))
        self.assertEqual([], find_n_grams_list_of_lists(None, 3))

    def test_find_ngrams_list_of_list_bad_n(self):
        input = [['one', 'sentence'], ['two', 'sentences'], ['to', 'be', 'or', 'not', 'to', 'be'],
                 ['whatever'], ['the', 'problem', 'is', 'that', 'i', 'dont', 'even', 'know', 'what', 'a',
                                'sentence', 'is']]

        self.assertEqual([], find_n_grams_list_of_lists(input, 0))
        self.assertEqual([], find_n_grams_list_of_lists(input, -1))

    def test_find_n_grams_from_text(self):
        text = "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is."
        expected_result = [
            ['one', 'sentence'], ['two', 'sentences'], ['to', 'be'], ['be', 'or'], ['or', 'not'],
            ['not', 'to'], ['to', 'be'], ['the', 'problem'], ['problem', 'is'], ['is', 'that'], ['that', 'i'],
            ['i', 'dont'], ['dont', 'even'], ['even', 'know'], ['know', 'what'], ['what', 'a'], ['a', 'sentence'],
            ['sentence', 'is']
        ]
        result = find_n_grams_from_text(text, 2)
        print(result)
        self.assertEqual(expected_result, result)

    def test_convert_n_grams_to_hist_df(self):
        n_grams = [
            ['on', 'the', 'fridge'], ['on', 'the', 'table'], ['on', 'the', 'fridge'], ['the', 'stupid', 'duck'],
            ['the', 'stupid', 'dog'], ['on', 'the', 'fridge'], ['on', 'the', 'table'], ['the', 'stupid', 'duck']
        ]
        hist = convert_n_grams_to_hist_df(n_grams)
        self.assertEqual(4, hist.shape[0])
        on_the_fridge = hist[hist.gram.str.contains("on the fridge")]
        self.assertEqual(1, on_the_fridge.shape[0])
        self.assertEqual(3, on_the_fridge.iloc[0]["count"])
        # on_the_fridge.iloc[0].count does not work because count() is a function

    def test_can_match_from_n_gram_hist(self):
        n_grams = [
            ['on', 'the', 'fridge'], ['on', 'the', 'table'], ['on', 'the', 'fridge'], ['the', 'stupid', 'duck'],
            ['the', 'stupid', 'dog'], ['on', 'the', 'fridge'], ['on', 'the', 'table'], ['the', 'stupid', 'duck']
        ]
        hist = convert_n_grams_to_hist_df(n_grams)
        on_the = hist[hist.gram.str.contains("on the")]
        self.assertEqual(2, on_the.shape[0])
        # self.assertEqual(pd.DataFrame({"gram": ["on the fridge"], "count": [3]}).iloc[0], on_the.iloc[0])
        self.assertTrue(pd.DataFrame({"gram": ["on the fridge"], "count": [3]}).iloc[0].equals(on_the.iloc[0]))
        self.assertTrue(pd.DataFrame({"gram": ["on the table"], "count": [2]}).iloc[0].equals(on_the.iloc[1]))


class PrefixMapTest(unittest.TestCase):

    def test_split_prefix(self):
        self.assertEqual("", split_prefix("one"))
        self.assertEqual("", split_prefix(""))
        self.assertEqual("one", split_prefix("one two"))
        self.assertEqual("one two", split_prefix("one two three"))
        self.assertEqual("one two", split_prefix("one two three"))
        self.assertEqual("one two three", split_prefix("one two three four"))
        self.assertEqual("one two three four", split_prefix("one two three four five"))

    def test_create_prefix_map(self):
        # create a df, make a prefix map
        gram = ["thanks for the follow", "cant wait to see", "some other word pair", "thanks for the rt"]
        count = [8, 3, 1, 27]
        histogram = pd.DataFrame({"gram": gram, "count": count})
        prefix_map = create_prefix_map(histogram)
        self.assertTrue(type(prefix_map) == dict)
        self.assertEqual(dict, type(prefix_map))
        self.assertEqual([0, 3], prefix_map["thanks for the"])
        self.assertEqual([1], prefix_map["cant wait to"])
        self.assertEqual([2], prefix_map["some other word"])

    @unittest.skip("comment this out to create the prefix map")
    def test_create_an_actual_prefix_map(self):
        four_gram_hist = pd.read_csv('../en_US.twitter.txt_4_grams.csv')
        four_gram_prefix_map = create_prefix_map(four_gram_hist)
        with open("../word_stats_pkls/four_grams_prefix_map.pkl", 'wb') as f:
            pickle.dump(four_gram_prefix_map, f)

        five_gram_hist = pd.read_csv('../en_US.twitter.txt_5_grams.csv')
        five_gram_prefix_map = create_prefix_map(five_gram_hist)
        with open("../word_stats_pkls/five_grams_prefix_map.pkl", 'wb') as f:
            pickle.dump(five_gram_prefix_map, f)

        six_gram_hist = pd.read_csv('../en_US.twitter.txt_6_grams.csv')
        six_gram_prefix_map = create_prefix_map(six_gram_hist)
        with open("../word_stats_pkls/six_grams_prefix_map.pkl", 'wb') as f:
            pickle.dump(six_gram_prefix_map, f)


class TrainingSetTest(unittest.TestCase):

    def test_get_random_sentences(self):
        np.random.seed(16)  # You must set the seed for the unit test to work.
        # The function should work equally well with numbers or strings.
        s = [i for i in range(50)]
        result = get_random_sentences(s, 10)
        exp_result = [ 3, 34, 25, 36, 20, 23, 35, 21, 40, 39]
        self.assertTrue((exp_result == result).all())

        # With the same values as strings
        np.random.seed(16)
        s = [str(i) for i in range(50)]
        result = get_random_sentences(s, 10)
        exp_result = [ '3', '34', '25', '36', '20', '23', '35', '21', '40', '39']
        self.assertTrue((exp_result == result).all())

        np.random.seed(16)
        result_50 = get_random_sentences(s, 51)

        # Now try trying to get too many sentences
        np.random.seed(16)
        result_51 = get_random_sentences(s, 51)
        # The results for 50 and 51 should be the same.
        self.assertTrue((result_50 == result_51).all())

        np.random.seed(16)
        result_5127 = get_random_sentences(s, 5127)
        self.assertTrue((result_50 == result_5127).all())

        # now test for checking the word length
        # First, try with 0 length; should be the same as before
        s = [((str(i) + ' ') * i).strip() for i in range(50)]
        s = [(str(i) + ' ') * i for i in range(50)]
        min_length = 0
        np.random.seed(16)
        result = get_random_sentences(s, 10, min_length)
        self.assertEqual('3 3 3 ', result[0])
        self.assertEqual("34 " * 34, result[1])
        all_indices = range(len(s))
        np.random.seed(16)
        shuffled_indices = np.random.choice(all_indices, size=len(s), replace=False)
        shuffled_str = []
        for i in range(len(shuffled_indices)):
            shuffled_str.append((str(shuffled_indices[i]) + " ") * shuffled_indices[i])
        # self.assertEqual(shuffled_str[:10], list(result[:10])) This is equivalent to the line below.
        self.assertTrue((shuffled_str[:10] == result[:10]).all())

        # next, another length
        np.random.seed(16)
        min_length = 17
        how_many=10
        result = get_random_sentences(s, how_many=how_many, min_num_words=min_length)
        exp_result = [x for x in shuffled_str if len(x.split(" ")) >= min_length][:how_many]
        print(exp_result)
        self.assertTrue((exp_result == result).all())

    def test_get_training_sentences(self):
        text = "0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49."
        np.random.seed(16)
        result = get_training_sentences(text, 10)
        exp_result = [ '3', '34', '25', '36', '20', '23', '35', '21', '40', '39']
        self.assertTrue((exp_result == result).all())

    # now from an actual file
    def test_get_training_sentences_from_file(self):
        file_name = "../corpus/moby_dick.txt"
        #with open(file_name, 'r') as f:
        #    text = f.read()
        n = 1000
        result = get_training_sentences_from_file(file_name, n)
        self.assertEqual(n, len(result))
        for sentence in result:
            self.assertTrue(sentence == sentence.lower())
            self.assertTrue(sentence == sentence.replace(".", ""))

        # With a minimum number of words
        result = get_training_sentences_from_file(file_name, how_many=n, min_num_words=6)
        self.assertEqual(n, len(result))
        for sentence in result:
            self.assertTrue(sentence == sentence.lower())
            self.assertTrue(sentence == sentence.replace(".", ""))


if __name__ == '__main__':
    unittest.main()
