import unittest
from basic_operations import *

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
                          "The problem is that I don't even know what a sentence is", ''], sentences)
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
        bi_grams = find_n_grams_list_of_strings(input, 2)
        # print(bi_grams)
        self.assertEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], bi_grams)

    def test_tri_grams(self):
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tri_grams = find_n_grams_list_of_strings(input, 3)
        # print(tri_grams)
        self.assertEqual([ [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9] ],
                          tri_grams)

    def test_less_intelligent_n(self):
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], find_n_grams_list_of_strings(input, 1))
        self.assertEqual([], find_n_grams_list_of_strings(input, 0))
        self.assertEquals([], find_n_grams_list_of_strings(input, -1))

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
        pass
        text = "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is."
        expected_result = [ ['one', 'sentence'], ['two', 'sentences'], ['to', 'be'], ['be', 'or'], ['or', 'not'],
                           ['not', 'to'], ['to', 'be'], ['the', 'problem'], ['problem', 'is'], ['is', 'that'],
                           ['that', 'i'], ['i', 'dont'], ['dont', 'even'], ['even', 'know'], ['know', 'what'],
                           ['what', 'a'], ['a', 'sentence'], ['sentence', 'is']]
        result = find_n_grams_from_text(text, 2)
        print(result)
        self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()
