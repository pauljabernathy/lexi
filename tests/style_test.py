import unittest
import spacy
import style as st
import pandas as pd
import pickle


class SentenceLength(unittest.TestCase):

    def setUp(self):
        self.en = spacy.load('en_core_web_md')
        self.es = spacy.load('es_core_news_md')

    def test_find_sentence_lengths(self):
        text = 'I am golden.  I am made of gold.  Just kidding - I am actually made of cheaper materials.  ' \
               'My name is Thomas.  I am a very useful engine.'
        text = 'I am golden. I am made of gold. Just kidding - I am actually made of cheaper materials. ' \
               'My name is Thomas. I am a very useful engine.'
        # 3, 5, 9, 4, 6
        # text = "John ate eggs and Mary ate potatoes"
        lengths = st.find_sentence_lengths(self.en(text))
        print(lengths)
        self.assertEqual(3, lengths[0])
        self.assertEqual([3, 5, 9, 4, 6], lengths)

    def test_find_sentence_lengths_2(self):
        text = 'I am golden.  I am made of gold.  Just kidding - I am actually made of cheaper materials.  ' \
               'My name is Thomas.  I am a very useful engine.'
        text = 'I am golden. I am made of gold. Just kidding - I am actually made of cheaper materials. ' \
               'My name is Thomas. I am a very useful engine.'
        # 3, 5, 9, 4, 6
        # text = "John ate eggs and Mary ate potatoes"
        lengths = st.find_sentence_lengths_2(self.en(text))
        print(lengths)
        self.assertEqual(3, lengths[0])
        self.assertEqual([3, 5, 9, 4, 6], lengths)

    def test_compare_sentence_lengths_1_and_2(self):
        import pickle
        print("moby")
        with open('../en_docs_lms/moby_md.pkl', 'rb') as f:
            moby = pickle.load(f)
        """self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)"""

        print("\nfyj")
        with open('../fyj/fyj_md.pkl', 'rb') as f:
            fyj = pickle.load(f)
        """self.compare_sentence_lengths_1_and_2(fyj)
        self.compare_sentence_lengths_1_and_2(fyj)
        self.compare_sentence_lengths_1_and_2(fyj)
        self.compare_sentence_lengths_1_and_2(fyj)"""

        num_runs = 8
        results_1 = []
        results_2 = []
        for i in range(num_runs):
            results_1.append(self.compare_sentence_lengths_1_and_2(moby))
            results_2.append(self.compare_sentence_lengths_1_and_2(fyj))

        #first = [x[0] for x in results_1] + [x[0] for x in results_2]
        #second = [x[1] for x in results_1] + [x[1] for x in results_2]
        doc_names = ['moby', 'fyj']
        results_df = pd.DataFrame({'doc': [doc_names[0]] * num_runs + [doc_names[1]] * num_runs,   # ['moby'] * num_runs + ['fyj'] * num_runs,
                                   'first': [x[0] for x in results_1] + [x[0] for x in results_2],
                                   'second': [x[1] for x in results_1] + [x[1] for x in results_2]
                                   })
        results_df['ratio'] = results_df['second'] / results_df['first']
        print(results_df)
        print(results_df.groupby('doc').mean())
        mt = results_df[results_df.doc == 'moby']
        ft = results_df[results_df.doc == 'fyj']
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.displot(mt['ratio'])
        sns.displot(mt['ratio'])
        plt.show()
        a = 'a'

    def compare_sentence_lengths_1_and_2(self, doc):
        from time import time
        start_1 = time()
        result_1 = st.find_sentence_lengths(doc)
        end_1 = time()

        start_2 = time()
        result_2 = st.find_sentence_lengths_2(doc)
        end_2 = time()

        diff_1 = end_1 - start_1
        diff_2 = end_2 - start_2
        print(diff_1)
        print(diff_2)
        print(result_1 == result_2)
        return diff_1, diff_2

    def test_find_sentence_lengths_hist(self):
        text = 'I am golden. I am made of gold. Just kidding - I am actually made of cheaper materials. ' \
               'My name is Thomas. I am a very useful engine.'
        #lengths = st.find_sentence_lengths_hist(self.en(text))
        #self.assertEqual(3, lengths[0])
        text = 'My name is Thomas. I am a very useful engine. I am a train. I am blue. ' \
               'Henry is a green train engine. James is red.'
        lengths = st.find_sentence_lengths_hist(self.en(text))
        # 4, 6, 4, 3, 6, 3 => 4:2, 6: 2, 3:1 => 3:1, 4:2, 6:2
        self.assertTrue(((pd.Series([2, 2, 2], index=[3, 4, 6])) == lengths).all())

    def test_find_sentence_lengths_map(self):
        text = 'My name is Thomas. I am a very useful engine. I am a train. I am blue. ' \
               'Henry is a green train engine. James is red.'
        result = st.find_sentence_lengths_map(self.en(text))
        print(result)
        self.assertEquals([3, 5], result[result.length == 3].sentences.iloc[0])
        self.assertEquals([0, 2], result[result.length == 4].sentences.iloc[0])
        self.assertEquals([1, 4], result[result.length == 6].sentences.iloc[0])

    def test_find_num_verbs(self):
        text = 'This is Jane. This is Dick. See Dick run. See Jane play with the dog while skydiving.'
        doc = self.en(text)
        sentences = list(doc.sents)
        self.assertEqual(1, st.find_num_verbs(sentences[0]))
        self.assertEqual(1, st.find_num_verbs(sentences[1]))
        self.assertEqual(2, st.find_num_verbs(sentences[2]))
        self.assertEqual(2, st.find_num_verbs(sentences[3]))

    def test_find_num_part_of_speech(self):
        text = 'This is Jane. This is Dick. See Dick run. See Jane play with the dog while skydiving.'
        doc = self.en(text)
        sentences = list(doc.sents)
        '''self.assertEqual(st.find_num_verbs(sentences[0]), st.find_num_part_of_speech(sentences[0], ['VERB', 'AUX']))
        self.assertEqual(st.find_num_verbs(sentences[1]), st.find_num_part_of_speech(sentences[1], ['VERB', 'AUX']))
        self.assertEqual(st.find_num_verbs(sentences[2]), st.find_num_part_of_speech(sentences[2], ['VERB', 'AUX']))
        self.assertEqual(st.find_num_verbs(sentences[3]), st.find_num_part_of_speech(sentences[3], ['VERB', 'AUX']))'''

        text = "The yellow curtain swayed silently in the gentle breeze."
        doc = self.en(text)
        sentences = list(doc.sents)
        self.assertEqual(2, st.find_num_part_of_speech(sentences[0], ['ADJ']))
        self.assertEqual(2, st.find_num_part_of_speech(sentences[0], ['NOUN']))
        self.assertEqual(1, st.find_num_part_of_speech(sentences[0], ['ADV']))

    def test_find_pos_in_sentences_counts(self):
        text = "The yellow curtain swayed silently in the gentle breeze. The crow croaked outside. A mouse " \
               "scampered across the floor, spreading Bubonic Plague as it went."
        doc = self.en(text)
        pos_map = st.find_POSs_in_sentences_counts(doc)
        print(pos_map)

    def test_isnan(self):
        st.isnan(float('nan'))
        st.isnan(([1,2]))

    def test_fill_nas(self):
        s = pd.Series([[1,2], float('nan')])
        r = st.fill_nas(s)
        self.assertTrue(r.equals(pd.Series([[1, 2], []])))
        # self.assertEqual(pd.Series([[1, 2], []]), r)
        # self.assertEqual(pd.Series([[1, 2], [8, 17]]), st.fill_nas(pd.Series([1, 2], [8, 17])))
        r = st.fill_nas(pd.Series([[1, 2], [8, 17]]))
        self.assertTrue(pd.Series([[1, 2], [8, 17]]).equals(st.fill_nas(pd.Series([[1, 2], [8, 17]]))))
        # self.assertEqual(pd.Series([[1, 2], [], [8, 17]]), st.fill_nas(pd.Series([1, 2], float('nan'), [8, 17])))
        self.assertTrue(pd.Series([[1, 2], [], [8, 17]]).equals(
            st.fill_nas(pd.Series([[1, 2], float('nan'), [8, 17]])))
        )

    def test_find_pos_histogram(self):
        text = "the dog ate the food"
        doc = self.en(text)
        result = st.find_pos_histogram(doc)
        # print(result)
        self.assertEqual(2, result.loc['NOUN'].iloc[0])
        self.assertEqual(2, result.loc['DET'].iloc[0])
        self.assertEqual(1, result.loc['VERB'].iloc[0])

        with open("../en_docs_lms/moby_md.pkl", 'rb') as f:
            md = pickle.load(f)
        md_hist = st.find_pos_histogram(md)
        md_hist2 = st.find_pos_histogram2(md)
        md_hist3 = st.find_pos_histogram3(md)
        self.assertTrue(md_hist.equals(md_hist2))
        self.assertTrue(md_hist.equals(md_hist3))

        from functools import partial
        from timeit import timeit
        with_map = partial(st.find_pos_histogram, doc=md)
        with_get_token_info = partial(st.find_pos_histogram2, doc=md)
        with_value_counts = partial(st.find_pos_histogram3, doc=md)
        n = 10
        map_result = timeit(with_map, number=n)
        token_info_result = timeit(with_get_token_info, number=n)
        value_counts_result = timeit(with_value_counts, number=n)
        print(map_result)
        print(token_info_result)
        print(value_counts_result)
        '''
        1.4961301999999996
        13.7671323
        1.0406632999999985
        => Using lu.find_pos_histogram is much slower, probably because that function does a lot of things.
        Using just value_counts() is a hair faster than the function I wrote that uses a map (OK, a "dict" in python 
        terminology).  I don't know how much all this will affect things in the context of the entire program through.
        '''

    def test_pos_pos_map(self):
        text = "The color of the telescope. Eat apples and bananas."
        result = st.make_pos_pos_map(self.en(text))
        print(result)
        self.assertEqual({'NOUN': 2}, result['DET'])
        self.assertEqual({'ADP': 1, 'CCONJ': 1}, result['NOUN'])
        self.assertEqual({'NOUN': 1}, result['VERB'])
        self.assertEqual({'NOUN': 1}, result['CCONJ'])
        self.assertEqual({'DET': 1}, result['ADP'])

    def test_pos_pos_df(self):
        text = "The color of the telescope. Eat apples and bananas."
        result = st.make_pos_pos_df(self.en(text))
        result = result.sort_index()
        print(result)
        self.assertEqual(1.0, result['count_after_NOUN']['ADP'])
        '''self.assertTrue(([1, 1, 0, 0] == result['count_after_NOUN'].values).all())
        #self.assertEqual(1.0, result['count_after_'][''])
        self.assertTrue(([0.0, 0, 1, 0] == result['count_after_ADP'].values).all())
        self.assertTrue(([0, 0, 0, 2] == result['count_after_DET'].values).all())
        self.assertTrue(([0, 0, 0, 1] == result['count_after_VERB'].values).all())
        self.assertTrue(([0, 0, 0, 1] == result['count_after_CCONJ'].values).all())
        # adp cconj det noun verb     adp cconj det noun
        '''

        with open("../en_docs_lms/moby_md.pkl", 'rb') as f:
            md = pickle.load(f)
        moby_result = st.make_pos_pos_df(md)
        print(moby_result)
        #moby_result.append(pd.DataFrame())

        # Now make sure we can save it and reload it and it is the same.
        moby_result.to_csv("moby_pos_pos.csv", index=True)
        md_pos_pos = pd.read_csv("moby_pos_pos.csv", index_col=0)  # Need to specify that we are loading the index.
        print(md_pos_pos.equals(moby_result))
        self.assertTrue(md_pos_pos.equals(moby_result))

    def test_word_pos_map(self):
        text = "The color of the telescope. Eat apples and bananas."
        doc = self.en(text)
        result = st.make_word_pos_map(doc)
        print(result)
        self.assertTrue("The" not in result)
        self.assertTrue("the" in result and result['the'] == {'NOUN': 2})
        self.assertTrue("color" in result and result["color"] == {"ADP": 1})
        self.assertTrue("of" in result and result["of"] == {"DET": 1})
        self.assertTrue("telescope" not in result)  # because it is the last word in the sentence
        self.assertTrue("eat" in result and result["eat"] == {"NOUN": 1})
        self.assertTrue("apples" in result and result["apples"] == {"CCONJ": 1})
        self.assertTrue("and" in result and result["and"] == {"NOUN": 1})
        self.assertTrue("bananas" not in result)  # because it is the last word in the sentence
        # self.assertTrue("" in result)


if __name__ == '__main__':
    unittest.main()
