import unittest
import spacy
import style as st
import pandas as pd


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
        with open('en_docs_lms/moby_no_header_md.pkl', 'rb') as f:
            moby = pickle.load(f)
        """self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)
        self.compare_sentence_lengths_1_and_2(moby)"""

        print("\nfyj")
        with open('fyj/fyj_md.pkl', 'rb') as f:
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

    def test_verbs_map(self):
        text = 'This is Jane. This is Dick. See Dick run. See Jane play with the dog while skydiving.'
        doc = self.en(text)
        map = st.find_POS_maps(doc)
        print(map)
        a = 'b'

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

    def test_find_pos_map(self):
        text = "The yellow curtain swayed silently in the gentle breeze. The crow croaked outside. A mouse " \
               "scampered across the floor, spreading Bubonic Plague as it went."
        doc = self.en(text)
        pos_map = st.find_POS_maps(doc)
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

if __name__ == '__main__':
    unittest.main()
