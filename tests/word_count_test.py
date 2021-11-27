import unittest
import spacy
import pandas as pd
import word_count as wc
import pickle
import lemma_utils as lu


class CreateLemmaMapTest(unittest.TestCase):

    def test_create_lemma_map_from_doc(self):
        es = wc.es#spacy.load('es_core_news_md')
        doc1 = es('No quiero taco bell.  Nosotros queremos taco bell.  Ellos quieren taco bell.')
        doc1 = es('No quiero taco bell.  Ellos no quieren taco bell.  Él quiere taco bell.  Ustedes quieren taco '
                  'bell.  Tú no quieres taco bell.  Nosotros queremos taco bell.')

        doc2 = es('Él quiere tacos.  Sin embargo, no los quiero.  Yo prefiero hamburguesas.  Siempre queremos '
                  'hamburguesas cada fin de semana.')
        result = wc.create_lemma_map_from_doc(doc2)
        #print(result)
        self.assertTrue(True)
        print("now print the occurences:")
        sents = [sent for sent in doc2.sents]
        for word in result:
            print('\n', word)
            for n in result[word]:
                #print(sents[n])
                pass

        for n in result[result['word'] == 'querer']:
            print(n)

    def test_lemma_map_from_doc_and_file_agree(self):
        file_name = "corpus_test_2.txt"
        #file_name = 'novelas_cortas.txt'
        with open(file_name, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = wc.es(text)
        lm_text = wc.create_lemma_map_from_doc(doc)
        doc_file, lm_file = wc.create_lemma_map_from_file(file_name)
        self.assertTrue(lm_file.equals(lm_text))

    def test_create_lemma_map_from_file(self):
        directory = 'C:/Users/paulj_1e1uzlz/courses/portilla_nlp_python/'
        file_names = ['la_odisea.txt',
                      'fortunata_y_jacinta.txt',
                      'dona_perfecta.txt',
                      'heath_spanish_am_reader.txt',
                      'novelas_cortas.txt',
                      'corpus_test_2.txt',
                      'fyj_test_1.txt'
                      ]
        doc, lemma_map = wc.create_lemma_map_from_file(file_names[1])

        should_save_files = False
        if should_save_files:
            with open('../fyj/fyj.pkl', 'wb') as doc_pkl:
                pickle.dump(doc, doc_pkl)
            with open('../fyj/fyj_lemma_map.pkl', 'wb') as lm_pkl:
                pickle.dump(lemma_map, lm_pkl)
        print(lemma_map.head())
        sents = [x for x in doc.sents]
        for s in sents:
            print('\n___\n', sents[s])

        for s in list(lemma_map.iloc[1].sentences)[:20]:
            pass
        self.assertTrue(True)

    def test_create_lemma_map_from_multipe_files(self):
        directory = 'C:/Users/paulj_1e1uzlz/courses/portilla_nlp_python/'
        file_names = ['la_odisea.txt',
                      'fortunata_y_jacinta.txt',
                      'dona_perfecta.txt',
                      'heath_spanish_am_reader.txt',
                      'novelas_cortas.txt',
                      ]
        doc, lemma_map = wc.create_lemma_map_from_file(file_names[:2])
        print(lemma_map.head())
        sents = [x for x in doc.sents]
        for s in list(lemma_map.iloc[1].sentences)[:20]:
            print('\n___\n', sents[s])

    def test_spacy(self):
        eslg = spacy.load('es_core_news_lg')
        # wc.show_info(eslg('yo me personé'))
        wc.show_info(eslg("Yo avisaré a otra persona, y vamos a escape, que la muerte nos coge la delantera"))

    def test_create_lemma_map_from_sentences(self):
        en = spacy.load('en_core_web_md')
        doc = en('It is way far away.  There is no way to get there.')
        lu.show_info(doc)
        lm = wc.create_lemma_map_from_sentences(doc.sents)
        print(lm.head())

    def test_create_lemma_map_from_sentences_actual_file(self):
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)
            sentences = [s for s in fyj.sents]
        lemma_map = wc.create_lemma_map_from_sentences(sentences)
        print(lemma_map.head())
        with_dashes = lemma_map[(lemma_map.word.str.startswith('--')) |  (lemma_map.word.str.endswith('--'))]
        self.assertTrue(True)


class CorpusTest(unittest.TestCase):

    def test_clean(self):
        file_name = "corpus_text.txt"
        file_name = "spanishText_345000_350000.txt"
        output_file_name = "spanishText_345000_350000_cleaned.txt"
        text = wc.clean_corpus_file(file_name, output_file_name)
        print(len(text))


class WorkSplittingTest(unittest.TestCase):

    def test_multiple_sentences_lists(self):
        with open('fyj_excerpt_1.txt', encoding='utf-8') as f:
            text_1 = f.read()

        with open('fyj_excerpt_2.txt', encoding='utf-8') as f:
            text_2 = f.read()

        with open('fyj_excerpt_3.txt', encoding='utf-8') as f:
            text_3 = f.read()

        all_text = text_1 + '\n' + text_2 + '\n' + text_3
        doc_all = wc.es(all_text)
        all_sentences = [s for s in doc_all.sents]
        lm_all = wc.create_lemma_map_from_doc(doc_all)

        doc_1 = wc.es(text_1)
        doc_2 = wc.es(text_2)
        doc_3 = wc.es(text_3)
        sentences_1 = [s for s in doc_1.sents]
        sentences_2 = [s for s in doc_2.sents]
        sentences_3 = [s for s in doc_3.sents]
        combined_sentences = sentences_1 + sentences_2 + sentences_3
        lm_combined = wc.create_lemma_map_from_sentences(combined_sentences)

        self.assertTrue(lm_all.equals(lm_combined))

    def test_create_and_serialize_docs(self):
        file_names = ['fortunata_y_jacinta_1.txt', 'fortunata_y_jacinta_2.txt']
        nlp = wc.es
        import os
        try:
            os.remove('fortunata_y_jacinta_1.pkl')
        except Exception as e:
            print(e)
        try:
            os.remove('fortunata_y_jacinta_2.pkl')
        except Exception as e:
            print(e)
        self.assertFalse('fortunata_y_jacinta_1.pkl' in os.listdir())
        self.assertFalse('fortunata_y_jacinta_2.pkl' in os.listdir())
        wc.create_docs_and_serialize_docs(file_names, nlp)
        self.assertTrue('fortunata_y_jacinta_1.pkl' in os.listdir())
        self.assertTrue('fortunata_y_jacinta_2.pkl' in os.listdir())

    def test_create_and_serialize_fyj_docs(self):
        file_names = []
        for i in range(29):
            file_names.append(f'fyj/fortunata_y_jacinta_{i}.txt')
        print(file_names)
        wc.create_docs_and_serialize_docs(file_names, wc.es)

    def test_combine_from_multiple_nlp_docs(self):
        with open('../fyj/fyj_lemma_map.pkl', 'rb') as f:
            lm_from_file = pickle.load(f)
        #lm_from_file = wc.create_lemma_map_from_file('fortunata_y_jacinta.txt')[1]
        print(lm_from_file.head(25))
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)
        sents_from_doc = [s for s in fyj.sents]

        file_names = []
        for i in range(29):
            file_names.append(f'fyj/fortunata_y_jacinta_{i}.pkl')
        all_sentences = []
        for file_name in file_names:
            with open(file_name, 'rb') as f:
                current_doc = pickle.load(f)
                current_sentencs = [s for s in current_doc.sents]
                print(f'{len(current_sentencs)} in {file_name}')
                all_sentences.extend(current_sentencs)
        print(len(all_sentences))
        combined_lm = wc.create_lemma_map_from_sentences(all_sentences)
        print(combined_lm.head(25))
        print(combined_lm.equals(lm_from_file))
