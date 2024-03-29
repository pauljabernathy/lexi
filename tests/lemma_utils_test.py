import unittest
import spacy
import pickle
import lemma_utils as lu
import time
import word_count as wc


class ShowExamplesTest(unittest.TestCase):

    def setUp(self):
        # self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_lg')

    def test_get_example_real_file(self):
        with open("../fyj/fyj_sentences.pkl", 'rb') as f:
            sentences = pickle.load(f)

        with open("../fyj/fyj_lemma_map.pkl", 'rb') as f:
            fyj_lm = pickle.load(f)

        examples = lu.get_examples(sentences, fyj_lm, 'alejar', 20)
        for i in range(len(examples)):
            print(f'{i}: {examples[i]}')

    def test_filter_sentences_with_lemma(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.' \
               'Se habla español.  Tengo ganas de escribir una programa.'
        doc = self.es(text)
        all_sentences = list(doc.sents)
        lemma_map = wc.create_lemma_map_from_sentences(all_sentences)
        sentences_with_lemma = lu.filter_sentences_with_lemma(all_sentences, lemma_map, 'tener')
        sentences_with_lemma = [str(s).strip(" ") for s in sentences_with_lemma]
        self.assertEqual(4, len(sentences_with_lemma))
        self.assertEqual(['Que tengas buen día.', 'Tienes un buen día.', 'Espero que tenga mucho dinero.',
                          'Tengo ganas de escribir una programa.'], sentences_with_lemma)

    def test_get_examples(self):
        doc1 = self.es('No quiero taco bell.  Nosotros queremos taco bell.  Ellos quieren taco bell.')
        doc2 = self.es('No quiero taco bell.  Ellos no quieren taco bell.  Él quiere taco bell.  Ustedes quieren taco '
                  'bell.  Tú no quieres taco bell.  Nosotros queremos taco bell.')

        doc3 = self.es('Él quiere tacos.  Sin embargo, no los quiero.  Yo prefiero hamburguesas.  Siempre queremos '
                  'hamburguesas cada fin de semana.')
        lm1 = wc.create_lemma_map_from_doc(doc1)
        lm2 = wc.create_lemma_map_from_doc(doc2)
        lm3 = wc.create_lemma_map_from_doc(doc3)

        sentences_1 = list(doc1.sents)
        result1 = lu.get_examples(sentences_1, lm1, "querer")
        self.assertEquals(3, len(result1))
        self.assertEqual("No quiero taco bell.", result1[0])
        self.assertEqual("Nosotros queremos taco bell.", result1[1])
        self.assertEqual("Ellos quieren taco bell.", result1[2])

        result1 = lu.get_examples(sentences_1, lm1, "no")
        self.assertEquals(1, len(result1))
        self.assertEqual("No quiero taco bell.", result1[0])

        sentences_2 = list(doc2.sents)
        result2 = lu.get_examples(sentences_2, lm2, "ellos")  # Spacy classifies the lemma for "ellos" as "él",
        # not "ellos"
        self.assertEquals(0, len(result2))

        result2 = lu.get_examples(sentences_2, lm2, "él")
        self.assertEquals(2, len(result2))
        self.assertEqual("Ellos no quieren taco bell.", result2[0])
        self.assertEqual("Él quiere taco bell.", result2[1])

        sentences_3 = list(doc3.sents)
        result3 = lu.get_examples(sentences_3, lm3, "preferir")
        self.assertEquals(1, len(result3))
        self.assertEqual("Yo prefiero hamburguesas.", result3[0])

    def test_show_examples_multiple_docs(self):
        doc1 = self.es('No quiero taco bell.  Nosotros queremos taco bell.  Ellos quieren taco bell.')
        doc2 = self.es('No quiero taco bell.  Ellos no quieren taco bell.  Él quiere taco bell.  Ustedes quieren taco '
                     'bell.  Tú no quieres taco bell.  Nosotros queremos taco bell.')

        doc3 = self.es('Él quiere tacos.  Sin embargo, no los quiero.  Yo prefiero hamburguesas.  Siempre queremos '
                     'hamburguesas cada fin de semana.')
        lm1 = wc.create_lemma_map_from_doc(doc1)
        lm2 = wc.create_lemma_map_from_doc(doc2)
        lm3 = wc.create_lemma_map_from_doc(doc3)

        pairs = [(doc1, lm1), (doc2, lm2), (doc3, lm3)]
        él = lu.get_examples_multiple_docs(pairs, "él")
        # self.assertEqual(4, len(él))
        # For the sentence "Sin embargo, no los quiero.", Spacy classifies "los" as a variant of "él".
        # los          PRON   Case=Acc|Gender=Masc|Number=Plur|Person=3|PrepCase=Npr|PronType=Prs "lemma:"  él
        # But if you do "Sin embargo, no quiero los tacos." it classifies "los" as a variant of "el"
        # "los          DET    Definite=Def|Gender=Masc|Number=Plur|PronType=Art "lemma:"  el"
        # Not what I initially expected, but maybe it is correct.
        self.assertEquals("Ellos quieren taco bell.", él[0])
        self.assertEquals("Ellos no quieren taco bell.", él[1])
        self.assertEquals("Él quiere taco bell.", él[2])
        self.assertEquals("Él quiere tacos.", él[3])

        yo = lu.get_examples_multiple_docs(pairs, "yo")  # The lemma for nosotros is "yo".
        self.assertEqual(3, len(yo))
        self.assertEqual("Nosotros queremos taco bell.", yo[0])
        self.assertEqual("Nosotros queremos taco bell.", yo[1])
        self.assertEqual("Yo prefiero hamburguesas.", yo[2])

        preferir = lu.get_examples_multiple_docs(pairs, "preferir")
        self.assertEqual(1, len(preferir))
        self.assertEqual("Yo prefiero hamburguesas.", preferir[-1])

    def test_show_examples_multiple_docs_real_files(self):
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_1.pkl", 'rb') as f:
            cp_doc_1 = pickle.load(f)
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_1_lm.pkl", 'rb') as f:
            cp_lm_1 = pickle.load(f)
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_2.pkl", 'rb') as f:
            cp_doc_2 = pickle.load(f)
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_2_lm.pkl", 'rb') as f:
            cp_lm_2 = pickle.load(f)
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_3.pkl", 'rb') as f:
            cp_doc_3 = pickle.load(f)
        with open("../../../courses/portilla_nlp_python/corpus/spanishText_345000_350000_3_lm.pkl", 'rb') as f:
            cp_lm_3 = pickle.load(f)
        pairs = [(cp_doc_1, cp_lm_1), (cp_doc_2, cp_lm_2), (cp_doc_3, cp_lm_3)]
        lu.show_examples_multiple_docs(pairs, 'penumbra')


class ShowInfoTest(unittest.TestCase):

    def setUp(self):
        self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_lg')

    def test_show_info(self):
        sentence = 'The food we had yesterday was delicious'
        sentence = "I'll go away sometime."
        lu.show_info(self.en(sentence))

    def test_show_info_es(self):
        sentence = 'La comida que comimos ayer era deliciosa.'
        self.es = spacy.load('es_core_news_lg')
        lu.show_info(self.es(sentence))
        lu.show_info(self.es("La comida que comimos ayer estuvo deliciosa.")) # GT says "estuvo" not "era".

        sentence = 'No quiero que te vayas.'
        #sentence = 'Que tengas buen día.'
        self.es = spacy.load('es_core_news_lg')
        lu.show_info(self.es(sentence))
        lu.show_info(self.es("No quiero bailar."))
        lu.show_info(self.es("No quiero que tu bailes."))
        lu.show_info(self.es("Estoy bailando."))
        lu.show_info(self.es("Si tu cantas yo bailaré."))
        lu.show_info(self.es("Si tu cantaras yo bailaría."))
        lu.show_info(self.es("Yo bailé ayer."))
        lu.show_info(self.es("Si fuera más joven, estudiaría química."))

        lu.show_info(self.es('Que tengas buen día.'))
        lu.show_info(self.es('Tienes un buen día.'))
        lu.show_info(self.es('Espero que tenga mucho dinero.'))

    def test_show_info_for_sentence_obj_not_doc(self):
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)
        sents = [s for s in fyj.sents]
        sents2 = list(fyj.sents)
        print(sents == sents2)
        lu.show_info(sents[0])


class SearchForExamplesTest(unittest.TestCase):

    def setUp(self):
        self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_lg')

    def test_search_doc_for_examples(self):
        doc = self.es('Que tengas buen día.  Tienes un buen día.  Espero que tenga mucho dinero.')
        examples = lu.search_doc_for_examples(doc, 'tener', {'Mood': 'Sub'})
        print(examples)
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

    def test_search_doc_for_examples_file(self):
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)

        alejar_fin = lu.search_doc_for_examples(fyj, 'alejar', {'VerbForm': 'Fin'})
        print(len(alejar_fin))
        for x in alejar_fin:
            print(x)

        alejar_inf = lu.search_doc_for_examples(fyj, 'alejar', {'VerbForm': 'Inf'})
        print(len(alejar_inf))
        for x in alejar_inf:
            print(x)

        alejar_part = lu.search_doc_for_examples(fyj, 'alejar', {'VerbForm': 'Part'})
        print(len(alejar_part))
        for x in alejar_part:
            print(x)

        alejar_ger = lu.search_doc_for_examples(fyj, 'alejar', {'VerbForm': 'Get'})
        print(len(alejar_ger))
        for x in alejar_ger:
            print(x)

        alejar_ger = lu.search_doc_for_examples(fyj, 'alejar', {'VerbForm': 'Ger'})
        print(len(alejar_ger))
        for x in alejar_ger:
            print(x)

    def test_search_doc_for_examples_multiple_attrs(self):
        # If you have more than one attribute that matches the lemma, it should only return it once
        doc = self.es('Que tengas buen día.')
        lu.show_info(doc)
        examples = lu.search_doc_for_examples(doc, 'tener', {'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(1, len(examples))

        # It should match all the attrs, not just one.
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        examples = lu.search_doc_for_examples(doc, 'tener', {'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(2, len(examples))

    def test_search_doc_for_examples_blank_attrs(self):
        # should return all examples of that lemma
        doc = self.es('Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.')
        examples = lu.search_doc_for_examples(doc, 'tener', morphology_attributes={})
        print(examples)
        self.assertEqual(3, len(examples))

    def test_search_doc_for_examples_None_attrs(self):
        # should return all examples of that lemma
        doc = self.es('Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.')
        examples = lu.search_doc_for_examples(doc, 'tener', morphology_attributes=None)
        print(examples)
        self.assertEqual(3, len(examples))

    def test_search_sentences_for_examples(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        sentences = list(doc.sents)
        lm = wc.create_lemma_map_from_sentences(sentences)
        examples = lu.search_sentences_for_examples(sentences, lm, 'tener', {'Mood': 'Sub', 'Number': 'Sing'})
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

        # With additional attrs that would not work
        examples = lu.search_sentences_for_examples(sentences, lm, 'tener', {'Mood': 'Sub', 'Number': 'Sing',
                                                                             'something': 'else'})
        self.assertEqual(0, len(examples))
        self.assertEqual([], examples)

    def test_search_sentences_for_examples_blank_attrs(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        sentences = list(doc.sents)
        lm = wc.create_lemma_map_from_sentences(sentences)
        examples = lu.search_sentences_for_examples(sentences, lm, 'tener', morphology_attributes={})
        self.assertEqual(3, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Tienes un buen día.', 'Espero que tenga mucho dinero.'], examples)

    def test_search_sentences_for_examples_None_attrs(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        sentences = list(doc.sents)
        lm = wc.create_lemma_map_from_sentences(sentences)
        examples = lu.search_sentences_for_examples(sentences, lm, 'tener', morphology_attributes=None)
        self.assertEqual(3, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Tienes un buen día.', 'Espero que tenga mucho dinero.'], examples)

    def test_search_doc_for_examples_attrs_only(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        examples = lu.search_doc_for_examples(doc, lemma=None, morphology_attributes={'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

        examples = lu.search_doc_for_examples(doc, lemma='', morphology_attributes={'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

    def test_search_sentences_for_examples_attrs_only(self):
        text = 'Que tengas buen día.  Tienes un buen día.  No me gusta bailar.  Espero que tenga mucho dinero.'
        doc = self.es(text)
        sentences = list(doc.sents)
        lm = wc.create_lemma_map_from_sentences(sentences)
        examples = lu.search_sentences_for_examples(sentences, lm, lemma=None,
                                                    morphology_attributes={'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

        examples = lu.search_sentences_for_examples(sentences, lm, lemma='',
                                                    morphology_attributes={'Mood': 'Sub', 'Number': 'Sing'})
        print(examples)
        self.assertEqual(2, len(examples))
        self.assertEqual(['Que tengas buen día.', 'Espero que tenga mucho dinero.'], examples)

    def test_search_sentences_for_examples_real_file(self):
        #with open("fyj/fyj_sentences.pkl", 'rb') as f:
        #    sentences = pickle.load(f)
        with open("../fyj/fyj_lemma_map.pkl", 'rb') as f:
            fyj_lm = pickle.load(f)

        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)

        sentences = list(fyj.sents)
        examples = lu.search_sentences_for_examples(sentences, fyj_lm, lemma=None,
                                                    morphology_attributes={'Mood': 'Sub', 'Number': 'Sing'})
        print(len(examples))
        # print(examples)
        #for i in range(len(examples)):
        #    print(i, examples[i])

        #examples = lu.search_sentences_for_examples(sentences, fyj_lm, lemma='hablar',
        #                                            morphology_attributes= {'Mood': 'Sub', 'Number': 'Sing'})
        examples = lu.search_sentences_for_examples(sentences, fyj_lm, lemma='tener',
                                                    morphology_attributes= {'Mood': 'Sub', 'VerbForm': 'Fin'})
        examples2 = lu.search_sentences_for_examples(sentences, fyj_lm, lemma='tener',
                                                    morphology_attributes= {'VerbForm': 'Fin'})
        print(examples)

    def test_match_all(self):
        text = "Que tengas buen día."
        lu.show_info(self.es('Que tengas buen día.'))
        lu.show_info(self.es('Que tengan buen día.'))
        doc = self.es(text)
        token = doc[1]
        did_it_match = lu.match_all_attrs(token, {'Mood': 'Sub', 'Number': 'Plur'})
        self.assertFalse(did_it_match)
        self.assertFalse(lu.match_all_attrs(doc[1], {'Mood': 'Sub', 'Number': 'Plur'}))
        self.assertTrue(lu.match_all_attrs(doc[1], {'Mood': 'Sub', 'Number': 'Sing'}))
        self.assertFalse(lu.match_all_attrs(doc[1], {'Mood': 'Sub', 'Number': 'Sing', 'something': 'else'}))

        self.assertFalse(lu.match_all_attrs(doc[2], {'Gender': 'Masc', 'Number': 'Sing', 'VerbForm': 'Fin'}))
        self.assertFalse(lu.match_all_attrs(doc[2], {'VerbForm': 'Fin'}))
        self.assertTrue(lu.match_all_attrs(doc[2], {'Gender': 'Masc', 'Number': 'Sing'}))

        '''self.assertFalse(lu.match_all_attrs(doc[0], {}))
        self.assertFalse(lu.match_all_attrs(doc[1], {}))
        self.assertFalse(lu.match_all_attrs(doc[2], {}))
        self.assertFalse(lu.match_all_attrs(doc[3], {}))
        
        self.assertFalse(lu.match_all_attrs(doc[0], None))
        self.assertFalse(lu.match_all_attrs(doc[1], None))
        self.assertFalse(lu.match_all_attrs(doc[2], None))
        self.assertFalse(lu.match_all_attrs(doc[3], None))'''

        # At this time, we are saying that it returns true for an empty dict.
        self.assertTrue(lu.match_all_attrs(doc[0], {}))
        self.assertTrue(lu.match_all_attrs(doc[1], {}))
        self.assertTrue(lu.match_all_attrs(doc[2], {}))
        self.assertTrue(lu.match_all_attrs(doc[3], {}))

        self.assertTrue(lu.match_all_attrs(doc[0], None))
        self.assertTrue(lu.match_all_attrs(doc[1], None))
        self.assertTrue(lu.match_all_attrs(doc[2], None))
        self.assertTrue(lu.match_all_attrs(doc[3], None))

    def test_match_any(self):
        text = "Que tengas buen día."
        doc = self.es(text)
        token = doc[1]
        did_it_match = lu.match_any_attrs(token, {'Mood': 'Sub', 'Number': 'Plur'})
        self.assertTrue(did_it_match)
        self.assertFalse(lu.match_any_attrs(doc[2], {'Mood': 'Sub', 'Number': 'Plur'}))

        self.assertTrue(lu.match_any_attrs(doc[2], {'Gender': 'Masc', 'Number': 'Sing', 'VerbForm': 'Fin'}))
        self.assertFalse(lu.match_any_attrs(doc[2], {'VerbForm': 'Fin'}))

        self.assertFalse(lu.match_any_attrs(doc[0], {}))
        self.assertFalse(lu.match_any_attrs(doc[1], {}))
        self.assertFalse(lu.match_any_attrs(doc[2], {}))
        self.assertFalse(lu.match_any_attrs(doc[3], {}))

        self.assertFalse(lu.match_any_attrs(doc[0], None))
        self.assertFalse(lu.match_any_attrs(doc[1], None))
        self.assertFalse(lu.match_any_attrs(doc[2], None))
        self.assertFalse(lu.match_any_attrs(doc[3], None))


class VerbFormHistTest(unittest.TestCase):

    def setUp(self):
        # self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_lg')

    def test_verb_forms_hist(self):
        doc = self.es('Que tengas buen día.  Tienes un buen día.  Espero que tenga mucho dinero.')
        doc = self.es('No quiero bailar.  No quiero que tu bailes.  Estoy bailando.')
        hist = lu.verb_form_hist(doc)
        print(hist)
        self.assertEqual({"Fin": 3, "Inf": 1, 'Ger': 1}, hist)


class CompareMapAndAdHoc(unittest.TestCase):

    def setUp(self):
        # self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_lg')

    # TODO:  Fix this test failure for hablar.
    def test_compare_results_for_fyj(self):
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)
        ad_hoc_examples = lu.search_doc_for_examples(fyj, 'alejar', None)
        #ad_hoc_examples = lu.search_doc_for_examples(fyj, 'resolver', None)
        print('ad hoc')
        print(ad_hoc_examples)

        with open("../fyj/fyj_sentences.pkl", 'rb') as f:
            sentences = pickle.load(f)

        with open("../fyj/fyj_lemma_map.pkl", 'rb') as f:
            fyj_lm = pickle.load(f)

        map_examples = lu.get_examples(sentences, fyj_lm, 'alejar', 2000)
        #map_examples = lu.get_examples(sentences, fyj_lm, 'resolver', 2000)
        print('map')
        print(map_examples)

        #self.assertEqual(len(ad_hoc_examples), len(map_examples))
        self.assertEqual(sorted(ad_hoc_examples), sorted(map_examples))

        word = 'hablar'
        ad_hoc_examples = lu.search_doc_for_examples(fyj, word, None)[:2000]
        map_examples = lu.get_examples(sentences, fyj_lm, word, 2000)

        words = ['tener', 'cocinar', 'hablar']
        for word in words:
            print(word)
            ad_hoc_examples = lu.search_doc_for_examples(fyj, word, None)[:2001]
            map_examples = lu.get_examples(sentences, fyj_lm, word, 2000)
            if word is 'hablar':
                print(ad_hoc_examples)
                print(map_examples)
                self.assertEquals(sorted(map_examples), sorted(ad_hoc_examples))
            self.assertEqual(len(map_examples), len(ad_hoc_examples))    # Fails here

    def test_for_replication(self):
        sentence1 = '--¿Qué le digo?... Porque aunque no le he hablado nunca, le hablaré, si usted me lo manda.'
        sentence2 = 'Creyó ver a Segunda y oírla hablar con Encarnación; pero hablaban a la carrera, como seres endemoniados, pasando y perdiéndose en un término vago que caía hacia la mano derecha.'

        doc1 = self.es(sentence1)
        doc2 = self.es(sentence2)
        s1 = list(doc1.sents)
        s2 = list(doc2.sents)
        lm_1 = wc.create_lemma_map_from_sentences(s1)
        lm_2 = wc.create_lemma_map_from_sentences(s2)
        map_examples1 = lu.get_examples(s1, lm_1, 'hablar', None)
        map_examples2 = lu.get_examples(s2, lm_2, 'hablar', None)

        ad_hoc_examples_1 = lu.search_doc_for_examples(doc1, 'hablar', None)
        ad_hoc_examples_2 = lu.search_doc_for_examples(doc2, 'hablar', None)
        bp = 'bp'


    # TODO:  Move somewhere else because this isn't really a unit test.
    @unittest.skip("Only do this ad hoc because of how long it takes.")
    def test_compare_time_fyj(self):
        with open('../fyj/fyj.pkl', 'rb') as f:
            fyj = pickle.load(f)

        with open("../fyj/fyj_sentences.pkl", 'rb') as f:
            sentences = pickle.load(f)
        with open("../fyj/fyj_lemma_map.pkl", 'rb') as f:
            fyj_lm = pickle.load(f)

        print('a list of words')
        words = ['tener', 'cocinar', 'hablar', 'no', 'el', 'él', 'y', 'comer', 'encontrar']# * 10
        self.compare_time(words, fyj, sentences, fyj_lm)
        self.compare_time(words * 2, fyj, sentences, fyj_lm)
        self.compare_time(words * 4, fyj, sentences, fyj_lm)
        self.compare_time(words * 8, fyj, sentences, fyj_lm)
        self.compare_time(words * 16, fyj, sentences, fyj_lm)

        # on second thought, that might take all day
        #print('now all of them')
        #self.compare_time(fyj_lm['words'])

    def compare_time(self, words, doc, sentences, lemma_map):
        print(len(words), ' words')
        start_ad_hoc = time.time()
        for word in words:
            ad_hoc_examples = lu.search_doc_for_examples(doc, word, None)
        end_ad_hoc = time.time()
        ad_hoc_time_taken = end_ad_hoc - start_ad_hoc
        print(ad_hoc_time_taken)

        start_map = time.time()
        for word in words:
            map_examples = lu.get_examples(sentences, lemma_map, word, 2000)
        end_map = time.time()
        map_time_taken = end_map - start_map
        print(map_time_taken)
        print(ad_hoc_time_taken > map_time_taken)


class Experiments(unittest.TestCase):

    def setUp(self):
        # self.en = spacy.load('en_core_web_lg')
        self.es = spacy.load('es_core_news_md')
        self.en = spacy.load('en_core_web_md')
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)  # needed so it doesn't show ... for POS column
        pd.set_option('display.width', 10000)

    def test_spacy_doc_similarity(self):
        doc1 = self.es('Él come tacos.')
        lu.show_info(doc1)
        doc2 = self.es('Nosotros comemos tacos.')
        lu.show_info(doc2)
        '''similarity12 = doc1.similarity(doc2)
        similarity21 = doc2.similarity(doc1)
        print(similarity12)
        print(similarity21)'''
        doc3 = self.es('Ellos comen tacos.')
        lu.show_info(doc3)
        doc4 = self.es('Ellas comen tacos.')
        lu.show_info(doc4)
        docs = [doc1, doc2, doc3, doc4]
        for i in range(len(docs)):
            for j in range(0, i):
                print(i + 1, j + 1, docs[i].similarity(docs[j]))

    def test_pipeline_excludes(self):
        frase = 'Que tengas buen dia.'
        d0 = self.es(frase)
        lu.show_info(d0)

        # The spanish pipelines do not come with a tagger, so excluding tagger shouldn't matter
        d1 = [d for d in self.es.pipe([frase], disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])][0]
        lu.show_info(d1)
        d2 = [d for d in self.es.pipe([frase], disable=["parser", "attribute_ruler", "lemmatizer"])][0]
        lu.show_info(d2)
        # self.assertEqual(d1, d2)
        d3 = [d for d in self.es.pipe([frase], disable=["parser", "attribute_ruler"])][0]
        lu.show_info(d3)

        d4 = [d for d in self.es.pipe([frase], disable=["attribute_ruler"])][0]
        lu.show_info(d4)

        # pos is present in all of the above.  Where does it come from?

        d5 = [d for d in self.es.pipe([frase], disable=["morphologizer"]) ] [0]
        # morph is missing, as expected, as is pos
        # => pos comes from the morphologizer step

        lu.show_info(d5)

        '''
        parser -> token.dep_   parser is "Dependency Parser"  https://spacy.io/api/dependencyparser
            What does the dependency parser actually do? 
        tagger -> not relevant for Spanish
        attribute_ruler -> ?  Doesn't seem to be something I am interested in right now.
        morphologizer -> morph dict and pos_
        '''

    def test_pipeline_excludes_english(self):
        phrase = 'Have a nice day.'
        en0 = self.en(phrase)
        lu.show_info(en0)

        # The English pipelines do not have a morphologizer.  https://spacy.io/models/en
        # But it has the information like {'Definite': 'Ind', 'PronType': 'Art'} that comes from token.morph.to_dict(
        # ).  How?  Seems to come from the tagger.
        # Spanish: morph comes from the morphologizer step; contains no tagger step.
        # English: morph comes from the tagger step
        en1 = [d for d in self.en.pipe([phrase], disable=["tagger"]) ] [0]
        lu.show_info(en1)
        # missing morph and pos

        en2 = [d for d in self.en.pipe([phrase], disable=["parser"]) ] [0]
        lu.show_info(en2)
        # missing dep

        en3 = [d for d in self.en.pipe([phrase], disable=["lemmatizer"]) ] [0]
        lu.show_info(en3)
        # missing lemmas, as you can imagine

        en4 = [d for d in self.en.pipe([phrase], disable=["morphologizer"]) ] [0]
        lu.show_info(en4)
        # same as en3

        '''
        tagger -> pos and morph
        parser -> token.dep_
        
        '''

    '''
    In English, the "tagger" step does the morph and the pos.
    In Spanish, those are done by the "morphologizer"
    =>  English tagger  <--> Spanish morphologizer, at least roughly
    '''

    def test_debugging(self):
        text = "Ver D. Bonifacio las primeras muestras del estilo de Senquá y chiflarse por completo, fue todo uno"
        #lu.show_info(self.es(text))
        #lu.show_info(self.es('una cosa mas'))
        #lu.show_info(self.es('tres cosas mas'))
        lu.show_info(self.en('I run fast.  That is the first run of the program, this is the second run.'))
        lu.show_info(self.en('I run fast.  That is the first run of, this is the second run.'))
        lu.show_info(self.es(' Does the Whale’s Magnitude Diminish?—Will He Perish?'))
        lu.show_info(self.es("Nosotros queremos taco bell"))  # nosotros -> yo

    def test_examine_dep_parser(self):
        text = "John ate eggs and  Mary ate potatoes"  # If there are double spaces, it will classify this as two
        # sentences.
        text = "I am golden."
        lu.show_info(self.en(text))


if __name__ == '__main__':
    unittest.main()
