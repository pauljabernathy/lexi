import pandas as pd
import constants
import spacy
import vector_utils as vu
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from functools import partial
#import pickle
#import basic_operations as bo
#import time


def match_n_grams_one_hist_original(tokens:list, ngrams_hist:pd.Series):
    """
    :param tokens: a list of strings
    :param ngrams: a Series that comes from Series.value_counts()
    :return:
    """
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens) + " "
    matches = ngrams_hist[ngrams_hist.gram.str.startswith(text)]   # This line takes some time
    return matches


def match_n_grams_one_hist_2(tokens:list, ngrams_hist:pd.Series):
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens)

    def f(gram_text):
        return gram_text.startswith(text)

    matches2 = ngrams_hist[np.vectorize(f)(ngrams_hist.gram)]
    return matches2


def match_n_grams_one_hist_2_2(tokens:list, ngrams_hist:pd.Series):
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens)

    def f(gram_text):
        return gram_text.startswith(text)

    # matches2 = ngrams_hist[np.vectorize(f)(ngrams_hist.gram)]
    array = np.array(ngrams_hist.gram)
    matches2 = ngrams_hist[np.vectorize(f)(array)]
    return matches2


def find_it(gram_text, search_text):
    return gram_text.startswith(search_text)


def match_n_grams_one_hist_2_3(tokens:list, ngrams_hist:pd.Series):
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens)
    f = partial(find_it, search_text=text)

    # matches2 = ngrams_hist[np.vectorize(f)(ngrams_hist.gram)]
    array = np.array(ngrams_hist.gram)
    matches2 = ngrams_hist[np.vectorize(f)(array)]
    return matches2


def match_n_grams_one_hist_3(tokens:list, ngrams_hist:pd.Series):
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens)

    def f(gram_text):
        return gram_text.startswith(text)

    # matches2 = ngrams_hist[np.vectorize(f)(ngrams_hist.gram)]
    array = np.array(ngrams_hist.gram)
    matches2 = ngrams_hist[np.vectorize(f)(array)]
    return matches2


def match_n_grams_by_index(indices, n_grams_hist):
    matches = n_grams_hist.iloc[indices]
    return matches


def match_n_grams_one_hist_with_array(tokens:list, array:np.ndarray):
    """
    Only passing in the np array.  This won't solve anything because we need the words that are in the pd DataFrame.
    only for performance testing for curiosity
    :param tokens:
    :param array:
    :return:
    """
    #print(tokens)
    text = constants.N_GRAM_SEPARATOR.join(tokens)

    def f(gram_text):
        return gram_text.startswith(text)

    matches2 = array[np.vectorize(f)(array)]
    return matches2


def match_n_grams_one_hist_4(tokens:list, ngrams_hist:pd.Series):
    text = constants.N_GRAM_SEPARATOR.join(tokens)
    matches3 = ngrams_hist[ngrams_hist.gram.apply(lambda g: g.startswith(text))]
    return matches3


def collect_n_grams_matches_original(tokens, list_of_hists):
    results = []
    if type(tokens) == str:
        tokens = tokens.split(" ")
    elif type(tokens) is not list:
        return results  # Do what here?  Return a YoureAnIdiotException.

    for hist in list_of_hists:
        word_length = len(hist.iloc[0][constants.GRAM_COLUMN_NAME].split(" "))
        current_result = match_n_grams_one_hist_original(tokens[-(word_length - 1):], hist)
        results.append(current_result.head(25))
    return results


def collect_n_grams_matches_indices(text, list_of_hists, list_of_prefix_maps):
    results = []
    if type(text) == list:
        text = " ".join(text)
    elif type(text) is not str:
        return results  # Do what here?  Return a YoureAnIdiotException.

    if len(list_of_hists) != len(list_of_prefix_maps):
        return results  # Do what here?  Return a YoureAnIdiotException.

    # Iterate through the list of word stat histograms and the list of prefix maps simultaneously
    for i in range(len(list_of_hists)):
        # word_length = len(hist.iloc[0][constants.GRAM_COLUMN_NAME].split(" "))
        hist = list_of_hists[i]
        prefix_map = list_of_prefix_maps[i]
        word_length = len(hist.iloc[0][constants.GRAM_COLUMN_NAME].split(" "))
        search_text = " ".join(text.split(" ")[-(word_length - 1):])
        if search_text in prefix_map:
            indices = prefix_map[search_text]
        else:
            indices = []
        current_result = match_n_grams_by_index(indices, hist)
        results.append(current_result)
    return results


def collect_word_vector_associations(tokens, matrix):
    """
    Find the word vector associations for the given words.
    :param tokens:
    :param matrix:
    :return:
    """
    closest_vectors_map = {}
    for token in tokens:
        # TODO: Handle the case where the word you are looking at is the same as constants.WORD.  Which should be unlikely.
        if token not in spacy.lang.en.STOP_WORDS:
            closest_word_vectors = vu.find_closest_word_vectors_from_matrix(token, matrix)
            # Try removing all the one with a similarity of 1, as being either the same word, or erroneously close
            closest_word_vectors = closest_word_vectors[closest_word_vectors[constants.SIMILARITY] != 1]
            closest_vectors_map[token] = closest_word_vectors
    non_blank_keys = []
    for key in closest_vectors_map.keys():
        if closest_vectors_map[key].shape[0] > 0:
            non_blank_keys.append(key)
    if len(non_blank_keys) == 0:
        results = pd.DataFrame(columns=[constants.WORD, constants.POS, constants.SIMILARITY])
        results[constants.POS] = matrix[constants.POS]
        results[constants.SIMILARITY] = constants.GENERIC_SIMILARITY
        results[constants.WORD] = matrix.index
        results.index = range(results.shape[0])
    else:
        results = closest_vectors_map[non_blank_keys[0]]
        a = [constants.WORD, constants.POS, non_blank_keys[0]]
        for i in range(1, len(non_blank_keys)):
            if closest_vectors_map[non_blank_keys[i]].shape[0] > 0:
                results = results.merge(closest_vectors_map[non_blank_keys[i]], on=[constants.WORD, constants.POS])
                # I expect the part of speech column to always be the same; using that as a merge criteria so it will
                # maintain on POS column for the whole thing instead of one for each part joined.
                a.append(non_blank_keys[i])
        results.columns = a

    results[constants.SUM_COLUMN_NAME] = 0
    for column in non_blank_keys:
        results[constants.SUM_COLUMN_NAME] += results[column]
    results[constants.PRODUCT_COLUMN_NAME] = 1
    for column in non_blank_keys:
        results[constants.PRODUCT_COLUMN_NAME] *= results[column]
    results[constants.SUM_SQ_COLUMN_NAME] = 0
    for column in non_blank_keys:
        results[constants.SUM_SQ_COLUMN_NAME] += (results[column]) ** 2
    results[constants.PRODUCT_SQ_COLUMN_NAME] = 1
    for column in non_blank_keys:
        results[constants.PRODUCT_SQ_COLUMN_NAME] *= ((results[column]) ** 2)
    return results


def get_top_results(all_associations_df, nlp, top_number, pos="NOUN", sort_column=constants.SUM_SQ_COLUMN_NAME):
    """
    Grab the top results for the word vector matrix.
    It also takes into account the part of speech, using the idea that you usually want a particular part of speech
    (a noun or a verb) as the most likely match.
    :param all_associations_df:
    :param nlp: the spacy object; used for detecting the part of speech of the word
    :param top_number: how many results you want
    :param pos: part of speech
    :param sort_column: which of the score aggregation columns to sort on
    :return: the top results; the columns should all be the same as all_associations_df; length should be top_number,
    or fewer if there were fewer than top_number rows to begin with
    """
    all_associations_df = all_associations_df.sort_values(sort_column, ascending=False)
    if constants.POS not in all_associations_df:
        all_associations_df[constants.POS] = all_associations_df[constants.WORD].apply(lambda w: nlp(w)[0].pos_)
    all_associations_df = all_associations_df[all_associations_df[constants.POS] == pos]
    # all_associations_df = all_associations_df.drop(['pos'], axis=1)
    top_results = all_associations_df.iloc[0:top_number]
    return top_results


def predict_from_word_vectors(tokens, word_list, spacy_vocab, nlp, POS="NOUN", top_number=constants.DEFAULT_TOP_NGRAMS):
    """
    Make a prediction based on the word vectors.
    deprecated in favor of predict_from_word_vectors_matrix
    :param tokens: a list of words, the input phrase
    :param word_list: a list of all words
    :param spacy_vocab: the vocabulary of all words in the spacy object
    :param nlp: the spacy language object
    :param POS:
    :param top_number:
    :return:
    """
    closest_vectors_map = {}
    for token in tokens:
        if token not in spacy.lang.en.STOP_WORDS:
            closest_word_vectors = vu.find_closest_word_vectors(token, word_list, nlp)
            # Try removing all the one with a similarity of 1, as being either the same word, or erroneously close
            closest_word_vectors = closest_word_vectors[closest_word_vectors[constants.SIMILARITY] != 1]
            #print(closest_word_vectors.head(10))
            closest_vectors_map[token] = closest_word_vectors
    # At this point, we need to do something with all the dfs of word vectors.
    m = closest_vectors_map
    keys = list(m.keys())
    r = m[keys[0]]
    for i in range(1, len(m.keys())):
        r = r.merge(m[keys[i]], on=constants.WORD)
    a = [constants.WORD]
    a.extend(keys)
    r.columns = a
    r[constants.SUM_COLUMN_NAME] = 0
    for column in keys:
        r[constants.SUM_COLUMN_NAME] += r[column]
    r[constants.PRODUCT_COLUMN_NAME] = 1
    for column in keys:
        r[constants.PRODUCT_COLUMN_NAME] *= r[column]
    r[constants.SUM_SQ_COLUMN_NAME] = 0
    for column in keys:
        r[constants.SUM_SQ_COLUMN_NAME] += (r[column]) ** 2
    r[constants.PRODUCT_SQ_COLUMN_NAME] = 1
    for column in keys:
        r[constants.PRODUCT_SQ_COLUMN_NAME] *= ((r[column]) ** 2)

    top_results = get_top_results(r, nlp, top_number, POS)
    return top_results


def predict_from_word_vectors_matrix(tokens, matrix, nlp, POS="NOUN", top_number=constants.DEFAULT_TOP_ASSOCIATIONS):
    """
    Make a prediction based on the word vectors
    :param tokens:
    :param matrix:
    :param nlp:
    :param POS:
    :param top_number:
    :return:
    """
    vector_results = collect_word_vector_associations(tokens, matrix)
    top_results = get_top_results(vector_results, nlp, top_number, POS)
    return top_results


# TODO:  A parameter for max results. ng_result = ngram_matches[i].head(max_results); straightforward enough but will
#  need some regression
def predict(phrase, list_of_hists, list_of_prefix_maps, matrix_df, nlp, POS="NOUN", threshold=5):
    """
    Make a predictino of the next word
    :param phrase: the input words (a sentence fragment) you are trying to predict what will come next
    :param list_of_hists: a list of ngram histograms
    :param list_of_prefix_maps: a list of prefix maps, that map phrases to places in the ngram hist that they appear
    :param matrix_df: word vectors DataFrame
    :param nlp: the spacy object
    :param POS: the default part of speech that all results should match
    :param threshold: the minimum threshold for the ngram "backoff" algorithm
    :return: a DataFrame, with a list of words that could match, in order of most likely to least likely
    """
    ngram_matches = collect_n_grams_matches_indices(phrase, list_of_hists, list_of_prefix_maps)

    # TODO: Use the choose_best_n_gram() function.
    ng_result = None  # Find a way to default it to the overall word frequency
    for i in [x for x in range(len(ngram_matches))][::-1]:
        if ngram_matches[i][constants.COUNT_COLUMN_NAME].sum() > threshold:
            ng_result = ngram_matches[i]#.head(5)
            break

    if ng_result is not None:
        result = ng_result[[constants.TARGET]].rename(columns={constants.TARGET: constants.PREDICTION})
        result[constants.RESULT_TYPE] = constants.NGRAM
    else:
        tokens = phrase.split(" ")
        # print(tokens)
        association_results = predict_from_word_vectors_matrix(tokens, matrix_df, nlp, POS)
        result = association_results[[constants.WORD]].rename(columns={constants.WORD: constants.PREDICTION})
        result[constants.RESULT_TYPE] = constants.VECTOR
    return result


def choose_best_n_gram(results_list):
    pass


def do_predictions():
    en = spacy.load('en_core_web_md')
    vocab = list(en.vocab.strings)
    en_words = set(v.lower() for v in vocab)
    queries = ["and a case of", "it would mean the", "can you follow me and make me the",
               "offense still struggling but the", "go on a romantic date at the", "ill dust them off and be on my",
               "and havent seen it in quite some", "out of his eyes with his little", "and keep the faith during the",
               " then you must be"]
    queries = ["offense still struggling but the", "go on a romantic date at the",
               "out of his eyes with his little", "and keep the faith during the",
               " then you must be"]

    queries = ["id live and id", "started telling me about his"]
    queries = ["helps reduce your"]

    '''hist4 = load_hist("twitter_train.txt_4_grams.csv")
    hist5 = load_hist("twitter_train.txt_5_grams.csv")
    hist6 = load_hist("twitter_train.txt_6_grams.csv")

    hist4 = load_hist("twitter_sample.txt_4_grams.csv")
    hist5 = load_hist("twitter_sample.txt_5_grams.csv")
    hist6 = load_hist("twitter_sample.txt_6_grams.csv")'''

    hist2 = load_hist("en_US.twitter.txt_2_grams.csv")
    hist3 = load_hist("en_US.twitter.txt_3_grams.csv")
    hist4 = load_hist("en_US.twitter.txt_4_grams.csv")
    hist5 = load_hist("en_US.twitter.txt_5_grams.csv")
    hist6 = load_hist("en_US.twitter.txt_6_grams.csv")
    hist_list = [hist2, hist3, hist4, hist5, hist6]

    #queries = [queries[5]]
    for string in queries:
        tokens = string.split(" ")
        '''if len(tokens) >= 4:
            four_result = predict_from_n_grams_one_hist(tokens[:3], hist4)
            print(four_result)
        if len(tokens) >= 5:
            five_result = predict_from_n_grams_one_hist(tokens[:4], hist5)
            print(five_result)
        if len(tokens) >= 6:
            six_result = predict_from_n_grams_one_hist(tokens[:5], hist6)
            print(six_result)'''
        collect_n_grams_matches_original(tokens, hist_list)


def load_hist(file_name):
    """
    Load a histogram DataFrame
    not sure that we rrreeeaaaaalllllllyyyyy need this function.
    :param file_name: path and name of the .csv file that pandas can load
    :return: a pandas DataFrame
    """
    df = pd.read_csv(file_name)
    return df


