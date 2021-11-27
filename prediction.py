import pandas as pd
import constants
import spacy
import vector_utils as vu
import numpy as np
from functools import partial


def match_n_grams_one_hist(tokens:list, ngrams_hist:pd.Series):
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


def collect_n_grams_matches(tokens, list_of_hists):
    results = []
    if type(tokens) == str:
        tokens = tokens.split(" ")
    elif type(tokens) is not list:
        return results  # Do what here?  Return a YoureAnIdiotException.

    for hist in list_of_hists:
        word_length = len(hist.iloc[0][constants.GRAM_COLUMN_NAME].split(" "))
        current_result = match_n_grams_one_hist(tokens[-(word_length - 1):], hist)
        print(current_result.head(25))
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
        indices = prefix_map[text]
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
        if token not in spacy.lang.en.STOP_WORDS:
            closest_word_vectors = vu.find_closest_word_vectors_from_matrix(token, matrix)
            # Try removing all the one with a similarity of 1, as being either the same word, or erroneously close
            closest_word_vectors = closest_word_vectors[closest_word_vectors['similarity'] != 1]
            #print(closest_word_vectors.head(10))
            closest_vectors_map[token] = closest_word_vectors
    #return closest_vectors_map
    m = closest_vectors_map
    keys = list(m.keys())
    results = m[keys[0]]
    for i in range(1, len(m.keys())):
        results = results.merge(m[keys[i]], on="word")
    a = ['word']
    a.extend(keys)
    results.columns = a
    results['sum'] = 0
    for column in keys:
        results['sum'] += results[column]
    results['product'] = 1
    for column in keys:
        results['product'] *= results[column]
    results['sum_sq'] = 0
    for column in keys:
        results['sum_sq'] += (results[column]) ** 2
    results['prd_sq'] = 1
    for column in keys:
        results['prd_sq'] *= ((results[column]) ** 2)
    return results


def get_top_results(all_associations_df, nlp, top_number, pos="NOUN", sort_column='sum_sq'):
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
    all_associations_df['pos'] = all_associations_df[constants.WORD].apply(lambda w: nlp(w)[0].pos_)
    all_associations_df = all_associations_df[all_associations_df.pos == pos]
    all_associations_df = all_associations_df.drop(['pos'], axis=1)
    top_results = all_associations_df.iloc[0:top_number]
    return top_results


def predict_from_word_vectors(tokens, word_list, spacy_vocab, nlp, POS="NOUN", top_number=constants.DEFAULT_TOP_NGRAMS):
    closest_vectors_map = {}
    for token in tokens:
        if token not in spacy.lang.en.STOP_WORDS:
            closest_word_vectors = vu.find_closest_word_vectors(token, word_list, spacy_vocab)
            # Try removing all the one with a similarity of 1, as being either the same word, or erroneously close
            closest_word_vectors = closest_word_vectors[closest_word_vectors[constants.SIMILARITY] != 1]
            #print(closest_word_vectors.head(10))
            closest_vectors_map[token] = closest_word_vectors
    # At this point, we need to do something with all the dfs of word vectors.
    m = closest_vectors_map
    keys = list(m.keys())
    r = m[keys[0]]
    for i in range(1, len(m.keys())):
        r = r.merge(m[keys[i]], on="word")
    a = ['word']
    a.extend(keys)
    r.columns = a
    r['sum'] = 0
    for column in keys:
        r['sum'] += r[column]
    r['product'] = 1
    for column in keys:
        r['product'] *= r[column]
    r['sum_sq'] = 0
    for column in keys:
        r['sum_sq'] += (r[column]) ** 2
    r['prd_sq'] = 1
    for column in keys:
        r['prd_sq'] *= ((r[column]) ** 2)

    top_results = get_top_results(r, nlp, top_number, POS)
    return top_results


def predict_from_word_vectors_matrix(tokens, matrix, nlp, POS="NOUN", top_number=constants.DEFAULT_TOP_ASSOCIATIONS):
    r = collect_word_vector_associations(tokens, matrix)
    top_results = get_top_results(r, nlp, top_number, POS)
    return top_results


#def predict(tokens, list_of_hists, word_list, spacy_vocab, nlp, POS="NOUN"):
def predict(tokens, list_of_hists, matrix_df, nlp, POS="NOUN"):
    ngram_results = collect_n_grams_matches(tokens, list_of_hists)
    # If there is a clear winner in n grams, use that.
    # Defining "clear winner" could be a huge task in itself...
    threshold = 20
    ng_result = None # Find a way to default it to the overall word frequency
    for i in [x for x in range(len(ngram_results))][::-1]:
        if ngram_results[i]['count'].sum() > threshold:
            ng_result = ngram_results[i].head(5)
            break

    # If not, use the word vectors.
    #association_results = predict_from_word_vectors(tokens, word_list, spacy_vocab, nlp, POS)
    association_results = predict_from_word_vectors_matrix(tokens, matrix_df, nlp)
    return "?"


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
        collect_n_grams_matches(tokens, hist_list)


def load_hist(file_name):
    """
    Load a histogram DataFrame
    :param file_name: path and name of the .csv file that pandas can load
    :return: a pandas DataFrame
    """
    df = pd.read_csv(file_name)
    return df


if __name__ == "__main__":
    #hist = load_hist("twitter_sample.txt_4_grams.csv")
    do_predictions()
