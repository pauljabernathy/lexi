import pandas as pd
import constants
import spacy
import vector_utils as vu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import pickle
import basic_operations as bo
import time


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
        if token not in spacy.lang.en.STOP_WORDS:
            closest_word_vectors = vu.find_closest_word_vectors_from_matrix(token, matrix)
            # Try removing all the one with a similarity of 1, as being either the same word, or erroneously close
            closest_word_vectors = closest_word_vectors[closest_word_vectors[constants.SIMILARITY] != 1]
            #print(closest_word_vectors.head(10))
            closest_vectors_map[token] = closest_word_vectors
    #return closest_vectors_map
    m = closest_vectors_map
    keys = list(m.keys())
    non_blank_keys = []
    for key in m.keys():
        if m[key].shape[0] > 0:
            non_blank_keys.append(key)
    #results = m[keys[0]]  # TODO: check for empty data frame here also
    # matrix.mean().mean() = 0.12434165442582384
    if len(non_blank_keys) == 0:
        results = pd.DataFrame(columns=[constants.WORD, constants.POS, constants.SIMILARITY])
        results[constants.POS] = matrix[constants.POS]
        results[constants.SIMILARITY] = 0.12434165442582384
        # The mean value of the matrix df I was looking at, I think news.
        # TODO:  replace with a constant or a calculated value (which is slower).
        results[constants.WORD] = matrix.index
        results.index = range(results.shape[0])
    else:
        results = m[non_blank_keys[0]]
        a = [constants.WORD, constants.POS, non_blank_keys[0]]
        for i in range(1, len(non_blank_keys)):
            if m[non_blank_keys[i]].shape[0] > 0:
                results = results.merge(m[non_blank_keys[i]], on=[constants.WORD, constants.POS])
                # I expect the part of speech column to always be the same; using that as a merge criteria so it will
                # maintain on POS column for the whole thing instead of one for each part joined.
                a.append(non_blank_keys[i])
        #a = ['word']
        #a.extend(keys)
        results.columns = a

    results['sum'] = 0
    for column in non_blank_keys:
        results['sum'] += results[column]
    results['product'] = 1
    for column in non_blank_keys:
        results['product'] *= results[column]
    results['sum_sq'] = 0
    for column in non_blank_keys:
        results['sum_sq'] += (results[column]) ** 2
    results['prd_sq'] = 1
    for column in non_blank_keys:
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
    if constants.POS not in all_associations_df:
        all_associations_df[constants.POS] = all_associations_df[constants.WORD].apply(lambda w: nlp(w)[0].pos_)
    all_associations_df = all_associations_df[all_associations_df[constants.POS] == pos]
    # all_associations_df = all_associations_df.drop(['pos'], axis=1)
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
def predict(tokens, list_of_hists, list_of_prefix_maps, matrix_df, nlp, POS="NOUN"):
    # ngram_results = collect_n_grams_matches_original(tokens, list_of_hists)
    ngram_results = collect_n_grams_matches_indices(tokens, list_of_hists, list_of_prefix_maps)
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


def test_one_ngram_hist(training_df, ngram_hist, prefix_maps):
    start = time.time()
    #for source, target in training_df.itertuples():
    #    pass
    # When I did a timeit, iterrows was faster than itertuples
    results = []
    source_length = 3
    source_length = len(ngram_hist.iloc[0].source.split(" "))
    print(f"testing {source_length + 1} grams; {training_df.shape[0]} sentences; {ngram_hist.shape[0]} ngams")
    # The above makes the following assumptions:
    # 1) ngrams_hist has rows
    # 2) ngrams_hist has a column called "source" (otherwise, we'll have to get it from the "gram" column)
    # 3) all ngrams in ngrams_hist are of the same length
    for i, row in training_df.iterrows():
        tokens = row.source.split(" ")
        source = " ".join(tokens[-source_length:])
        source = " ".join(tokens[-(source_length + 1):-1])
        if source in prefix_maps:
            indices = prefix_maps[source]
            source_matches = match_n_grams_by_index(indices, ngram_hist)
            # So we have places in the ngram histogram where the source was in the histogram.  Do any point to the
            # target?
            # target_matches = source_matches[source_matches.target.str == row.target]
            # results.append(Which place in the source_matches that the target match was.)
            source_matches.index = range(source_matches.shape[0])
            target = tokens[-1]
            target_matches = source_matches[source_matches.target == target]
            if target_matches.shape[0] == 1:
                results.append(target_matches.index[0])
            elif target_matches.shape[0] > 1:
                # would this ever happen?
                results.append(target_matches.index[0])
            else:
                results.append(-1)
        else:
            results.append(-2)
    end = time.time()
    print(f"tested in {(end - start)} seconds")
    return results


def run_one_ngram_test(source, n, training_df=None):
    ngrams, prefix_map = get_grams_and_prefix_map(source, n)
    if training_df is None:
        training_df = pd.read_csv(f"word_stats_pkls/training_df_{source}.csv")
    match_ranking = test_one_ngram_hist(training_df, ngrams, prefix_map)
    match_ranking_hist = pd.Series(match_ranking).value_counts()
    hist2 = match_ranking_hist.sort_index()
    cumulative_match_rankings = hist2.cumsum() / hist2.sum()
    print(cumulative_match_rankings.head(25))
    sns.lineplot(cumulative_match_rankings.index, cumulative_match_rankings.values)
    plt.show()
    return match_ranking, match_ranking_hist, cumulative_match_rankings


def test_associations(training_df, matrix_df, nlp):
    start = time.time()
    match_ranking = []
    for i, row in training_df.iterrows():
        tokens = row.source.split(" ")
        try:
            result = predict_from_word_vectors_matrix(tokens, matrix_df, nlp, POS="NOUN",
                                             top_number=matrix_df.shape[0])
            result.index = range(result.shape[0])
            target_matches = result[result[constants.WORD] == row.target]
            if target_matches.shape[0] == 1:
                match_ranking.append(target_matches.index[0])
            elif target_matches.shape[0] > 1:
                # would this ever happen?
                match_ranking.append(target_matches.index[0])
            else:
                match_ranking.append(-1)
        except Exception as e:
            print("uh oh, had a bit of a mishad trying to predict something")
            print(e.args)
            print(row.source)
            print(row.target)
    match_ranking_hist = pd.Series(match_ranking).value_counts()
    hist2 = match_ranking_hist.sort_index()
    cumulative_match_rankings = hist2.cumsum() / hist2.sum()
    print(cumulative_match_rankings.head(25))
    sns.lineplot(cumulative_match_rankings.index, cumulative_match_rankings.values)
    plt.show()
    end = time.time()
    print(f"tested in {(end - start)} seconds")
    return match_ranking, match_ranking_hist, cumulative_match_rankings


def do_one_prediction_test(source, target, list_of_hists, prefix_maps, matrix_df):
    sentence = source + " " + target
    ngram_results = collect_n_grams_matches_indices(sentence, list_of_hists, prefix_maps)


def test_many_predictions(sentences_df, lists_of_hists, prefix_maps, matrix_df):
    for source, target in sentences_df.itertuples():
        do_one_prediction_test()


'''
things to test
word vectors / associations
ngrams
    each one (four, five, six, etc.)
    all total
ngrams and associations combined


For each source/target and each method
    find the top matches
        one way is to see if the target is in the top X (3, 5, 10) things returned
        another way is to track where it is in the rankings (third, 8th, 27th, first, etc.) and show the distribution
        I think I'd rather start with the second
'''

# TODO:  might should move this to basic_operations.
def add_source_target_columns(ngrams_hist):
    """
    Add the source/prefix and target columns to make it a little easier to work with.  They might not be saved with
    the serialized version to save disk space.  The source column is all words except the last word.  The target
    column is the last word.
    :param ngrams_hist: a DataFrame representing the histogram of ngrams
    :return: the same DataFrame, but with the source and target columns.
    """
    if "source" not in ngrams_hist.columns:
        ngrams_hist['source'] = ngrams_hist.gram.apply(bo.split_prefix)

    if "target" not in ngrams_hist.columns:
        ngrams_hist['target'] = ngrams_hist.gram.apply(lambda text: text.split(" ")[-1])
    return ngrams_hist


def get_grams_and_prefix_map(source, n):
    ngrams = pd.read_csv(f"en_US.{source}.txt_{n}_grams.csv")
    ngrams = add_source_target_columns(ngrams)

    if n == 3:
        english_n = "three"
    if n == 4:
        english_n = "four"
    if n == 5:
        english_n = "five"
    if n == 6:
        english_n = "six"
    with open(f"word_stats_pkls/{english_n}_grams_prefix_map_news.pkl", 'rb') as f:
        prefix_map = pickle.load(f)

    return ngrams, prefix_map


if __name__ == "__main__":
    # do_predictions()

    source = "news"

    start_all = time.time()
    training_df = pd.read_csv(f"word_stats_pkls/training_df_{source}.csv")
    training_df_loaded = time.time()
    print(f"training data loaded after {training_df_loaded - start_all} seconds")
    run_one_ngram_test("news", 3, training_df)
    run_one_ngram_test("news", 4, training_df)
    run_one_ngram_test("news", 5, training_df)
    run_one_ngram_test("news", 6, training_df)

    # Now the word vector part.
    with open("word_stats_pkls/news_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    nlp = spacy.load("en_core_web_md")
    test_associations(training_df, matrix_df, nlp)

    end_all = time.time()
    print(f"all finished after {end_all - start_all} seconds")


