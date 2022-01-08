import pandas as pd
import constants
import spacy
# import vector_utils as vu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from functools import partial
import pickle
import basic_operations as bo
import time

import prediction as prd

def test_one_ngram_hist(training_df, ngram_hist, prefix_maps):
    start = time.time()
    #for source, target in training_df.itertuples():
    #    pass
    # When I did a timeit, iterrows was faster than itertuples
    results = []
    source_length = len(ngram_hist.iloc[0].source.split(" "))
    not_found_rank = ngram_hist.shape[0] + 1
    print(f"not found rank = {not_found_rank}")
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
            source_matches = prd.match_n_grams_by_index(indices, ngram_hist)
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
                results.append(not_found_rank)
        else:
            results.append(not_found_rank)
    end = time.time()
    print(f"tested in {(end - start)} seconds")
    return results


def run_one_ngram_test(source, n, training_df=None, ngrams=None, prefix_map=None):
    print(f"run one ngram_test({source}, {n}, {training_df.shape})")

    if ngrams is None or prefix_map is None:
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
    """
    Test accuracy of word vector associations.  Spoiler alert - it was pretty disappointing.
    :param training_df: The training input source and target pairs.
    :param matrix_df: word vectors DataFrame
    :param nlp: the spacy object
    :return:
    """
    start = time.time()
    match_ranking = []
    for i, row in training_df.iterrows():
        tokens = row.source.split(" ")
        try:
            result = prd.predict_from_word_vectors_matrix(tokens, matrix_df, nlp, POS="NOUN",
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


def do_one_prediction_test(source_phrase, target, list_of_hists, prefix_maps, matrix_df, nlp, threshold=1):
    """
    Make one prediction of the final word based off the input phrase.  Track how it did, and whether it came from
    ngrams (usual scenario) or word vectors.
    :param source_phrase: the input phrase
    :param target: the correct target (we are hoping that predict() have us this as one of the top options)
    :param list_of_hists: list of ngram histograms
    :param prefix_maps: list of prefix maps
    :param matrix_df: word vectors DataFrame
    :param nlp: the spacy object
    :param threshold: the minimum threshold for the ngram "backoff" algorithm
    :return: a tuple with prediction rank and prediction source
    """
    lengths = [df.shape[0] for df in list_of_hists]
    not_found_rank = np.max(lengths) + 1
    result = prd.predict(source_phrase, list_of_hists, prefix_maps, matrix_df, nlp, threshold=threshold)

    # Check if there actually are results.  I think there always should be, but don't presume.
    if result is None:
        return not_found_rank, constants.NEITHER
    if result.shape[0] == 0:
        return not_found_rank, constants.NEITHER

    # Expected condition.  Now make sure the results are numbered 0 to whatever so we can see where the match is.
    result.index = range(result.shape[0])

    target_matches = result[result[constants.PREDICTION] == target]
    result_type = result[constants.RESULT_TYPE].iloc[0]
    # Checking only the first row is theoretically a flaw, but all the rows should be the same

    rank = not_found_rank
    # Check if the target word is actually in the results
    if target_matches.shape[0] >= 1:
        rank = target_matches.index[0]
    else:
        # oops, it was actually empty
        result_type = constants.NEITHER

    # print(source_phrase, target, rank, result_type)
    return rank, result_type


def test_many_predictions(sentences_df, list_of_hists, prefix_maps, matrix_df, nlp):
    """
    Run a bunch of predictions and see how well the prediction matches up to the actual value.
    :param sentences_df: The input sentences
    :param list_of_hists: list of ngram histograms
    :param prefix_maps: list of prefix maps
    :param matrix_df: word vector associations DataFrame
    :param nlp: the spacy object
    :return: at the moment, nothing
    """
    ranks = []
    result_types = []
    for index, row in sentences_df.iterrows():
        rank, result_type = do_one_prediction_test(row.source, row.target, list_of_hists, prefix_maps, matrix_df, nlp,
                                                   threshold=1)
        ranks.append(rank)
        result_types.append(result_type)
    sns.displot(ranks)
    plt.show()
    sns.displot(result_types)
    plt.show()


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
    """
    Get the lists of ngram histograms and prefix maps, by loading from the presaved files.
    :param source: the source (twitter, news, or blogs).  These refer to the files from the Data Science Capstone
    project.
    :param n: the "n" from "ngram"
    :return: a tuple containing 1) the list of ngrams histograms and 2) the list of prefix maps.  It will be a very
    large object in terms of memory.
    """
    ngrams = pd.read_csv(f"en_US.{source}.txt_{n}_grams.csv")
    ngrams = add_source_target_columns(ngrams)

    # Of course, this function is very specific to the way files are saved at this particular time, and is not a
    # general algorithm.
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


def do_tests():

    training_source = "news"
    testing_source = "news"

    start_all = time.time()
    print(f"starting tests at time {start_all}")
    testing_sentences_df = pd.read_csv(f"word_stats_pkls/training_df_{testing_source}.csv")
    training_df_loaded = time.time()
    print(f"training sentences loaded after {training_df_loaded - start_all} seconds")
    # '''
    three_grams, three_gram_prefix_map = get_grams_and_prefix_map(training_source, 3)
    four_grams, four_gram_prefix_map = get_grams_and_prefix_map(training_source, 4)
    five_grams, five_gram_prefix_map = get_grams_and_prefix_map(training_source, 5)
    six_grams, six_gram_prefix_map = get_grams_and_prefix_map(training_source, 6)
    ngrams_loaded = time.time()
    print(f"ngram data loaded after {ngrams_loaded - training_df_loaded} seconds")

    run_one_ngram_test(training_source, 3, testing_sentences_df, three_grams, three_gram_prefix_map)
    run_one_ngram_test(training_source, 4, testing_sentences_df, four_grams, four_gram_prefix_map)
    run_one_ngram_test(training_source, 5, testing_sentences_df, five_grams, five_gram_prefix_map)
    run_one_ngram_test(training_source, 6, testing_sentences_df, six_grams, six_gram_prefix_map)
    all_ngrams_done = time.time()
    print(f"ngram tests run in {all_ngrams_done - training_df_loaded} seconds")
    # '''

    # Now the word vector part.
    with open("word_stats_pkls/news_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    matrix_df_loaded = time.time()
    print(f"loaded the word vector matrix in {matrix_df_loaded - all_ngrams_done} seconds")
    nlp = spacy.load("en_core_web_md")
    nlp_loaded = time.time()
    print(f"loaded spacy object in {nlp_loaded - matrix_df_loaded} seconds")
    #test_associations(testing_sentences_df, matrix_df, nlp)
    vector_tests_done = time.time()

    test_many_predictions(testing_sentences_df, [four_grams, five_grams, six_grams],
                          [four_gram_prefix_map, five_gram_prefix_map, six_gram_prefix_map], matrix_df, nlp)

    end_all = time.time()
    print(f"predictions done in {end_all - vector_tests_done} seconds")
    print(f"all finished at time {end_all}, total {end_all - start_all} seconds")


if __name__ == "__main__":
    do_tests()



