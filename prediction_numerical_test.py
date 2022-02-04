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


def run_one_ngram_test(source, n, training_df=None, ngrams=None, prefix_map=None, directory="data_with_apostrophes/"):
    print(f"run one ngram_test({source}, {n}, {training_df.shape})")

    if ngrams is None or prefix_map is None:
        ngrams, prefix_map = get_grams_and_prefix_map(source, n)
    if training_df is None:
        training_df = pd.read_csv(f"{directory}training_df_{source}.csv")
    match_ranking = test_one_ngram_hist(training_df, ngrams, prefix_map)
    match_ranking_hist = pd.Series(match_ranking).value_counts()
    hist2 = match_ranking_hist.sort_index()
    cumulative_match_rankings = hist2.cumsum() / hist2.sum()
    print(cumulative_match_rankings.head(25))
    sns.lineplot(cumulative_match_rankings.index, cumulative_match_rankings.values)
    plt.show()
    return match_ranking, match_ranking_hist, cumulative_match_rankings


def test_associations(training_df, matrix_df, nlp, pos_pos_map=None):
    """
    Test accuracy of word vector associations.  Spoiler alert - it was pretty disappointing.
    :param training_df: The training input source and target pairs.
    :param matrix_df: word vectors DataFrame
    :param nlp: the spacy object
    :return:
    """
    start = time.time()
    match_ranking = []
    not_found_rank = matrix_df.shape[0] + 1
    for i, row in training_df.iterrows():
        tokens = row.source.split(" ")
        try:
            if pos_pos_map is None:
                # print("use filter")
                result = prd.predict_from_word_vectors_matrix(tokens, matrix_df, nlp, pos="NOUN",
                                                              top_number=matrix_df.shape[0])
            else:
                # print("use pos map")
                result = prd.predict_from_word_vectors_matrix(tokens, matrix_df, nlp, None,
                                                              top_number=matrix_df.shape[0], pos_pos_map=pos_pos_map)
            result.index = range(result.shape[0])
            target_matches = result[result[constants.WORD] == row.target]
            if target_matches.shape[0] >= 1:
                match_ranking.append(target_matches.index[0])
            else:
                match_ranking.append(not_found_rank)
        except Exception as e:
            print("uh oh, had a bit of a mishad trying to predict something")
            print(e.args)
            print(row.source)
            print(row.target)
    match_ranking_hist = pd.Series(match_ranking).value_counts()
    hist2 = match_ranking_hist.sort_index()
    cumulative_match_rankings = hist2.cumsum() / hist2.sum()
    # print(cumulative_match_rankings.head(25))
    both_hists = pd.DataFrame({"raw_ratio": hist2 / hist2.sum(), "cumulative": cumulative_match_rankings})
    print(both_hists.head(50))
    sns.lineplot(cumulative_match_rankings.index, cumulative_match_rankings.values)
    plt.show()
    sns.displot(match_ranking_hist)
    plt.show()
    end = time.time()
    print(f"tested in {(end - start)} seconds")
    return match_ranking, match_ranking_hist, cumulative_match_rankings

# TODO:  ability to pass in which prediction function you want to use; a paramaeter dict might be necessary.
def do_one_prediction_test(source_phrase, target, list_of_hists, prefix_maps, matrix_df, nlp, threshold=1,
                           predict_function=prd.predict, kwargs={}):
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
    #print(f"{source_phrase} -> {target}")
    lengths = [df.shape[0] for df in list_of_hists]
    not_found_rank = np.max(lengths) + 1
    #result = prd.predict(source_phrase, list_of_hists, prefix_maps, matrix_df, nlp, threshold=threshold)
    result = predict_function(source_phrase, list_of_hists, prefix_maps, matrix_df, nlp, threshold=threshold, **kwargs)
    #print(result)

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
    #print(f"{rank}, {result_type}")
    return rank, result_type


def test_many_predictions(sentences_df, list_of_hists, prefix_maps, matrix_df, nlp, predict_function=prd.predict,
                          kwargs={}):
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
                                                   threshold=1, predict_function=predict_function, kwargs=kwargs)
        ranks.append(rank)
        result_types.append(result_type)
    #print(pd.Series(ranks).value_counts().sort_index())
    #sns.displot(ranks)
    #plt.show()
    label = f'weight = {kwargs["ngram_weight"]}' if "ngram_weight" in kwargs else ""
    sns.displot(result_types, label=label)
    plt.show()
    match_ranking_hist = pd.Series(ranks).value_counts()
    hist2 = match_ranking_hist.sort_index()
    print(hist2)
    cumulative_match_rankings = hist2.cumsum() / hist2.sum()
    print(cumulative_match_rankings.head(25))
    rankings_to_graph = cumulative_match_rankings[cumulative_match_rankings.index < 1000]
    sns.lineplot(rankings_to_graph.index, rankings_to_graph.values, label=label)
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


def get_grams_and_prefix_map(source, n, directory="data_with_apostrophes/"):
    """
    TODO:  Move this somewhere else.  Probably a utilities module.
    Get the lists of ngram histograms and prefix maps, by loading from the presaved files.
    :param source: the source (twitter, news, or blogs).  These refer to the files from the Data Science Capstone
    project.
    :param n: the "n" from "ngram"
    :return: a tuple containing 1) the list of ngrams histograms and 2) the list of prefix maps.  It will be a very
    large object in terms of memory.
    """
    if not directory.endswith("/"):
        directory += '/'
    ngrams = pd.read_csv(f"{directory}en_US.{source}.txt_{n}_grams.csv")
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
    with open(f"{directory}{n}_grams_prefix_map_news.pkl", 'rb') as f:
        prefix_map = pickle.load(f)

    return ngrams, prefix_map


def do_tests():

    print("\n\n\n\n----\ntesting predict")

    training_source = "news"
    testing_source = "twitter"

    start_all = time.time()
    print(f"starting tests at time {start_all}")
    directory = "word_stats_pkls/"
    directory = "data_with_apostrophes/"

    # testing_sentences_df = pd.read_csv(f"{directory}training_df_{testing_source}.csv")
    testing_sentences_df = pd.read_csv(f"{directory}test_sentences_df_{testing_source}.csv")
    testing_df_loaded = time.time()
    print(f"training sentences loaded after {testing_df_loaded - start_all} seconds")
    # '''
    two_grams, two_gram_prefix_map = get_grams_and_prefix_map(training_source, 2)
    three_grams, three_gram_prefix_map = get_grams_and_prefix_map(training_source, 3)
    four_grams, four_gram_prefix_map = get_grams_and_prefix_map(training_source, 4)
    #five_grams, five_gram_prefix_map = get_grams_and_prefix_map(training_source, 5)
    #six_grams, six_gram_prefix_map = get_grams_and_prefix_map(training_source, 6)
    ngrams_loaded = time.time()
    print(f"ngram data loaded after {ngrams_loaded - testing_df_loaded} seconds")
    histograms = [two_grams, three_grams, four_grams]
    # histograms = [three_grams]
    prefix_maps = [two_gram_prefix_map, three_gram_prefix_map, four_gram_prefix_map]

    '''run_one_ngram_test(training_source, 3, testing_sentences_df, three_grams, three_gram_prefix_map)
    run_one_ngram_test(training_source, 4, testing_sentences_df, four_grams, four_gram_prefix_map)
    run_one_ngram_test(training_source, 5, testing_sentences_df, five_grams, five_gram_prefix_map)
    run_one_ngram_test(training_source, 6, testing_sentences_df, six_grams, six_gram_prefix_map)
    '''
    all_ngrams_done = time.time()
    print(f"ngram tests run in {all_ngrams_done - testing_df_loaded} seconds")
    # '''

    # Now the word vector part.
    with open(f"{directory}{training_source}_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    matrix_df_loaded = time.time()
    print(f"loaded the word vector matrix in {matrix_df_loaded - all_ngrams_done} seconds")
    nlp = spacy.load("en_core_web_md")
    nlp_loaded = time.time()
    print(f"loaded spacy object in {nlp_loaded - matrix_df_loaded} seconds")
    #test_associations(testing_sentences_df, matrix_df, nlp)
    vector_tests_done = time.time()

    test_many_predictions(testing_sentences_df, histograms, prefix_maps, matrix_df, nlp)

    end_all = time.time()
    print(f"predictions done in {end_all - vector_tests_done} seconds")
    print(f"all finished at time {end_all}, total {end_all - start_all} seconds")


def test_exp_predict_1():
    training_source = "news"
    testing_source = "news"

    start_all = time.time()
    print(f"starting tests at time {start_all}")
    testing_sentences_df = pd.read_csv(f"word_stats_pkls/training_df_{testing_source}.csv")
    testing_sentences_df = testing_sentences_df.iloc[:3]
    training_df_loaded = time.time()
    print(f"training sentences loaded after {training_df_loaded - start_all} seconds")
    # '''
    # three_grams, three_gram_prefix_map = get_grams_and_prefix_map(training_source, 3)
    four_grams, four_gram_prefix_map = get_grams_and_prefix_map(training_source, 4)
    # five_grams, five_gram_prefix_map = get_grams_and_prefix_map(training_source, 5)
    # six_grams, six_gram_prefix_map = get_grams_and_prefix_map(training_source, 6)
    ngrams_loaded = time.time()
    print(f"ngram data loaded after {ngrams_loaded - training_df_loaded} seconds")

    phrases = [
        "The guy in front of me just bought a pound of bacon, a bouquet, and a case of",
        "You're the reason why I smile everyday. Can you follow me please? It would mean the",
        "Hey sunshine, can you follow me and make me the",
        "Very early observations on the Bills game: Offense still struggling but the",
        "Go on a romantic date at the",
        "Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
        "Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",
        "After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
        "Be grateful for the good times and keep the faith during the",
        "If this isn't the cutest thing you've ever seen, then you must be",
    ]

    list_of_hists = [four_grams]
    prefix_maps = [four_gram_prefix_map]
    with open("word_stats_pkls/news_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    nlp = spacy.load("en_core_web_md")

    results = []
    for index, row in testing_sentences_df.iterrows():
        result = prd.exp_predict_1(row.source, list_of_hists, prefix_maps, matrix_df, nlp, threshold=5)
        results.append(result)
    comparison = pd.DataFrame({"actual": testing_sentences_df['target'], "predicted": results})
    print(comparison)
    print((comparison.actual == comparison.predicted).sum() / comparison.shape[0])


def compare_word_vectors_algorithms():
    training_source = "news"
    testing_source = "news"

    start_all = time.time()
    print(f"starting tests at time {start_all}")
    testing_sentences_df = pd.read_csv(f"word_stats_pkls/testing_df_{testing_source}.csv")
    testing_sentences_df = pd.read_csv(f"word_stats_pkls/testing_df_blogs_3.csv")
    # testing_sentences_df = testing_sentences_df.iloc[:100]
    training_df_loaded = time.time()
    print(f"training sentences loaded after {training_df_loaded - start_all} seconds")

    with open("word_stats_pkls/news_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    matrix_df_loaded = time.time()
    # print(f"loaded the word vector matrix in {matrix_df_loaded - all_ngrams_done} seconds")
    nlp = spacy.load("en_core_web_md")
    nlp_loaded = time.time()
    print(f"loaded spacy object in {nlp_loaded - matrix_df_loaded} seconds")
    test_associations(testing_sentences_df, matrix_df, nlp)
    vector_tests_done = time.time()
    print(f'finished filter vector tests in {vector_tests_done - nlp_loaded}')
    pos_pos_map = pd.read_csv("moby_pos_pos.csv", index_col=0)
    test_associations(testing_sentences_df, matrix_df, nlp, pos_pos_map=pos_pos_map)
    scale_vector_tests_done = time.time()
    print(f'finished scale vector tests in {scale_vector_tests_done - vector_tests_done}')
    end_all = time.time()
    print(f"all finished at time {end_all}, total {end_all - start_all} seconds")


def test_exp_predict_3():
    print("\n\n\n\n----\ntesting exp_predict_3")

    pd.options.mode.chained_assignment = None
    training_source = "news"
    testing_source = "twitter"

    start_all = time.time()
    print(f"starting tests at time {start_all}")
    directory = "data_with_apostrophes/"
    testing_sentences_df = pd.read_csv(f"{directory}test_sentences_df_{testing_source}.csv")
    testing_df_loaded = time.time()
    print(f"training sentences loaded after {testing_df_loaded - start_all} seconds")
    # '''
    two_grams, two_gram_prefix_map = get_grams_and_prefix_map(training_source, 2)
    three_grams, three_gram_prefix_map = get_grams_and_prefix_map(training_source, 3)
    four_grams, four_gram_prefix_map = get_grams_and_prefix_map(training_source, 4)
    #five_grams, five_gram_prefix_map = get_grams_and_prefix_map(training_source, 5)
    #six_grams, six_gram_prefix_map = get_grams_and_prefix_map(training_source, 6)
    ngrams_loaded = time.time()
    print(f"ngram data loaded after {ngrams_loaded - testing_df_loaded} seconds")
    histograms = [two_grams, three_grams, four_grams]
    # histograms = [three_grams]
    prefix_maps = [two_gram_prefix_map, three_gram_prefix_map, four_gram_prefix_map]
    # prefix_maps = [three_gram_prefix_map]

    '''run_one_ngram_test(training_source, 3, testing_sentences_df, three_grams, three_gram_prefix_map)
    run_one_ngram_test(training_source, 4, testing_sentences_df, four_grams, four_gram_prefix_map)
    run_one_ngram_test(training_source, 5, testing_sentences_df, five_grams, five_gram_prefix_map)
    run_one_ngram_test(training_source, 6, testing_sentences_df, six_grams, six_gram_prefix_map)'''
    all_ngrams_done = time.time()
    print(f"ngram tests run in {all_ngrams_done - testing_df_loaded} seconds")
    # '''

    # Now the word vector part.
    with open(f"{directory}{training_source}_matrix_df.pkl", 'rb') as f:
        matrix_df = pickle.load(f)
    matrix_df_loaded = time.time()
    print(f"loaded the word vector matrix in {matrix_df_loaded - all_ngrams_done} seconds")
    nlp = spacy.load("en_core_web_md")
    nlp_loaded = time.time()
    print(f"loaded spacy object in {nlp_loaded - matrix_df_loaded} seconds")
    #test_associations(testing_sentences_df, matrix_df, nlp)
    vector_tests_done = time.time()

    ratios = np.arange(0.6, 1.01, 0.05)
    ratios = np.arange(0.6, 1.01, .1)
    ratios = [.7, .99]
    testing_sentences_df = testing_sentences_df.iloc[:]
    for ratio in ratios:
        print(f"\n\n\n\nrunning many predictions with ngram weight of {ratio}")
        test_many_predictions(testing_sentences_df, histograms,
                              prefix_maps, matrix_df, nlp,
                              predict_function=prd.exp_predict_3, kwargs={"ngram_weight": ratio})

    end_all = time.time()
    print(f"predictions done in {end_all - vector_tests_done} seconds")
    print(f"all finished at time {end_all}, total {end_all - start_all} seconds")


if __name__ == "__main__":
    do_tests()
    # test_exp_predict_1()
    # compare_word_vectors_algorithms()
    test_exp_predict_3()



