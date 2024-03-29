import re
import pandas as pd
import numpy as np
import time
import constants
import pickle


# Cleansing

def cleanse(text):
    """
    Clean up the text a little by removing punctuation, extra spaces, new lines, etc.
    This should be run after split_to_sentences(), tokenize_by_sentence() because I think it removes all the
    punctuation you will need to split the text into sentences.
    :param text:
    :return:
    """
    text = text.lower()
    # text = text.replace("'", '')    # Remove apostrophes
    # text = re.sub('[^\w\s]', ' ', text)
    # Replace punct with a space so that when someone does something like <word comma word> you don't accidentally
    # transform it into one word.  We remove extra spaces in the next line.
    text = re.sub('[^\w\s\']', ' ', text)
    text = re.sub('\\n', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()     # Small thing but remove a trailing space.
    return text


# Tokenization

def tokenize_string(sentence):
    """
    cleanse the string and tokenize to individual words
    :param sentence: a string of text (in the context of this application, most likely an individual sentence)
    :return: a list of strings
    """
    sentence = cleanse(sentence)
    return sentence.split(' ')
# TODO: Convert contractions to the "uncontracted" two words.  Ex "you'll" -> "you are".
# Would need some list of common contractions.  Of course, this is language dependent.


def split_to_sentences(text):
    """
    Gets a bunch of text and returns the sentences as a list.  It attempts to split the text up into its component
    sentences, using punctuation that typically ends a sentence (see constants.PUNCTUATION_REGEX, which at the moment is
    '[.?!]').  Text that does not behave this way, for example when each line is intended to be independent,
    will likely give an unexpected result.
    :param text: regular text; for example the contents of a file of text
    :return: a list, where each element in the list is a sentence
    """
    # TODO: A way to handle text that is broken up by lines (for example, poetry); maybe allow the call to specify
    #  the regex.
    p = re.compile(constants.SENTENCE_END_PUNCT_REGEX)
    sentences = p.split(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
    if sentences[-1] == '':
        sentences = sentences[:-1]
    return sentences


def tokenize_by_sentence(text):
    """
    Tokenize the text, but group words in sentences together.  The input, and constraints on tokenization,
    are the same as for split_to_sentences().
    :param text: regular text; for example the contents of a file of text
    :return: A list of lists.  Each list inside the overall list is the words in a given sentence.
    """
    sentences = split_to_sentences(text)
    result = []
    for sentence in sentences:
        current_result = tokenize_string(sentence)
        if current_result is not None and current_result != ['']:
            result.append(current_result)
    return result


# Statistics

def find_word_stats(text):
    """
    Get statistics on word frequencies.  This tells you the word and how many times it occurred in the text.  There
    are also columns for the fraction of total words that it represents, cumulative count and cumulative ratio of all
    words, "cumulative" being if you count that word and all other words above it.
    :param text:
    :return: a DataFrame, sorted by most common words first
    """
    tokens = tokenize_string(text)
    tokens_pd = pd.Series(tokens)
    token_hist = tokens_pd.value_counts()
    stats = pd.DataFrame({constants.COUNT_COLUMN_NAME: token_hist, constants.FRACTION_COLUMN_NAME: token_hist / len(tokens)})
    stats[constants.FRACTION_COLUMN_NAME] = stats[constants.COUNT_COLUMN_NAME] / len(tokens)
    stats[constants.CUM_SUM_COLUMN_NAME] = stats[constants.COUNT_COLUMN_NAME].cumsum()
    stats[constants.CUM_FRAC_COLUMN_NAME] = stats[constants.CUM_SUM_COLUMN_NAME] / len(tokens)
    return stats


def find_sentence_lengths_hist(list_of_sentences):
    """
    Find a histogram of all sentences lengths.  Could be useful in looking at the writing style of an author,
    for example.
    :param list_of_sentences:
    :return: a pandas Series
    """
    lengths = []
    for i in range(len(list_of_sentences)):
        lengths.append(len(list_of_sentences[i]))
    hist = pd.Series(lengths).value_counts().sort_index()
    return hist


# n grams

def find_n_grams_1_d_list(input: list, n: int):
    ngrams = []
    if n <= 0:
        return ngrams
    for i in range(len(input) - n + 1):
        ngrams.append(input[i:(i+n)])
    return ngrams


def find_n_grams_list_of_lists(list_of_sentences: list, n: int):
    """
    Make ngrams from the input.  The input is structured so that each sentence is separate, and ngrams do not cross
    sentences.  For example, in the text "One sentence.  Two sentences.", 2 grams would be "one sentence" and "two
    sentences", but not "sentence two".
    :param list_of_sentences: Must be a list where each item is itself a list.  For example, the following text
    "One sentence.  Two sentences.  To be or not to be.  Whatever.  The problem is that I don't even know " \
               "what a sentence is." would need to be converted to the following format to be used in this function:
    [['one', 'sentence'], ['two', 'sentences'], ['to', 'be', 'or', 'not', 'to', 'be'],
    ['whatever'], ['the', 'problem', 'is', 'that', 'i', 'dont', 'even', 'know', 'what', 'a', 'sentence', 'is']
    :param n: the "gram length"; for example 2 => bi grams ("to be", "be or", "or not", etc.)
    3 => tri grams ("to be or", "be or not" etc.)
    :return: a "flat" (one dimensional) list with all the n grams
    """
    ngrams = []
    if list_of_sentences is None:
        return ngrams
    for item in list_of_sentences:
        current_ngrams = find_n_grams_1_d_list(item, n)
        ngrams.extend(current_ngrams)
    return ngrams


def find_n_grams_from_text(text, n):
    lists_of_words = tokenize_by_sentence(text)
    ngrams = find_n_grams_list_of_lists(lists_of_words, n)
    return ngrams


def convert_n_grams_to_hist_df(ngrams_list):
    """
    Take the raw list of n grams and convert to a DataFrame that maps the n gram to the count
    :param ngrams_list: a list of all ngrams
    :return: a DataFrame
    """
    # Use Series.value_counts() to get the counts.
    n_grams_series = pd.Series(ngrams_list)

    # However, using value_counts() on a list is extremely slow, so convert it to a string.
    # example ['in', 'the', 'fridge'] -> "in the fridge"
    # I know, maybe find_n_grams_from_text should return the results as a string so we aren't making it a list and
    # then converting back.  TODO:  Determine at some point if we should generate n grams as a string not a list.
    n_grams_series = n_grams_series.apply(constants.N_GRAM_SEPARATOR.join)
    n_grams_hist = n_grams_series.value_counts()
    grams_hist_df = pd.DataFrame({constants.GRAM_COLUMN_NAME: n_grams_hist.index, constants.COUNT_COLUMN_NAME: n_grams_hist.values})
    return grams_hist_df


def process_one_file(input_file_name, directory=None):
    start = time.time()
    print(start)
    with open(input_file_name, 'r', encoding='UTF-8') as f:
        file_text = f.read()
    # word_stats_df = find_word_stats(file_text)
    sentences = tokenize_by_sentence(file_text)
    # sentence_lengths = find_sentence_lengths_hist(sentences)

    output_file_stem = input_file_name.split("/")[-1]
    if directory is not None:
        if not directory.endswith("/"):
            directory = directory + "/"
        output_file_stem = directory + output_file_stem

    two_grams = find_n_grams_list_of_lists(sentences, 2)
    two_grams_hist_df = convert_n_grams_to_hist_df(two_grams)
    print('we have the two grams hist now', time.time())
    two_grams_hist_df.to_csv(output_file_stem + "_2_grams.csv", index=False)

    three_grams = find_n_grams_list_of_lists(sentences, 3)
    three_grams_hist_df = convert_n_grams_to_hist_df(three_grams)
    print('we have the three grams hist now', time.time())
    three_grams_hist_df.to_csv(output_file_stem + "_3_grams.csv", index=False)

    four_grams = find_n_grams_list_of_lists(sentences, 4)
    four_grams_hist_df = convert_n_grams_to_hist_df(four_grams)
    four_grams_hist_df.to_csv(output_file_stem + "_4_grams.csv", index=False)

    five_grams = find_n_grams_list_of_lists(sentences, 5)
    five_grams_hist_df = convert_n_grams_to_hist_df(five_grams)
    five_grams_hist_df.to_csv(output_file_stem + "_5_grams.csv", index=False)

    six_grams = find_n_grams_list_of_lists(sentences, 6)
    six_grams_hist_df = convert_n_grams_to_hist_df(six_grams)
    six_grams_hist_df.to_csv(output_file_stem + "_6_grams.csv", index=False)

    print('we have the n grams now', time.time())

    end = time.time()
    print("processing completed in", (end - start), "seconds")

    '''pd.DataFrame(two_grams_hist_df).to_csv(file_name.split("/")[-1] + "_2_grams.csv", index=True)
    pd.DataFrame(three_grams_hist_df).to_csv(file_name.split("/")[-1] + "_3_grams.csv", index=True)
    pd.DataFrame(four_grams_hist_df).to_csv(file_name.split("/")[-1] + "_4_grams.csv", index=True)
    pd.DataFrame(five_grams_hist_df).to_csv(file_name.split("/")[-1] + "_5_grams.csv", index=True)
    pd.DataFrame(six_grams_hist_df).to_csv(file_name.split("/")[-1] + "_6_grams.csv", index=True)'''

    print("files saved")


def split_prefix(text):
    """
    get all the words in the sentence except the last
    :param text:
    :return:
    """
    tokens = text.split(" ")
    return " ".join(tokens[:(len(tokens) - 1)])


def create_prefix_map(ngrams_hist):
    """
    Create a Dict object that maps the prefix to all indices in the input histogram where that prefix occurs
    :param ngrams_hist: a DataFrame
    :return: a dictionary
    """
    if constants.SOURCE not in ngrams_hist.columns:
        # ngrams_hist['prefix'] = ngrams_hist.gram.apply(split_prefix)
        ngrams_hist[constants.SOURCE] = ngrams_hist.gram.apply(split_prefix)
    if constants.TARGET not in ngrams_hist.columns:
        ngrams_hist[constants.TARGET] = ngrams_hist.gram.apply(lambda text: text.split(" ")[-1])
    prefix_map = {}
    for (index, gram, count, prefix, target) in ngrams_hist.itertuples():
        # print(gram, count, prefix, target)
        if prefix not in prefix_map:
            prefix_map[prefix] = [index]
            #prefix_map[prefix] = np.array([index])
        else:
            prefix_map[prefix].append(index)
            #prefix_map[prefix] = np.append(prefix_map[prefix], index)
    return prefix_map


def make_one_prefix_map(source, number, directory="../"):
    # ngram_hist = pd.read_csv(f'../en_US.{source}.txt_{number}_grams.csv')
    ngram_hist = pd.read_csv(f'{directory}en_US.{source}.txt_{number}_grams.csv')
    prefix_map = create_prefix_map(ngram_hist)
    # directory = "../word_stats_pkls/"
    # directory = "../data_with_apostrophes/"
    with open(f"{directory}{number}_grams_prefix_map_{source}.pkl", 'wb') as f:
        pickle.dump(prefix_map, f)


def make_prefix_maps(sources, numbers, directory="./"):
    # make_one_prefix_map(source, 2, directory)
    '''make_one_prefix_map(source, 3, directory)
    make_one_prefix_map(source, 4, directory)
    make_one_prefix_map(source, 5, directory)
    make_one_prefix_map(source, 6, directory)'''
    for source in sources:
        for n in numbers:
            make_one_prefix_map(source, n, directory)


# Creating training data

def get_random_sentences(sentences_as_list, how_many, min_num_words=None):
    """
    Get random sentences from a list of sentences.  Technically, all this function does, at least right now,
    is one line, return np.random.choice(sentences_as_list, how_many, replace=False).  But that could change.
    :param sentences_as_list:
    :param how_many:
    :return:
    """
    if how_many > len(sentences_as_list):
        how_many = len(sentences_as_list)
    # TODO: The line below returns an np array.  Should it return a regular list?
    # return np.random.choice(sentences_as_list, how_many, replace=False)
    # To test for minimum number of words, let's shuffle the indices first, then, if necessary, test each sentence at
    # the random index to see if it is long enough.
    all_indices = range(len(sentences_as_list))
    shuffled_indices = np.random.choice(all_indices, size=len(sentences_as_list), replace=False)
    if min_num_words is None:
        indices_to_use = shuffled_indices[:how_many]
    else:
        indices_to_use = []
        for i in shuffled_indices:
            if len(sentences_as_list[i].split(" ")) >= min_num_words:
                indices_to_use.append(i)
            if len(indices_to_use) >= how_many:
                break
    return np.array(sentences_as_list)[indices_to_use]


def get_training_sentences(text, how_many, min_num_words=None):
    """
    Get a random subset of sentences from the text for training.  Although technically it would work for some purpose other than training.
    :param text: The text (such as the contents of a text file) to get sentences from.
    :param how_many: number of sentences to get
    :return: a list of sentences
    """
    #text = cleanse(text)
    sentences = split_to_sentences(text)
    training_sentences = get_random_sentences(sentences, how_many, min_num_words)
    for i in range(len(training_sentences)):
        training_sentences[i] = cleanse(training_sentences[i])
    return training_sentences


def get_training_sentences_from_file(full_path, how_many, random_seed=37, min_num_words=None):
    """
    Get a random subset of sentences form the specifies file.
    TODO:  Rename to get_sample_sentences() on a future commit.
    :param full_path: the directory and name of the file
    :param how_many: the number of sentences to choose
    :param random_seed: optional random seed for consistent results
    :param min_num_words: the minimum number of words a sentence must be to be included; if not specified, sentences of
    any length are allowed
    :return: a list of sentences taken from the file
    """

    # might need to detect encoding rather than just assume utf-8
    with open(full_path, 'r', encoding='UTF-8') as f:
        text = f.read()

    np.random.seed(random_seed)
    training_sentences = get_training_sentences(text, how_many, min_num_words=min_num_words)
    return training_sentences


def get_training_df(sentences_list):
    """
    converts the list of sentences into a DataFrame
    :param sentences_list: a liste of sentences
    :return: a DataFrame; the columns are "source" containing all but the last word of the sentence, and "target"
    containing the last word
    """
    # TODO:  I notice that uses the term "source" where earlier it uses "prefix" as in get_prefix_map().
    sources = []
    targets = []
    for sentence in sentences_list:
        words = sentence.split(" ")
        target = words[-1]
        # source = " ".join(words[0:(len(words) - 1)])
        source = split_prefix(sentence)
        sources.append(source)
        targets.append(target)
    training_df = pd.DataFrame({'source': sources, 'target': targets})
    return training_df


def create_and_save_training_df(sources):
    for source in sources:
        input_file_name = f"../../courses/data_science_capstone/en_US/en_US.{source}.txt"
        sentences = get_training_sentences_from_file(input_file_name, 1000, min_num_words=6)

        # with open("training_sentences.pkl", 'rb') as f:
        #    sentences = pickle.load(f)
        # '''
        training_df = get_training_df(sentences)
        print(training_df.head())

        # Save the training_df.  Both ways, and see which file is bigger.
        # training_df_file_stem = f"{directory}test_sentences_df_{source}"
        output_file = f"{directory}test_sentences_df_{source}.pkl"
        output_file_csv = f"{directory}test_sentences_df_{source}.csv"
        with open(output_file, 'wb') as f:
            pickle.dump(training_df, f)
        training_df.to_csv(output_file_csv, index=False)
        # '''


if __name__ == "__main__":
    source = "news"
    source = "blogs"
    source = "twitter"
    input_file_name = f"../../courses/data_science_capstone/en_US/en_US.{source}.txt"

    should_find_word_stats = False
    should_find_ngrams = False
    should_make_prefix_maps = False
    should_make_prefix_maps = True
    should_make_test_sentence_df = False
    # should_make_test_sentence_df = True

    directory = "./data_with_apostrophes/"

    if should_find_word_stats:
        with open(input_file_name, 'r', encoding='UTF-8') as f:
            text = f.read()
            stats = find_word_stats(text)
            output_file_name = f"{directory}{source}_stats.csv"
            stats.to_csv(output_file_name)

    if should_find_ngrams:
        process_one_file(input_file_name, directory)

    if should_make_prefix_maps:
        make_prefix_maps(["blogs", "twitter"], [2, 3, 4], directory)

    # TODO: Distinguish between a sample for training and for testing.  Right now they are only for testing.  But it
    #  would be nice to have an algorithm that uses a bunch for training in some way then tests with a different set.
    if should_make_test_sentence_df:
        create_and_save_training_df(['twitter', 'news', 'blogs'])




