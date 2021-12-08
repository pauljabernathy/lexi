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
    text = text.replace("'", '')    # Remove apostrophes
    # text = re.sub('[^\w\s]', ' ', text)
    # Replace punct with a space so that when someone does something like <word comma word> you don't accidentally
    # transform it into one word.  We remove extra spaces in the next line.
    text = re.sub('[^\w\s]', ' ', text)
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
    p = re.compile(constants.PUNCTUATION_REGEX)
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


def process_one_file(file_name):
    start = time.time()
    print(start)
    with open(file_name, 'r', encoding='UTF-8') as f:
        file_text = f.read()
    # word_stats_df = find_word_stats(file_text)
    sentences = tokenize_by_sentence(file_text)
    # sentence_lengths = find_sentence_lengths_hist(sentences)
    two_grams = find_n_grams_list_of_lists(sentences, 2)
    three_grams = find_n_grams_list_of_lists(sentences, 3)
    four_grams = find_n_grams_list_of_lists(sentences, 4)
    five_grams = find_n_grams_list_of_lists(sentences, 5)
    six_grams = find_n_grams_list_of_lists(sentences, 6)
    print('we have the n grams now', time.time())
    #two_grams_series = pd.Series(two_grams)
    # two_grams_series = two_grams_series.apply(",".join)
    #two_grams_series = two_grams_series.apply(constants.N_GRAM_SEPARATOR.join)
    #two_grams_hist = two_grams_series.value_counts()
    two_grams_hist_df = convert_n_grams_to_hist_df(two_grams)
    print('we have the two grams hist now', time.time())

    #three_grams_series = pd.Series(three_grams)
    # three_grams_series = three_grams_series.apply(",".join)
    #three_grams_series = three_grams_series.apply(constants.N_GRAM_SEPARATOR.join)
    #three_grams_hist = three_grams_series.value_counts()
    three_grams_hist_df = convert_n_grams_to_hist_df(three_grams)
    print('we have the three grams hist now', time.time())

    #four_grams_series = pd.Series(four_grams)
    #four_grams_series = four_grams_series.apply(constants.N_GRAM_SEPARATOR.join)
    #four_grams_hist = four_grams_series.value_counts()
    four_grams_hist_df = convert_n_grams_to_hist_df(four_grams)
    five_grams_hist_df = convert_n_grams_to_hist_df(five_grams)
    six_grams_hist_df = convert_n_grams_to_hist_df(six_grams)
    end = time.time()
    print("processing completed in", (end - start), "seconds")

    '''pd.DataFrame(two_grams_hist_df).to_csv(file_name.split("/")[-1] + "_2_grams.csv", index=True)
    pd.DataFrame(three_grams_hist_df).to_csv(file_name.split("/")[-1] + "_3_grams.csv", index=True)
    pd.DataFrame(four_grams_hist_df).to_csv(file_name.split("/")[-1] + "_4_grams.csv", index=True)
    pd.DataFrame(five_grams_hist_df).to_csv(file_name.split("/")[-1] + "_5_grams.csv", index=True)
    pd.DataFrame(six_grams_hist_df).to_csv(file_name.split("/")[-1] + "_6_grams.csv", index=True)'''

    two_grams_hist_df.to_csv(file_name.split("/")[-1] + "_2_grams.csv", index=False)
    three_grams_hist_df.to_csv(file_name.split("/")[-1] + "_3_grams.csv", index=False)
    four_grams_hist_df.to_csv(file_name.split("/")[-1] + "_4_grams.csv", index=False)
    five_grams_hist_df.to_csv(file_name.split("/")[-1] + "_5_grams.csv", index=False)
    six_grams_hist_df.to_csv(file_name.split("/")[-1] + "_6_grams.csv", index=False)

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
    ngrams_hist['prefix'] = ngrams_hist.gram.apply(split_prefix)
    ngrams_hist['target'] = ngrams_hist.gram.apply(lambda text: text.split(" ")[-1])
    prefix_map = {}
    for (index, gram, count, prefix, target) in ngrams_hist.itertuples():
        # print(gram, count, prefix, target)
        if prefix not in prefix_map:
            prefix_map[prefix] = [index]
        else:
            prefix_map[prefix].append(index)
    return prefix_map


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


if __name__ == "__main__":

    file_name = '../../courses/data_science_capstone/en_US/twitter_train.txt'
    # file_name = '../../courses/data_science_capstone/en_US/moby_dick_no_header.txt'
    # file_name = '../../courses/data_science_capstone/en_US/twitter_test_1.txt'
    file_name = '../../courses/data_science_capstone/en_US/twitter_sample.txt'
    file_name = '../../courses/data_science_capstone/en_US/en_US.twitter.txt'
    #file_name = '../../courses/data_science_capstone/en_US/en_US.news.txt'

    # process_one_file(file_name)

    output_file = "training_sentences.pkl"
    #'''
    sentences = get_training_sentences_from_file(file_name, 1000, min_num_words=6)
    with open(output_file, 'wb') as f:
       pickle.dump(sentences, f)
    #    '''

    #with open("training_sentences.pkl", 'rb') as f:
    #    sentences = pickle.load(f)
    training_df = get_training_df(sentences)
    print(training_df.head())

    # Save the training_df.  Both ways, and see which file is bigger.
    with open("training_df.pkl", 'wb') as f:
        pickle.dump(training_df, f)
    training_df.to_csv("training_df.csv", index=False)
    # Strange.  I generated training sentences and the data frame from the full twitter file.  The pkl df file was
    # 140640 KB.  The csv df was 134748 KB.  The pkl of the sentences list was almost 5 Gigs.  The full twitter file
    # itself is only 163189 KB.
    # That was with a bug where it used all the sentences.
    # 4294463 total sentences in the twitter file
    # With only 1000 sentences, the sentences list pkl is 2516 kb, training_df.pkl is 72 kb, and trainind_df.csv is
    # 68 kb.


