import re
import pandas as pd
import time

PUNCTUATION_REGEX = '[.?!]'  # '[\.?!]' is unnecessary; don't need the \ before the . but I don't understand why


# Cleansing

def cleanse(text):
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
    sentence = cleanse(sentence)
    return sentence.split(' ')


def split_to_sentences(text):
    p = re.compile(PUNCTUATION_REGEX)
    sentences = p.split(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
    return sentences


def tokenize_by_sentence(text):
    sentences = split_to_sentences(text)
    result = []
    for sentence in sentences:
        current_result = tokenize_string(sentence)
        if current_result is not None and current_result != ['']:
            result.append(current_result)
    return result


# Statistics

def find_word_stats(text):
    tokens = tokenize_string(text)
    tokens_pd = pd.Series(tokens)
    token_hist = tokens_pd.value_counts()
    stats = pd.DataFrame({'count': token_hist, 'fraction': token_hist / len(tokens)})
    stats['fraction'] = stats['count'] / len(tokens)
    stats['cum_sum'] = stats['count'].cumsum()
    stats['cum_frac'] = stats['cum_sum'] / len(tokens)
    return stats


def find_sentence_lengths_hist(list_of_sentences):
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


if __name__ == "__main__":

    file_name = '../../courses/data_science_capstone/en_US/twitter_train.txt'
    file_name = '../../courses/data_science_capstone/en_US/moby_dick_no_header.txt'

    start = time.time()
    with open(file_name, 'r', encoding='UTF-8') as f:
        file_text = f.read()
    word_stats_df = find_word_stats(file_text)
    sentences = tokenize_by_sentence(file_text)
    sentence_lengths = find_sentence_lengths_hist(sentences)
    two_grams = find_n_grams_list_of_lists(sentences, 2)
    three_grams = find_n_grams_list_of_lists(sentences, 3)
    end = time.time()
    print("completed in", (end - start), "seconds")