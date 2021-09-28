import re
import pandas as pd


def cleanse(text):
    text = text.lower()
    text = text.replace("'", '')    # Remove apostrophes
    # text = re.sub('[^\w\s]', ' ', text)
    # Replace punct with a space so that when someone does something like <word comma word> you don't accidentally
    # transform it into one word.  We remove extra spaces in the next line.
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('\\n', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()     # Small think but remove a trailing space.
    return text


def tokenize(sentence):
    sentence = cleanse(sentence)
    return sentence.split(' ')


def find_word_stats(text):
    twitter_tokens = tokenize(text)
    tokens_pd = pd.Series(twitter_tokens)
    token_hist = tokens_pd.value_counts()
    stats = pd.DataFrame({'count': token_hist, 'fraction': token_hist / len(twitter_tokens)})
    stats['fraction'] = stats['count'] / len(twitter_tokens)
    stats['cum_sum'] = stats['count'].cumsum()
    stats['cum_frac'] = stats['cum_sum'] / len(twitter_tokens)
    return stats


def split_to_sentences(text):
    p = re.compile('[\.?!]')
    sentences = p.split(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
    return sentences


def tokenize_by_sentence(text):
    sentences = split_to_sentences(text)
    result = []
    for sentence in sentences:
        current_result = tokenize(sentence)
        if current_result is not None and current_result != ['']:
            result.append(current_result)
    return result


def find_n_grams_list_of_strings(input: list, n: int):
    ngrams = []
    if n <= 0:
        return ngrams
    for i in range(len(input) - n + 1):
        ngrams.append(input[i:(i+n)])
    return ngrams


def find_n_grams_list_of_lists(input: list, n: int):
    ngrams = []
    if input is None:
        return ngrams
    for item in input:
        current_ngrams = find_n_grams_list_of_strings(item, n)
        ngrams.extend(current_ngrams)
    return ngrams


def find_n_grams_from_text(text, n):
    lists_of_words = tokenize_by_sentence(text)
    ngrams = find_n_grams_list_of_strings(lists_of_words, n)
    return ngrams


if __name__ == "__main__":
    with open(r'C:\Users\paulj_1e1uzlz\courses\data_science_capstone\en_US/twitter_train.txt', 'r', encoding='UTF-8') as f:
        train = f.read()

    with open(r'C:\Users\paulj_1e1uzlz\courses\data_science_capstone\en_US/twitter_test.txt', 'r', encoding='UTF-8') as f:
        test = f.read()

    train_df = find_word_stats(train)
    a = 'b'