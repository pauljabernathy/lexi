import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import lemma_utils as lu
import constants

POS_LIST = ['VERB', 'NOUN', 'ADJ', 'ADV']
POS_EXCLUDE = ['PUNCT']
QUANTITY_PER_SENTENCE = 'how many in each sentence'

V_COUNT = 'num sentences with a given number of verbs'
NS_COUNT = 'num sentences with a given number of nounss'
ADV_COUNT = 'num sentences with a given number of adverbs'
ADJ_COUNT = 'num sentences with a given number of adjectives'

# TODO: Deal with sentences separated by double spaces.
# TODO: Consistency on parameters - string, doc, or pkl file


def find_sentence_length(spacy_sentence, pos_exclude=POS_EXCLUDE):
    tokens = [x for x in spacy_sentence if x.pos_ not in pos_exclude]
    length = len(tokens)
    return length


def find_sentence_lengths(doc):
    lengths = [len([x for x in sent if x.pos_ not in POS_EXCLUDE]) for sent in doc.sents]
    return lengths


def find_sentence_lengths_2(doc):
    sents = [sent for sent in doc.sents]
    length2 = []
    for i in range(len(sents)):
        length = find_sentence_length(sents[i])
        length2.append(length)
    # lengths = [len([x for x in sent if x.pos_ not in pos_exclude]) for sent in doc.sents]
    return length2


def find_sentence_lengths_hist(doc):
    lengths = find_sentence_lengths(doc)
    hist = pd.Series(lengths).value_counts().sort_index()
    return hist


def find_sentence_lengths_map(doc):
    lengths_map = {}
    sentences = list(doc.sents)
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        # current_length = len(current_sentence)
        current_length = find_sentence_length(current_sentence)
        if current_length in lengths_map:
            lengths_map[current_length].append(i)
        else:
            lengths_map[current_length] = [i]
    map = pd.DataFrame({'length': list(lengths_map.keys()), 'sentences': list(lengths_map.values())})
    map = map.sort_values('length')
    map['count'] = map.sentences.apply(lambda l: len(l))
    # sns.barplot(x=map.length, y=map['count'])
    map = map[map.length > 1]
    # sns.barplot(x=map.length, y=map['count'])
    return map


# TODO: A way to look at text files that have not been "spacied" yet.
def plot_sentence_length_map(spacy_doc_file_path):
    with open(spacy_doc_file_path, 'rb') as f:
        doc = pickle.load(f)
    map = find_sentence_lengths_map(doc)
    hist = find_sentence_lengths_hist(doc)
    # sns.barplot(x=hist.index, y=hist.values)
    sns.barplot(x=map.length, y=map['count'])
    sns.lineplot(x=map.length, y=map['count'])
    plt.show()


def find_POSs_in_sentences_counts(doc):
    verbs_map = {}
    nouns_map = {}
    adj_map = {}
    adv_map = {}
    sentences = list(doc.sents)
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        num_verbs = find_num_part_of_speech(current_sentence, ['VERB', 'AUX']) #find_num_verbs(current_sentence)
        if num_verbs in verbs_map:
            verbs_map[num_verbs].append(i)
        else:
            verbs_map[num_verbs] = [i]

        num_nouns = find_num_part_of_speech(current_sentence, ['NOUN'])
        if num_nouns in nouns_map:
            nouns_map[num_nouns].append(i)
        else:
            nouns_map[num_nouns] = [i]

        num_adv = find_num_part_of_speech(current_sentence, ['ADV'])
        if num_adv in adv_map:
            adv_map[num_adv].append(i)
        else:
            adv_map[num_adv] = [i]

        num_adj = find_num_part_of_speech(current_sentence, ['ADJ'])
        if num_adj in adj_map:
            adj_map[num_adj].append(i)
        else:
            adj_map[num_adj] = [i]

    verbs = pd.DataFrame({QUANTITY_PER_SENTENCE: list(verbs_map.keys()), 'verb_sentences': list(verbs_map.values())})
    nouns = pd.DataFrame({QUANTITY_PER_SENTENCE: list(nouns_map.keys()), 'noun_sentences': list(nouns_map.values())})
    adverbs = pd.DataFrame({QUANTITY_PER_SENTENCE: list(adv_map.keys()), 'adv_sentences': list(adv_map.values())})
    adjectives = pd.DataFrame({QUANTITY_PER_SENTENCE: list(adj_map.keys()), 'adj_sentences': list(adj_map.values())})
    combined = verbs.merge(nouns, on=QUANTITY_PER_SENTENCE, how='outer')
    combined = combined.merge(adverbs, on=QUANTITY_PER_SENTENCE, how='outer')
    combined = combined.merge(adjectives, on=QUANTITY_PER_SENTENCE, how='outer')

    combined = combined.sort_values(QUANTITY_PER_SENTENCE)
    combined = combined.apply(lambda row: fill_nas(row))
    combined[V_COUNT] = combined.apply(lambda row: len(row.verb_sentences), axis=1)
    combined[NS_COUNT] = combined.apply(lambda row: len(row.noun_sentences), axis=1)
    combined[ADV_COUNT] = combined.apply(lambda row: len(row.adv_sentences), axis=1)
    combined[ADJ_COUNT] = combined.apply(lambda row: len(row.adj_sentences), axis=1)
    return combined


def isnan(value):
    a = pd.isna(value)
    if type(a) == np.ndarray:
        return False
    if a == True:
        return True
    if a == False:
        return False
    return False


def fill_nas(series):
    series = series.apply(lambda column: [] if isnan(column) else column)
    return series


def find_num_verbs(sentence):
    #for word in sentence:
    #    if word.pos_ == 'VERB':
    #        pass
    return len([word for word in sentence if word.pos_ in ['VERB', 'AUX']])  # AUX for things like "is"
    # TODO:  Find out where all AUX is used because the above might be wrong in some cases.
    # TODO:  Determine if you even want "AUX" words to be counted!


def find_num_part_of_speech(sentence, pos_list):
    return len([word for word in sentence if word.pos_ in pos_list])  # AUX for things like "is"


def find_pos_histogram(doc, poses_to_ignore=['PUNCT', 'SPACE']):
    """
    Compute the histogram of parts of speech.  You can get the same data from lemma_utils.get_token_info(
    doc).value_counts(), except value_counts does not compute the ratios, and this allows you to ignore certain parts of
    speech.  For evaluating style, this will give you the proportion of nouns, verbs, etc.  In the context of
    predicting the next word, this will give you the prior probability of the part of speech of the next word.
    :param doc:
    :param poses_to_ignore:
    :return:
    """
    pos_map = {}
    for token in doc:
        if token.pos_ in poses_to_ignore:
            continue
        if token.pos_ in pos_map:
            pos_map[token.pos_] = pos_map[token.pos_] + 1
        else:
            pos_map[token.pos_] = 1
    result = pd.DataFrame(pd.Series(pos_map), columns=["count"])
    result['ratio'] = result['count'] / result['count'].sum()
    result = result.sort_values("count", ascending=False)
    return result


def find_pos_histogram2(doc, poses_to_ignore=['PUNCT', 'SPACE']):
    result2 = pd.DataFrame(lu.get_token_info(doc).pos.value_counts())
    result2.columns = ['count']
    result2 = result2.loc[set(result2.index).difference(poses_to_ignore)]
    result2['ratio'] = result2['count'] / result2['count'].sum()
    result2 = result2.sort_values("count", ascending=False)
    return result2


def find_pos_histogram3(doc, poses_to_ignore=['PUNCT', 'SPACE']):
    pos = [getattr(t, 'pos_', None) for t in doc] if hasattr(doc[0], 'pos_') else None
    result2 = pd.DataFrame(pd.Series(pos).value_counts())
    #result2 = pd.DataFrame(lu.get_token_info(doc).pos.value_counts())
    result2.columns = ['count']
    result2 = result2.loc[set(result2.index).difference(poses_to_ignore)]
    result2['ratio'] = result2['count'] / result2['count'].sum()
    result2 = result2.sort_values("count", ascending=False)
    return result2


def make_pos_pos_map(doc):
    """
    For every part of speech in the document, make a histogram of the counts of the parts of speech that follow it.
    :param doc:
    :return:
    """
    pos_map = {}
    poses_to_ignore = ['PUNCT', 'SPACE']
    # Break it down by sentence, because the POS of the first in a sentence is hardly related if at all to the POS of
    # the last word of the preceding sentence.
    for sentence in doc.sents:
        words = [word for word in sentence]
        for i in range(len(words) - 1):
            pre = words[i]
            post = words[i + 1]
            #if pre.pos_ in poses_to_ignore or post.pos_ in poses_to_ignore:
            if False: # pre.pos_ in poses_to_ignore:
                continue
            if pre.pos_ not in pos_map:
                pos_map[pre.pos_] = {}
            if post.pos_ in pos_map[pre.pos_]:
                pos_map[pre.pos_][post.pos_] = pos_map[pre.pos_][post.pos_] + 1
            else:
                pos_map[pre.pos_][post.pos_] = 1
    return pos_map


def make_pos_pos_df(doc):
    map = make_pos_pos_map(doc)
    keys = list(map.keys())
    if len(keys) == 0:
        return
    pos = keys[0]
    column = f"count_after_{pos}"
    result_df = pd.DataFrame(pd.Series(map[pos]), columns=[column])
    result_df[constants.POS] = result_df.index
    # TODO:  See if you can do pd.merge without adding the column, just on the index.

    for pos in keys[1:]:
        column = f"count_after_{pos}"
        current_df = pd.DataFrame(pd.Series(map[pos]), columns=[column])
        current_df[constants.POS] = current_df.index
        result_df = result_df.merge(current_df, on=constants.POS, how="outer")

    result_df = result_df.fillna(0)

    # Now make the "pos" column the index, not just some column.
    result_df.index = result_df.pos
    result_df = result_df.drop(constants.POS, axis=1)
    return result_df


def make_word_pos_map(doc):
    """
    Make a mapping of words and the histogram of parts of speech following the word.  I suspect this is only useful
    for common words that are not nouns, verbs, adjectives, and adverbs.  Words like "the", "of", "it" etc.
    :param doc:
    :return:
    """
    pos_map = {}
    # Break it down by sentence, because the POS of the first in a sentence is hardly related if at all to the POS of
    # the last word of the preceding sentence.
    poses_to_ignore = ['PUNCT', 'SPACE']
    for sentence in doc.sents:
        words = [word for word in sentence]
        for i in range(len(words) - 1):
            pre = words[i]
            post = words[i + 1]
            word = pre.text.lower()
            if post.pos_ not in poses_to_ignore and pre.pos_ not in poses_to_ignore:
                if word not in pos_map:
                    pos_map[word] = {}
                if post.pos_ in pos_map[word]:
                    pos_map[word][post.pos_] = pos_map[word][post.pos_] + 1
                else:
                    pos_map[word][post.pos_] = 1
    return pos_map

def analyze_doc(spacy_doc):
    sentence_lengths_hist = find_sentence_lengths_hist(spacy_doc)
    print(sentence_lengths_hist)
    print(sentence_lengths_hist / sentence_lengths_hist.sum())
    pos_map = find_POSs_in_sentences_counts(spacy_doc)
    pos_hist = pd.Series([word.pos_ for word in spacy_doc]).value_counts()
    print(pos_hist)
    print(pos_hist / pos_hist.sum())
    sns.lineplot(x=pos_map[QUANTITY_PER_SENTENCE], y=pos_map[V_COUNT])
    sns.lineplot(x=pos_map[QUANTITY_PER_SENTENCE], y=pos_map[NS_COUNT])
    sns.lineplot(x=pos_map[QUANTITY_PER_SENTENCE], y=pos_map[ADV_COUNT])
    sns.lineplot(x=pos_map[QUANTITY_PER_SENTENCE], y=pos_map[ADJ_COUNT])
    plt.show()


if __name__ == '__main__':
    # plot_sentence_length_map('en_docs_lms/moby_no_header_md.pkl')
    # plot_sentence_length_map('fyj/fyj_md.pkl')
    with open('en_docs_lms/moby_no_header_md.pkl', 'rb') as f:
        moby = pickle.load(f)
    '''moby_map = find_POS_maps(moby)
    sns.lineplot(x=moby_map[QUANTITY_PER_SENTENCE], y=moby_map[V_COUNT])
    sns.lineplot(x=moby_map[QUANTITY_PER_SENTENCE], y=moby_map[NS_COUNT])
    sns.lineplot(x=moby_map[QUANTITY_PER_SENTENCE], y=moby_map[ADV_COUNT])
    sns.lineplot(x=moby_map[QUANTITY_PER_SENTENCE], y=moby_map[ADJ_COUNT])
    plt.show()
    print(moby_map)'''
    analyze_doc(moby)

    with open('fyj/fyj_md.pkl', 'rb') as f:
        fyj = pickle.load(f)
    analyze_doc(fyj)
    e = 'f'
