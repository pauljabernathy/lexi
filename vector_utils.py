from scipy import spatial
import pandas as pd
import numpy as np
import constants


def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)


def find_closest_word_vectors(word, word_set, nlp):
    spacy_vocab = nlp.vocab
    q = spacy_vocab[word].vector
    if True: # q.sum() == 0:
        q = nlp(word).vector
    words = []
    similarities = []

    for current_word in word_set:
        try:
            similarities.append(cosine_similarity(spacy_vocab[current_word].vector, q))
            words.append(current_word)
        except Exception as e:
            #similarities.append(-2)
            pass
    # TODO:  If it can't compute the cosine similarity, should probably just skip it and not append
    # TODO:  Look at all items with invalid cosine similarity and see if any are valuable at all.  If not, don't add these.
    df = pd.DataFrame({constants.WORD: words, constants.SIMILARITY: similarities})
    df = df.sort_values(constants.SIMILARITY, ascending=False)
    return df


def find_closest_word_vectors_series(word, word_set, spacy_vocab):
    q = spacy_vocab[word].vector
    words = []
    similarities = []
    word_list_series = pd.Series(list(word_set))
    similarities = word_list_series.apply(lambda current_word: cosine_similarity(spacy_vocab[current_word].vector, q))
    df = pd.DataFrame({constants.WORD: word_list_series, constants.SIMILARITY: similarities})
    df = df.sort_values(constants.SIMILARITY, ascending=False)
    return df


def find_closest_word_vectors_from_matrix(word, similarity_matrix):
    """
    Find the closest word using word vectors.
    :param word:
    :param similarity_matrix:
    :return: a DataFrame
    """
    result = pd.DataFrame(columns=[constants.WORD, constants.POS, constants.SIMILARITY])
    if word in similarity_matrix:
        result = similarity_matrix[[word, constants.POS]].sort_values(word, ascending=False)
        result[constants.WORD] = result.index
        result.index = range(result.shape[0])
        result.columns = [constants.SIMILARITY, constants.POS, constants.WORD]
        result = result[[constants.WORD, constants.POS, constants.SIMILARITY]]
    else:
        pass
    return result


def make_word_similarity_matrix(words_to_use, nlp, matrix_size, spacy_words=None):
    """
    Construct a matrix that maps the similarity of the given words (as defined by the spacy .similiarity() function)
    to every other word.  More than half is redundant because the diagonal is all 1 and the matrix is symmetric about
    the diagonal.
    :param words_to_use: a list of words as strings
    :param nlp: the spacy object for the language
    :param matrix_size: how big the matrix should be
    :param spacy_words: list of words as spacy objects; optional, but if you don't pass it in, it calls nlp(word) for
    each word in the words list; providing a precomputed list of spacy objects allows it to run faster
    :return: a matrix_size x matrix_size numpy array
    """
    # TODO:  looks like words and spacy_words are redundant.  Why would you pass in spacy_words that are different
    #  from the word list?  Probably something that should be looked into.
    # TODO:  There is a defect in this.  If you pass in a spcay_words, it has to correspond to the strings in
    #  words_to_use.
    z = np.zeros([matrix_size, matrix_size])

    if matrix_size < len(words_to_use):
        words_to_use = words_to_use[:matrix_size]
    if spacy_words is None:
        spacy_words = [nlp(word) for word in words_to_use]

    sim = 0
    for i in range(matrix_size): # range(len(spacy_words)):
        # for j in range(matrix_size): # range(len(spacy_words)):
        for j in range(i): # range(len(spacy_words)):
            sim = spacy_words[i].similarity(spacy_words[j])
            z[i, j] = sim
            z[j, i] = sim
        z[i, i] = 1
    return z


def make_word_similarity_df_from_matrix(matrix, word_list, nlp):
    """
    Construct a DataFrame that maps the similarity of the given words (as defined by the spacy .similiarity() function)
    to every other word.  The matrix created by that function is numerical only.  The dataframe created here adds
    labels around those numbers.  Rows and columns are labeled with the word, and it adds a column for the part of
    speech of the word in the row.  This technically is a slight flaw because a word can have more than
    one part of speech.
    TODO:  See if there is another way to handle part of speech when looking at the actual sentence.  Maybe spacy can
    get a context dependent part of speech.
    speech.
    :param matrix: a square numpy array, presumably created by make_word_similarity_matrix()
    :param word_list: list of the words; must of course correspond to the words used to create the matrix
    :param nlp: the spacy object for the language
    :return: a pandas DataFrame
    """
    # assert len(word_list) == matrix.shape[0]
    # If the word list is longer, you can just take the top x number of words.
    # But if the word list is shorter, it just won't work.
    df = pd.DataFrame(data=matrix, columns=word_list, index=word_list)
    s = pd.Series(df.index).apply(lambda w: nlp(w)[0].pos_)
    s.index = df.index
    df[constants.POS] = s
    return df




