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
    # df = pd.DataFrame(columns=[constants.WORD, constants.SIMILARITY])
    s2 = pd.DataFrame(columns=[constants.WORD, constants.POS, constants.SIMILARITY])
    if word in similarity_matrix:
        # s = similarity_matrix[word].sort_values(ascending=False)
        # df[constants.WORD] = s.index
        # df[constants.SIMILARITY] = s.values
        s2 = similarity_matrix[[word, constants.POS]].sort_values(word, ascending=False)
        s2[constants.WORD] = s2.index
        s2.index = range(s2.shape[0])
        s2.columns = [constants.SIMILARITY, constants.POS, constants.WORD]
        s2 = s2[[constants.WORD, constants.POS, constants.SIMILARITY]]
    else:
        pass
    return s2


def make_word_similarity_matrix(words, nlp, matrix_size, spacy_words=None):
    #vocab = list(nlp.vocab.strings)
    #spacy_word_set = set(v.lower() for v in vocab)
    #spacy_word_list_series = pd.Series(list(spacy_word_set))
    # matrix = pd.DataFrame(columns=spacy_word_list_series, index=spacy_word_list_series) # numpy.core._exceptions._ArrayMemoryError: Unable to allocate 929. GiB for an array with shape (353032, 353032)
    #print(matrix.head())
    #words = word_hist.word[:matrix_size]
    z = np.zeros([matrix_size, matrix_size])
    #z = np.zeros([10, 10])

    '''sample = words[:10]
    for i in range(len(sample)):
        for j in range(len(sample)):
            z[i, j] = nlp(words[i]).similarity(nlp(words[j]))
    print(z)'''
    if matrix_size < len(words):
        words = words[:matrix_size]
    if spacy_words is None:
        spacy_words = [nlp(word) for word in words]

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
    # assert len(word_list) == matrix.shape[0]
    # If the word list is longer, you can just take the top x number of words.
    # But if the word list is shorter, it just won't work.
    df = pd.DataFrame(data=matrix, columns=word_list, index=word_list)
    s = pd.Series(df.index).apply(lambda w: nlp(w)[0].pos_)
    s.index = df.index
    df[constants.POS] = s
    return df




