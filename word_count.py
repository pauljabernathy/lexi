import spacy
import pandas as pd
from time import time
import pickle

# TODO:  Don't instantiate the NLP objects here.  The calling code should do that.
#es = spacy.load('es_core_news_md')
#es = spacy.load('es_core_news_lg')
#es.max_length = 2260721
#es.max_length = 100000000

WORD = 'word'
COUNT = 'count'
POS = 'pos'
SENTENCES = 'sentences'
DATA_COLUMNS = {COUNT: 0, POS: 1, SENTENCES: 2}

DEFAULT_MAX_EXAMPLES = 20


def get_counts_one_file(file_name, nlp, max=2000):
    with open(file_name, encoding='utf-8') as f:
        text_as_list = f.readlines()

    counts = {}

    if max > len(text_as_list):
        max = len(text_as_list)

    for i in range(460, max):
        doc = nlp(text_as_list[i].lower())
        for token in doc:
            word = clean_word(token.lemma_)
            if token.is_punct or word.isspace() or word.isnumeric():
                continue
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] = counts[word] + 1

    result = pd.DataFrame(data={'lemma':list(counts.keys()), 'count': list(counts.values())})
    result.sort_values('count', ascending=False, inplace=True)
    result['pos'] = result['lemma'].apply(lambda lemma: nlp(lemma)[0].pos_)
    return result


def clean_word(word):
    word = word.replace('--', '')
    return word


def get_counts(file_names:list, nlp, max=2000):
    result = pd.DataFrame(columns=['lemma', 'count'])
    for file_name in file_names:
        result = result.append(get_counts_one_file(file_name, nlp, max))
    return result


def create_lemma_map_from_doc(doc):
    return create_lemma_map_from_sentences(doc.sents)


def create_lemma_map_from_sentences(sentences):
    lemma_map_start_time = time()
    lemma_map = {}
    sentence_number = 0
    for sentence in sentences:
        for token in sentence:
            if token.is_punct or token.text.isspace() or token.text.isnumeric():
                continue
            #if token.pos_ not in pos_list:
            #    continue
            lemma = token.lemma_ #clean_word(token.lemma_).lower()
            # Calling clean_word causes unexpected behavior when it encounters words like "--El".  Spacy has already
            # classified it as PUNCT.

            # Check for lemma in lemma map and the part of speech being token.pos_
            # Because the same word could be a different part of speech.
            # For example 'I run fast.  That is the first run of the program, this is the second run.'
            # Also, I think that sometimes spacy misclassifies things.

            # TODO:  Use lemma, token.pos_ as the key; will require a slight change in create_lemma_df
            if lemma in lemma_map and lemma_map[lemma][DATA_COLUMNS[POS]] == token.pos_:
                lemma_map[lemma][DATA_COLUMNS[COUNT]] += 1
                lemma_map[lemma][DATA_COLUMNS[SENTENCES]].add(sentence_number)
            else:
                if lemma in lemma_map:
                    # already in the lemma_map, but under a different POS, possibly an error but can't guarantee
                    lemma = lemma + token.pos_
                lemma_map[lemma] = [0, 0, 0]
                lemma_map[lemma][DATA_COLUMNS[COUNT]] += 1
                lemma_map[lemma][DATA_COLUMNS[POS]] = token.pos_
                lemma_map[lemma][DATA_COLUMNS[SENTENCES]] = {sentence_number}

        sentence_number += 1
        if sentence_number % 10000 == 0:
            serialize(create_lemma_df(lemma_map), 'lemma_map_backup.pkl')

    lemma_map_created_time = time()
    print(str(lemma_map_created_time - lemma_map_start_time), ' seconds to create the lemma map')
    df = create_lemma_df(lemma_map)
    end_time = time()
    print(str(end_time - lemma_map_created_time), ' seconds to create the final dataframe')
    return df


def create_lemma_map_from_file(file_names, nlp, pos_list=['VERB', 'NOUN', 'ADJ', 'ADV'], encoding='utf-8'):
    start_time = time()
    if type(file_names) is str:
        with open(file_names, encoding=encoding) as f:
            text = f.read()
    elif type(file_names) is list:
        text = ''
        for file_name in file_names:
            with open(file_name, encoding=encoding) as f:
                text += '\n'
                text = text + f.read()


    doc = nlp(text)
    doc_created_time = time()
    print(str(doc_created_time - start_time), ' seconds to create the spacy document object')
    serialize(doc, "doc_backup.pkl")
    doc_serialized_time = time()
    print(str(doc_serialized_time - doc_created_time), ' seconds to serialize')
    df = create_lemma_map_from_doc(doc)
    end_time = time()
    #print(str(end_time - lemma_map_created_time), ' seconds to create the final dataframe')
    print(str(end_time - start_time), ' seconds total')

    return doc, df


def create_lemma_df(lemma_map):
    data = list(zip(*lemma_map.values()))
    df = pd.DataFrame({WORD: list(lemma_map.keys()), COUNT: data[DATA_COLUMNS[COUNT]], POS: data[
        DATA_COLUMNS[POS]], SENTENCES: data[DATA_COLUMNS[SENTENCES]]})
    df = df.sort_values([COUNT, WORD], ascending=False)
    df.sentences = df.sentences.apply(lambda s: sorted(s))
    return df


def serialize(python_object, output_file_name):
    with open(output_file_name, 'wb') as f:
        pickle.dump(python_object, f)


def show_examples(doc, lemma_map, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    sentences = [x for x in doc.sents]  # TODO: If performance becomes and issue, pass in sentences instead of doc.
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_examples]:
        print('\n___\n', sentences[s])


def get_examples(doc, lemma_map, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    # TODO:  Have the option of not putting in a lemma map; in that case, it would create it.  Passing in the lemma
    #  map technically is redundant.  However, if there is a large document, you don't want to recreated the lemma
    #  map each time you look for examples of a new lemma.
    sentences = [x for x in doc.sents]  # TODO: If performance becomes and issue, pass in sentences instead of doc.
    examples = []
    if lemma_map[lemma_map.word == lemma].empty:
        return examples
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_examples]:
        #print('\n___\n', sentences[s])
        examples.append(str(sentences[s]).strip(" "))
    # TODO:  See if there is a way other than a for statement.
    return examples


def show_examples_multiple_docs(doc_lm_pairs, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    '''
    :param doc_lm_pairs: a list of tuples in the form of (doc, lemma_map)
    :param lemma: the lemma (example a verb infinitive) you want to find examples of
    :param max_examples: the maximum number of examples
    :return:
    '''
    examples = get_examples_multiple_docs(doc_lm_pairs, lemma, max_examples)
    for example in examples:
        print(example)


def get_examples_multiple_docs(doc_lm_pairs, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    examples = []
    for doc, lm in doc_lm_pairs:
        examples.extend(get_examples(doc, lm, lemma, max_examples))
    return examples


def show_examples_2(sentences, lemma_map, lemma, max_index=20):
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_index]:
        print('\n___\n', sentences[s])


def show_info(doc):
    for token in doc:
        print(
            f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} "lemma:"  {token.lemma_:{20}} {spacy.explain(token.tag_)}')


def clean_corpus_file(file_name, cleaned_file_name):
    encoding = 'iso-8859-15'
    to_remove = ['<doc', '</doc>', 'ENDOFARTICLE']
    file_text = ''
    with open(file_name, encoding=encoding) as f:
        should_copy_line = True
        for line in f.readlines():
            should_copy_line = True
            for item in to_remove:
                if item in line:
                    should_copy_line = False
                    break
            if should_copy_line:
                file_text += line
    if cleaned_file_name:
        with open(cleaned_file_name, 'w', encoding='utf-8') as cleaned:
            cleaned.write(file_text)
    return file_text


def unite_split_files(file_names):
    for file_name in file_names:
        with open(file_name, encoding='utf-8') as f:
            pass


def create_docs_and_serialize_docs(file_names, nlp):
    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as f:
            text = f.read()
            doc = nlp(text)
            output_file_name = file_name.split('.')[0] + ".pkl"
            serialize(doc, output_file_name)


if __name__ == "__main__":
    pass