#import spacy
import pandas as pd
DEFAULT_MAX_EXAMPLES = 20
ALL_LEMMA_VALUES = [None, '']


def show_examples(sentences, lemma_map, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_examples]:
        print('\n___\n', sentences[s])


def get_examples(sentences, lemma_map, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    # TODO:  Have the option of not putting in a lemma map; in that case, it would create it.  Passing in the lemma
    #  map technically is redundant.  However, if there is a large document, you don't want to recreated the lemma
    #  map each time you look for examples of a new lemma.
    #sentences = [x for x in doc.sents]  # TODO: If performance becomes and issue, pass in sentences instead of doc.
    examples = []
    if lemma_map[lemma_map.word == lemma].empty:
        return examples
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_examples]:
        #print('\n___\n', sentences[s])
        sentence = str(sentences[s]).strip(" ")
        if sentence not in examples:
            examples.append(sentence)
    # TODO:  See if there is a way other than a for statement.
    return examples


def filter_sentences_with_lemma(sentences, lemma_map, lemma, max_examples=DEFAULT_MAX_EXAMPLES):
    sentences_with_lemma = []
    for i in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_examples]:
        sentences_with_lemma.append(sentences[i])
    # TODO:  do the above with an apply, not a for statement
    return sentences_with_lemma


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
        examples.extend(get_examples(list(doc.sents), lm, lemma, max_examples))
    return examples


def show_examples_2(sentences, lemma_map, lemma, max_index=20):
    for s in list(lemma_map[lemma_map.word == lemma].iloc[0].sentences)[:max_index]:
        print('\n___\n', sentences[s])


def show_info(doc):
    print()
    for token in doc:
        #print(
        #    f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} "lemma:"  {token.lemma_:{20}}    {spacy.explain(token.tag_)}')
        '''ancestors = [a for a in token.ancestors]
        token.morph.__repr__()
        m = token.morph
        parent = ancestors[0] if (ancestors is not None and len(ancestors) > 0) else '?' '''
        # assert token.head == parent
        '''print(token.text, token.pos_, token.dep_, token.morph.to_dict(),  # token.head.text,  # token.head.pos_,
              # [child for child in token.children]
              token.lemma_
              )'''
        pass

    print()
    texts = [getattr(t, 'text', None) for t in doc]
    lemmas = [t.lemma_ for t in doc] if hasattr(doc[0], 'lemma_') else None
    pos = [getattr(t, 'pos_', None) for t in doc] if hasattr(doc[0], 'pos_') else None
    dep = [getattr(t, 'dep_', None) for t in doc]
    morph_dict = [t.morph.to_dict() for t in doc] if hasattr(doc[0], 'morph') else None
    head = [t.head.text for t in doc] if hasattr(doc[0], 'head') else None
    df = pd.DataFrame({'text': texts, 'lemma': lemmas, 'pos': pos, 'dep': dep, 'morph': morph_dict, 'head': head})
    print(df)


def search_doc_for_examples(doc, lemma, morphology_attributes):
    """
    Searches the doc for examples of the word matching the given criteria.
    :param doc:
    :param lemma:
    :param morphology_attributes:
    :return:
    """
    sentences = list(doc.sents)
    return find_examples_in_sentences(sentences, lemma, morphology_attributes)


def search_sentences_for_examples(sentences, lemma_map, lemma, morphology_attributes, max_examples=DEFAULT_MAX_EXAMPLES):
    if lemma not in ALL_LEMMA_VALUES:
        sentences_to_search = filter_sentences_with_lemma(sentences, lemma_map, lemma)
    else:
        sentences_to_search = sentences
    return find_examples_in_sentences(sentences_to_search, lemma, morphology_attributes)


def find_examples_in_sentences(sentences, lemma, morphology_attributes):
    """
    meant to be a private function
    :param sentences:
    :param lemma:
    :param morphology_attributes:
    :return:
    """
    result = []
    for sentence in sentences:
        for token in sentence:
            if token.lemma_ == lemma or lemma in ALL_LEMMA_VALUES:
                if match_all_attrs(token, morphology_attributes):
                    sentence_text = str(sentence).strip(" ").strip('\n').replace('\n', ' ')
                    if sentence_text not in result:
                        result.append(sentence_text)
    return result


def match_all_attrs(token, attrs_to_match):
    """
    Return true if all the values in the given dict match the morphology attributes of the token.  So attrs_to_match
    is a way of filtering out which tokens will be matched.
    Returns true for an empty dict.
    :param token:
    :param attrs_to_match:
    :return:
    """
    if attrs_to_match is None:
        return True
    token_attrs = token.morph.to_dict()
    for attr in attrs_to_match:
        if attr not in token_attrs or attrs_to_match[attr] != token_attrs[attr]:
            return False
    return True


def match_any_attrs(token, attrs_to_match):
    """
    Returns True if any of the values in the given dict match the morphology attributes of the token.
    :param token:
    :param attrs_to_match:
    :return:
    """
    if attrs_to_match is None:
        return False
    token_attrs = token.morph.to_dict()
    for attr in attrs_to_match:
        if attr in token_attrs and attrs_to_match[attr] == token_attrs[attr]:
            return True
    return False


def verb_form_hist(doc):
    verb_forms = {}
    for token in doc:
        token_attrs = token.morph.to_dict()
        if 'VerbForm' not in token_attrs:
            continue
        current_verb_form = token_attrs['VerbForm']
        if current_verb_form in verb_forms:
            verb_forms[current_verb_form] = verb_forms[current_verb_form] + 1
        else:
            verb_forms[current_verb_form] = 1
    return verb_forms

