PUNCTUATION_REGEX = '[.?!]'  # '[\.?!]' is unnecessary; don't need the \ before the . but I don't understand why
KEY_COLUMN_NAME = "item"
GRAM_COLUMN_NAME = "gram"
COUNT_COLUMN_NAME = "count"
FRACTION_COLUMN_NAME = 'fraction'
PERCENTS_COLUMN_NAME = "percents"
CUM_SUM_COLUMN_NAME = "cum_sum"
CUM_FRAC_COLUMN_NAME = "cum_frac"
N_GRAM_SEPARATOR = " "       # For when representing n grams as a string of several words instead of a list

SOURCE = 'source'
TARGET = 'target'

WORD = "the_word"     # TODO:  Go back and see what all this could mess up!
SIMILARITY = "similarity"
POS = 'pos'

DEFAULT_NGRAMS_THRESHOLD = 10
DEFAULT_TOP_NGRAMS = 25
DEFAULT_TOP_ASSOCIATIONS = 10

SUM_COLUMN_NAME = "sum"
PRODUCT_COLUMN_NAME = "product"
SUM_SQ_COLUMN_NAME = "sum_sq"
PRODUCT_SQ_COLUMN_NAME = "prd_sq"

NGRAM = "ngram"
VECTOR = "vector"
NEITHER = "neither"

GENERIC_SIMILARITY = 0.12434165442582384  # The mean value of one of the vector associations matrix, I think news.
