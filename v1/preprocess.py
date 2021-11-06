# typed = Partial

import os
import json
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MIN_WORD_LEN = 4

# contractions + slang
with open(f"{DIR_PATH}/contractions/contractions.json", "r") as fc, open(
    f"{DIR_PATH}/contractions/slang.json", "r"
) as fs:
    contractions = dict(**json.load(fc), **json.load(fs))


def expand_contractions(text: str, contraction_mapping=contractions) -> str:
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(
            match
        ) or contraction_mapping.get(match.lower())

        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# remove urls
URL_PATTERN = (
    r"[A-Za-z0-9]+://[A-Za-z0-9%-_]+(/[A-Za-z0-9%-_])*(#|\\?)[A-Za-z0-9%-_&=]*"
)


def remove_urls(text: str) -> str:
    return re.sub(URL_PATTERN, " ", text)


# tokenize
def tokenize(text: str) -> list:
    return nltk.word_tokenize(text)


# lemmatize
lemmatizer = WordNetLemmatizer()


def lemmatize_noun(words: list) -> list:
    return [lemmatizer.lemmatize(word, pos="n") for word in words]


# stemmer
snowball = SnowballStemmer(language="english")


def stem_snowball(words: list) -> list:
    return [snowball.stem(word) for word in words]


# stop words (unnecessary words)
stop_words = set(stopwords.words("english"))


def remove_stop_words(words: list) -> list:
    return [word for word in words if word not in stop_words]


# numbers + punctuation
def remove_numbers_punctuation(words: list) -> list:
    return [word for word in words if re.match("[a-zA-z ]", word)]


# emojis or weird characters
def retain_alpha(words: list):
    return [word for word in words if word.isalpha()]


# combine processes for speed
# takes in string
# returns list of processed words
def preprocess(text: str) -> list:
    text = expand_contractions(remove_urls(text))
    words = tokenize(text)
    words = [snowball.stem(lemmatizer.lemmatize(word, pos="n")) for word in words]
    words = [
        word
        for word in words
        if len(word) >= MIN_WORD_LEN
        and word not in stop_words
        and re.match("[a-zA-Z ]", word)
        and word.isalpha()
    ]
    return words
