import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

client = MongoClient()
db = client['glove_db']

__dimensionality = 50


def get_collection(dim=__dimensionality) -> Collection:
    glove = db['glove_twitter_{}d'.format(dim)]
    print(glove)
    return glove


def get_default_dimensionality() -> int:
    return __dimensionality


def convert_to_vec(df: pd.DataFrame, glove=get_collection(), dimensionality=get_default_dimensionality()):
    vecs = []
    default_vec = __get_default_vector(glove)
    for sentence in df:
        words = sentence.strip().split()
        vec = default_vec
        for word in words:
            entity = word2vec(word, glove)
            # print(entity)
            if entity is not None:
                vec = [vec[i] + entity['vec'][i] for i in range(dimensionality)]
        vecs.append(vec)

    vecs = np.array(vecs)
    print('converted')
    return vecs


__default_vector = None


def __get_default_vector(glove=get_collection()):
    return __get_default_word2vec(glove)['vec']


def __get_default_word2vec(glove=get_collection()):
    global __default_vector
    if __default_vector is None:
        __default_vector = glove.find_one({'word': '__UNKNOWN__'})
    return __default_vector


def word2vec(word: str, glove: Collection, use_default=True):
    w2v = glove.find_one({'word': word})
    if w2v is not None:
        return {'word': w2v['word'], 'vec': w2v['vec']}
    w2v = __get_default_word2vec(glove).copy()
    w2v['word'] = ''

    for word in __retokenize(word):
        sub_w2v = glove.find_one({'word': word})
        if sub_w2v is None:
            if use_default:
                sub_w2v = __get_default_word2vec(glove)
            else:
                return None

        w2v['word'] = "{} {}".format(w2v['word'], sub_w2v['word'])
        w2v['vec'] = [w2v['vec'][i] + sub_w2v['vec'][i] for i in range(len(w2v))]

    return {'word': w2v['word'].strip(), 'vec': w2v['vec']}


replacements = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                "6": "six", "7": "seven", '8': 'eight', '9': 'nine', 'chatbot': 'chat bot', '"': '"',
                'tensorflow': 'tensor flow', 'cleverbot': 'organisation bot', 'eviebot': 'organisation bot',
                'boibot': 'organisation bot', 'pewdiebot': 'organisation bot', 'chimbot': 'organisation bot',
                'viewbot': 'organisation bot', 'www.clever.com': 'organisation bot',
                '?': '?', '.': '.', 'телеграм': 'telegram', 'bột': 'bot', 'thebot': 'the bot', '"bot"': '" bot "',
                'agario': 'customer', 'streamlabs': 'stream labs', 'nadeko': 'customer', 'jacksepticeye': 'customer',
                'overwatch': 'customer', 'aethex': 'customer', 'kahoot': 'customer', 'fortnite': 'customer',
                'spoopy': 'customer', 'fredboat': 'customer', 'welcomer': 'customer', 'meeseeks': 'customer',
                'pubg': 'customer', 'discoid': 'customer', 'bot': 'bot', 'c#': 'programming language'}


def __retokenize(word: str) -> []:
    if word.endswith("n't"):
        word = word[:len(word) - 3] + ' not'
    for key, value in replacements.items():
        word = word.replace(key, " {} ".format(value))

    return word.strip().split()
