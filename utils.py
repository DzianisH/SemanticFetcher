from pymongo import MongoClient
from pymongo.collection import Collection
import pandas as pd
import numpy as np

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
    default_vec = [0.0 for i in range(dimensionality)]
    for sentence in df:
        words = sentence.strip().split()
        vec = default_vec
        for word in words:
            entity = glove.find_one({'word': word})
            # print(entity)
            if entity is not None:
                vec = [vec[i] + entity['vec'][i] for i in range(dimensionality)]
        vecs.append(vec)

    vecs = np.array(vecs)
    print('converted')
    return vecs