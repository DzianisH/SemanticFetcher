import _csv
import time
from typing import Dict, List, Union

import pandas as pd

import utils

glove = utils.get_collection()


def create_doc(bulks: pd.DataFrame, idx: int) -> Dict[str, Union[List[float], str]]:
    return {
        'word': bulks['word'][idx],
        'vec': [bulks['ort' + str(ort)][idx] for ort in range(1, utils.get_default_dimensionality() + 1)]
    }


def populate_database(glove_source, glove_collection):
    df = pd.read_csv(glove_source, delim_whitespace=True, quoting=_csv.QUOTE_NONE)
    bulk_size = 8192
    words_number = df.shape[0]
    print("Going to convert", words_number, "items by", bulk_size, "in the batch")
    start_time = time.time()
    for first in range(0, words_number, bulk_size):
        bulks = df[first:(first + bulk_size)]
        docs = [create_doc(bulks, i + first) for i in range(bulks.shape[0])]

        glove_collection.insert_many(docs)

        done = float(first + len(docs)) / words_number
        time_delay = time.time() - start_time
        print("Done: {:.2f}%, spent: {:.2f}sec, ETA: {:.2f}sec"
              .format(done * 100, time_delay, time_delay * (1 - done) / done))
    print('Indexing..')
    glove.create_index("word")


count = glove.count()
if count < 1000000:
    populate_database('data/glove.twitter.27B.{}d.txt'.format(utils.get_default_dimensionality()), glove)
else:
    print("wont populate database, as it already contain", count, "items")
