import _csv
import time
from typing import Dict, List, Union

import pandas as pd
from pymongo import MongoClient

client = MongoClient()
db = client['glove_db']

dimensionality = 25
collection = db['glove_twitter_{}d'.format(dimensionality)]
print(collection)


def create_doc(bulks: pd.DataFrame, idx: int) -> Dict[str, Union[List[float], str]]:
    return {
        'word': bulks['word'][idx],
        'vec': [bulks['ort' + str(ort)][idx] for ort in range(1, dimensionality + 1)]
    }


df = pd.read_csv('data/glove.twitter.27B.25d.txt', delim_whitespace=True, quoting=_csv.QUOTE_NONE)

bulk_size = 32000
words_number = df.shape[0]
print("Going to convert", words_number, "items by", bulk_size, "in the batch")
start_time = time.time()
for first in range(0, words_number, bulk_size):
    bulks = df[first:(first + bulk_size)]
    docs = [create_doc(bulks, i + first) for i in range(bulks.shape[0])]

    collection.insert_many(docs)

    done = float(first + len(docs)) / words_number
    time_delay = time.time() - start_time
    print("Done: {:.2f}%, spent: {:.2f}sec, ETA: {:.2f}sec"
          .format(done * 100, time_delay, time_delay * (1 - done) / done))
