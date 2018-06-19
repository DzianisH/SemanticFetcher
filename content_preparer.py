import numpy as np
import pandas as pd

import utils

glove = utils.get_collection()


def check_unique(df, column='_id'):
    print("Checking all", column, "in unique...")
    vals = df[column]
    print("found", len(vals), column, "to check")
    val_set = set()
    duplicates_number = 0
    for val in vals:
        if val not in val_set:
            val_set.add(val)
        else:
            print("ERROR: found duplicate id:", val)
            duplicates_number += 1

    if duplicates_number is 0:
        print('no duplicates were wound')
        return True
    else:
        print('ERROR: found duplicate', column, 'in given data frame')
        return False


def check_all_words_known(df, glove, column='Keyword'):
    unknowns = {}
    sentences = []
    for sentence in df[column]:
        sen = ''
        for word in sentence.strip().split():
            w2v = utils.word2vec(word, glove)

            if w2v['word'].__contains__('__UNKNOWN__'):
                print("ERROR: can't find '{}' in glove database".format(word))
                occurs = unknowns.get(word)
                if occurs is None:
                    unknowns[word] = 1
                else:
                    unknowns[word] = occurs + 1
            sen += w2v['word'] + ' '
        sentences.append(sen)

    df['effective-keyword'] = sentences

    if len(unknowns) == 0:
        print('all words from given sequences are known')
        return True
    else:
        print('ERROR: there are {} unknown words in sequence'.format(len(unknowns)))
        items = []
        for key, value in unknowns.items():
            items.append((key, value))
        items.sort(key=lambda tuple2: tuple2[1], reverse=True)
        print(items)

        return False


df = pd.read_csv('data/bot-search-metrics.csv')
shape_ = [i for i in range(df.shape[0])]
array = np.array(shape_)
df['_id'] = array

is_ok = check_unique(df)
is_ok &= check_unique(df, 'Keyword')

is_ok &= check_all_words_known(df, glove)

if True:
    print("All checks OK, saving data")
    df.to_csv('data/bot-search-metrics-id.csv')
else:
    print("Won't save data frame, because some checks fails")
