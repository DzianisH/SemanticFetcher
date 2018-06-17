import pandas as pd
from pymongo import MongoClient

client = MongoClient()
db = client['glove_db']

dimensionality = 25
glove = db['glove_twitter_{}d'.format(dimensionality)]
print(glove)


def check_unique(df, column='id'):
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
    unknowns_number = 0
    for sentence in df[column]:
        for word in sentence.strip().split():
            val = glove.find_one({'word': word})
            if val is None:
                print("ERROR: can't find '{}' in glove database".format(word))
                unknowns_number += 1

    if unknowns_number is 0:
        print('all words from given sequences are known')
    else:
        print('ERROR: there are {} unknown words in sequence'.format(unknowns_number))


df = pd.read_csv('data/bot-search-metrics.csv')
check_unique(df)
check_unique(df, 'Keyword')

check_all_words_known(df, glove)
