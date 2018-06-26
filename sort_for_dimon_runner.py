import pandas as pd

df = pd.read_csv('data/bot-search-metrics-id-tsne-clustered.csv')

df = df.sort_values(by=['class', 'Volume (desc)'], ascending=False)

df.to_csv('data/bot-search-metrics-id-tsne-clustered-sorted.csv')
df.to_excel('data/bot-search-metrics-id-tsne-clustered-sorted.xls')