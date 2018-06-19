import time

import pandas as pd
from sklearn.manifold import TSNE

import utils

df = pd.read_csv('data/bot-search-metrics-id.csv')

queries = utils.convert_to_vec(df['Keyword'])

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=30.0, n_iter=10000)
tsne_results = tsne.fit_transform(queries)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

df_tsne: pd.DataFrame = df.copy()
df_tsne['x-tsne'] = tsne_results[:, 0]
df_tsne['y-tsne'] = tsne_results[:, 1]

df_tsne.to_csv("data/bot-search-metrics-id-tsne.csv")
