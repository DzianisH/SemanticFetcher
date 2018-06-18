import _csv
from sklearn import cluster
import pandas as pd
import numpy as np

import utils

glove = utils.get_collection()

df_tsne = pd.read_csv("data/bot-search-metrics.csv")


queries = utils.convert_to_vec(df_tsne['Keyword'])

print("Running k-means clustering...")
kmeans = cluster.KMeans(n_clusters=500, n_init=4, max_iter=300, n_jobs=-1, copy_x=True).fit(queries)

df_tsne['class'] = kmeans.predict(queries)

df_tsne.to_csv('data/bot-search-metrics-tsne-clustered.csv')