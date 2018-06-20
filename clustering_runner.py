import pandas as pd
import numpy as np
from sklearn import cluster

import utils

df_tsne = pd.read_csv("data/bot-search-metrics-id-tsne.csv")

queries = utils.convert_to_vec(df_tsne['Keyword'], utils.get_collection(100), 100)

print("Running DBSCAN clustering...")
classes = cluster.DBSCAN(n_jobs=-1, eps=6, min_samples=7).fit_predict(queries)

df_tsne['class'] = np.add([1], classes)

df_tsne.to_csv('data/bot-search-metrics-id-tsne-clustered.csv')
