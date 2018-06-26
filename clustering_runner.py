import pandas as pd
import numpy as np
from sklearn import cluster

df_tsne = pd.read_csv("data/bot-search-metrics-id-tsne.csv")

# queries = utils.convert_to_vec(df_tsne['Keyword'], utils.get_collection(100), 100)
queries = df_tsne[['x-tsne', 'y-tsne']].values

# print("Running DBSCAN clustering...")
# classes = cluster.DBSCAN(n_jobs=-1, eps=10, min_samples=10).fit_predict(queries)

# df_tsne['class'] = np.add([1], classes)

classes = cluster.KMeans(n_jobs=-1, n_clusters=8, n_init=128, max_iter=7000, tol=1e-6, verbose=0, precompute_distances=True)\
    .fit_predict(queries)

df_tsne['class'] = classes

df_tsne.to_csv('data/bot-search-metrics-id-tsne-clustered.csv')
