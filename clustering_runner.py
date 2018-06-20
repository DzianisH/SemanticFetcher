import pandas as pd
from sklearn import cluster

import utils

glove = utils.get_collection()

df_tsne = pd.read_csv("data/bot-search-metrics-id-tsne.csv")


queries = utils.convert_to_vec(df_tsne['Keyword'], utils.get_collection(100), 100)

print("Running k-means clustering...")
kmeans = cluster.KMeans(random_state=667, n_clusters=11, precompute_distances=True, n_init=64, max_iter=5000, tol=1e-5,
                            n_jobs=-1, copy_x=True).fit(queries)

df_tsne['class'] = kmeans.predict(queries)

df_tsne.to_csv('data/bot-search-metrics-id-tsne-clustered.csv')