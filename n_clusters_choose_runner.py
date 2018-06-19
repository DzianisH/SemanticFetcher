import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt

import utils

glove = utils.get_collection()

df_tsne = pd.read_csv("data/bot-search-metrics-id-tsne.csv")

queries = utils.convert_to_vec(df_tsne['Keyword'], utils.get_collection(100), 100)

n_clusters_range = range(2, 18)
print("Running k-means clustering for №clusters {}".format(n_clusters_range))
inertia = []
for n_clusters in n_clusters_range:
    kmeans = cluster.KMeans(random_state=667, n_clusters=n_clusters, precompute_distances=True, n_init=4, max_iter=3000, tol=1e-5,
                            n_jobs=-1, copy_x=True).fit(queries)
    loss = np.sqrt(kmeans.inertia_)
    inertia.append(loss)
    print('Got {:.2f} loss with {} clusters'.format(loss, n_clusters))

print(inertia)

diff = []
for idx in range(1, len(inertia)):
    diff.append(inertia[idx - 1] - inertia[idx])
plt.plot(n_clusters_range[1:], diff)
plt.xlabel('$k$')
plt.ylabel('$dJ(C_k)$')
plt.show()
