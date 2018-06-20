import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils

df_tsne = pd.read_csv("data/bot-search-metrics-id-tsne.csv")

queries: np.ndarray = utils.convert_to_vec(df_tsne['Keyword'], utils.get_collection(100), 100)


def get_kth_min(array, k):
    h = []
    for value in array:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(k)][-1]

# queries = queries[:500]

for min_neighbor in range(2, 16):
    distances = []
    for idx in range(queries.shape[0]):
        vec = queries[idx]
        # mtx = np.delete(queries, idx, 0)
        dist = np.square(np.subtract(vec, queries))
        dist = np.sqrt(np.sum(dist, 1))
        distances.append(get_kth_min(dist, min_neighbor))
        distances.sort()
    print(distances)
    plt.plot(range(queries.shape[0]), distances)
    plt.xlabel('$idx$')
    plt.ylabel('$D(idx, %d)$' % min_neighbor)
    plt.grid()
    plt.show()