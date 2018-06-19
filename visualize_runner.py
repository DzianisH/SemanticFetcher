import pandas as pd
from ggplot import *

df = pd.read_csv('data/bot-search-metrics-id-tsne-clustered.csv')

new_df = df[['x-tsne', 'y-tsne', 'class']]

chart = ggplot(new_df, aes(x='x-tsne', y='y-tsne', color='class')) \
        + geom_point(size=50, alpha=0.1) \
        + ggtitle("tSNE dimensions colored by classes")
# print(chart)
chart
chart.show()
