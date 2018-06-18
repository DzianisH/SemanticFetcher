from ggplot import *
import pandas as pd

df = pd.read_csv('data/bot-search-metrics-tsne-clustered.csv')

chart = ggplot(df, aes(x='x-tsne', y='y-tsne', color='class')) \
        + geom_point(size=50, alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
print(chart)
chart