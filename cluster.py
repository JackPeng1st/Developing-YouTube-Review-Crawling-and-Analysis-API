from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd

def dbscan(selected_words_data,comment_df):
    clustering = DBSCAN(eps=0.3,min_samples=10).fit(selected_words_data)
    comment_df['cluster'] = clustering.labels_
    return comment_df
def kmeans(selected_words_data,comment_df):
    clustering = KMeans(n_clusters=3, random_state=0).fit(selected_words_data)
    comment_df['cluster'] = clustering.labels_
    return comment_df