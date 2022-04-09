import numpy as np
import pandas as pd
import youtube_crawler
import text_cleaning
import cluster
import visualization
import json
from flask import Flask,request,jsonify
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

@app.route('/yt_crawler',methods=['POST'])
def postInput():
    
    insertValues = request.get_json()
    path = insertValues['path']
    video_id = path.split('v=')[1]
    YOUTUBE_API_KEY = " " #請放入自己申請的YouTube Data API Key
    comment_df,file = youtube_crawler.youtube_crawl(path,video_id,YOUTUBE_API_KEY)

    tfidf_dataframe = text_cleaning.doc_clean(comment_df)

    selected_words_data = text_cleaning.select_pos_neg_word(tfidf_dataframe)

    file_name = './'+file+"/top10_words"
    visualization.bar_chart(comment_df,file_name)

    file_name = './'+file+"/all_wordcloud"+"(num_"+str(len(comment_df))+")"
    visualization.word_cloud(comment_df,file_name)
    # DBSCAN
    comment_df_dbscan = cluster.dbscan(selected_words_data,comment_df)
    cluster_info = pd.Series(comment_df_dbscan['cluster']).value_counts()
	
    for clus in list(cluster_info.index):
        group_df = comment_df_dbscan[comment_df_dbscan['cluster']==clus].reset_index(drop=True)
        file_name = './'+file+"/dbscan/cluster"+str(clus)+"(num_"+str(cluster_info[clus])+")"
        visualization.word_cloud(group_df,file_name)
    # Kmeans 
    comment_df_Kmeans = cluster.kmeans(selected_words_data,comment_df)
    cluster_info = pd.Series(comment_df_Kmeans['cluster']).value_counts()

    for clus in list(cluster_info.index):
        group_df = comment_df_Kmeans[comment_df_Kmeans['cluster']==clus].reset_index(drop=True)
        file_name = './'+file+"/kmeans/cluster"+str(clus)+"(num_"+str(cluster_info[clus])+")"
        visualization.word_cloud(group_df,file_name)


    return jsonify({'Success':file})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000,debug=True, use_reloader=False)



