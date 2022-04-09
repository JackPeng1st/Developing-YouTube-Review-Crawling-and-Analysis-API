import requests
from pprint import pprint
from datetime import datetime
import numpy as np
import pandas as pd
import string
import os

class YoutubeSpider():
    def __init__(self, api_key):
        self.base_url = "https://www.googleapis.com/youtube/v3/"
        self.api_key = api_key

    def get_html_to_json(self, path):
        """組合 URL 後 GET 網頁並轉換成 JSON"""
        api_url = f"{self.base_url}{path}&key={self.api_key}"
        r = requests.get(api_url)
        if r.status_code == requests.codes.ok:
            data = r.json()
        else:
            data = None
        return data

    def get_video(self, video_id, part='snippet,statistics'):
        """取得影片資訊"""
        # jyordOSr4cI
        # part = 'contentDetails,id,liveStreamingDetails,localizations,player,recordingDetails,snippet,statistics,status,topicDetails'
        path = f'videos?part={part}&id={video_id}'
        data = self.get_html_to_json(path)
        if not data:
            return {}
        # 以下整理並提取需要的資料
        data_item = data['items'][0]

        try:
            # 2019-09-29T04:17:05Z
            time_ = datetime.strptime(data_item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # 日期格式錯誤
            time_ = None

        url_ = f"https://www.youtube.com/watch?v={data_item['id']}"

        info = {
            'id': data_item['id'],
            'channelTitle': data_item['snippet']['channelTitle'],
            'publishedAt': time_,
            'video_url': url_,
            'title': data_item['snippet']['title'],
            'description': data_item['snippet']['description'],
            'likeCount': data_item['statistics']['likeCount'],
            #'dislikeCount': data_item['statistics']['dislikeCount'],
            'commentCount': data_item['statistics']['commentCount'],
            'viewCount': data_item['statistics']['viewCount']
        }
        return info

    def get_comments(self, video_id, page_token='', part='snippet', max_results=100):
        """取得影片留言"""
        # jyordOSr4cI
        path = f'commentThreads?part={part}&videoId={video_id}&maxResults={max_results}&pageToken={page_token}'
        data = self.get_html_to_json(path)
        if not data:
            return [], ''
        # 下一頁的數值
        next_page_token = data.get('nextPageToken', '')

        # 以下整理並提取需要的資料
        comments = []
        for data_item in data['items']:
            data_item = data_item['snippet']
            top_comment = data_item['topLevelComment']
            try:
                # 2020-08-03T16:00:56Z
                time_ = datetime.strptime(top_comment['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                # 日期格式錯誤
                time_ = None

            if 'authorChannelId' in top_comment['snippet']:
                ru_id = top_comment['snippet']['authorChannelId']['value']
            else:
                ru_id = ''

            ru_name = top_comment['snippet'].get('authorDisplayName', '')
            if not ru_name:
                ru_name = ''

            comments.append({
                'reply_id': top_comment['id'],
                'ru_id': ru_id,
                'ru_name': ru_name,
                'reply_time': time_,
                'reply_content': top_comment['snippet']['textOriginal'],
                'rm_positive': int(top_comment['snippet']['likeCount']),
                'rn_comment': int(data_item['totalReplyCount'])
            })
        return comments, next_page_token


def youtube_crawl(path,video_id,YOUTUBE_API_KEY):
    youtube_spider = YoutubeSpider(YOUTUBE_API_KEY)

    video_info_dict = youtube_spider.get_video(video_id, part='snippet,statistics')
    print(video_info_dict)

    next_page_token = ''
    comment_lists = []
    while 1:
        comments, next_page_token = youtube_spider.get_comments(video_id, page_token=next_page_token)
        comment_lists.append(comments)
        # 如果沒有下一頁留言，則跳離
        if not next_page_token:
            break
    content_list = []
    positive_list = []
    re_comment = []
    name_list = []
    comment_df = pd.DataFrame()

    for comment_list in comment_lists:
        for comment in comment_list:
            content_list.append(comment['reply_content'])
            positive_list.append(comment['rm_positive'])
            re_comment.append(comment['rn_comment'])
            name_list.append(comment['ru_name'])
    comment_df['name'] = name_list
    comment_df['comment'] = content_list
    comment_df['positive_num'] = positive_list
    comment_df['re_comment_num'] = re_comment
    comment_df = comment_df.sort_values(['positive_num'],ascending=False)
    comment_df = comment_df.reset_index(drop=True)

    file = video_info_dict['title']
    exclude = set(string.punctuation)
    file = ''.join(ch for ch in file if ch not in exclude)
    os.makedirs('./'+file)
    os.makedirs('./'+file+'/dbscan')
    os.makedirs('./'+file+'/kmeans')
    comment_df.to_csv('./'+file+'/'+file+'.csv',index=False,encoding='utf-8-sig')

    return comment_df,file