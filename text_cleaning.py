import numpy as np
import pandas as pd
from nltk.tokenize import MWETokenizer
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import jieba
import math
from opencc import OpenCC
from sklearn.cluster import DBSCAN
#Read stopwords from file
cc = OpenCC('s2t')
bg_stopwords_path="./dictionary/chinese_stopwords.txt"
bg_stopwords = [cc.convert(line.strip()) for line in open(bg_stopwords_path, 'r', encoding='utf-8').readlines()]

def words_list(comment,bg_stopwords):
    tokens_list = jieba.cut(comment)
    tokens_list = [char for char in tokens_list if char not in bg_stopwords] 
    stopwords_list = ['\n','「','」',' ']
    bg_stopwords += stopwords_list
    tokens_list = [char for char in tokens_list if char not in bg_stopwords] 

    def preprocess(tokens_list):
        mw_tokenizer = MWETokenizer()
        hd_list = []
        for word in tokens_list:
            tokens = word_tokenize(word)
            tokens=[token.lower() for token in tokens if token.isalpha()]
            hd_list += tokens

        '''# Stemming
        ps = PorterStemmer()
        #stemming an tokenize word list
        hd_politics = [ps.stem(word) for word in hd_politics]'''

        # Lemmatization
        wnl = WordNetLemmatizer()
        hd_list = [wnl.lemmatize(word) for word in hd_list]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        hd_list = [word for word in hd_list if not word in stop_words]
        return hd_list
    tokens_list = preprocess(tokens_list)
    return tokens_list

def doc_clean(comment_df):
    voc_all = []
    voc_part = []
    for i in range(len(comment_df)):
        voc_part = words_list(comment_df['comment'][i],bg_stopwords)
        voc_all.append(voc_part)

    # Calculating bag of words
    for i in range(0,len(voc_all)-1,1):
        word_set = set(voc_all[i]).union(set(voc_all[i+1]))
    #print(voc_all) #各留言擷取重要單詞
    doc_all = {}
    for i in range(0,len(voc_all),1):
        dic = {"doc"+str(i):voc_all[i]}
        doc_all.update(dic)
    #print(doc_all)
    # create tf function
    def tf(term, token_doc):
        tf = token_doc.count(term)/len(token_doc)
        return tf

    # create function to calculate how many doc contain the term 
    def numDocsContaining(word, token_doclist):
        doccount = 0
        for doc_token in token_doclist:
            if doc_token.count(word) > 0:
                doccount +=1
        return doccount
    

    # create function to calculate  Inverse Document Frequency in doclist
    def idf(word, token_doclist):
        n = len(token_doclist)
        df = numDocsContaining(word, token_doclist)
        return math.log10(n/df)

    #create bag words
    bag_words =[] # declare bag_words is a list
    for doc in doc_all.keys():
        bag_words += doc_all[doc]
    bag_words=set(bag_words)
    #print(bag_words)

    #calculate idf for every word in bag_words
    bag_words_idf={} # declare "bag_words_idf" data structure is dictionary 
    for word in bag_words:
        bag_words_idf[word]= idf(word,doc_all.values())
    #print(bag_words_idf)

    #calculate tfidf without normalization
    tfidf={} # declare tfidf dictionary to store tfidf value
    for doc in doc_all.keys():
        tfidf_doc={} # delare tfidf_doc as a dictionary to store tfidf of each doc
        for term in set(doc_all[doc]): # "Set" is a mutual data collection
            tfidf_doc[term]= tf(term,doc_all[doc]) * bag_words_idf[term] # calculate tfidf for each doc
        tfidf[doc]= tfidf_doc


    tfidf_dataframe = pd.DataFrame(tfidf).transpose()
    #print(tfidf_dataframe)

    #define a function to do cosine normalization a data dictionary
    def cos_norm(dic): # dic is distionary data structure
        dic_norm={}
        factor=1.0/np.sqrt(sum([np.square(i) for i in dic.values()]))
        for k in dic:
            dic_norm[k] = dic[k]*factor
        return dic_norm

    #calculate tfidf with cosine normalization
    tfidf_norm={} # declare tfidf dictionary to store tfidf value
    for doc in doc_all.keys():
        tfidf_doc={} # delare tfidf_doc as a dictionary to store tfidf of each doc
        for term in set(doc_all[doc]):
            tfidf_doc[term]= tf(term,doc_all[doc]) * bag_words_idf[term] # calculate tfidf for each doc
        tfidf_norm[doc]= cos_norm(tfidf_doc)
    #print(tfidf_norm)

    #convert to pandas frame with normalization
    tfidf_dataframe = pd.DataFrame(tfidf_norm).transpose() # Transpose from N*M to M*N

    comment_df['words'] = voc_all
    return tfidf_dataframe

def select_pos_neg_word(tfidf_dataframe):
    
    pos_words_eng=[]
    f = open('./dictionary/positive-words.txt',"r")
    count=0
    for line in f.readlines():
        count+=1
        if(count<37):continue
        else: pos_words_eng.append(line[:-1])
    f.close

    neg_words_eng=[]
    f = open('./dictionary/negative-words.txt',"r")
    count=0
    for line in f.readlines():
        count+=1
        if(count<37):continue
        else: neg_words_eng.append(line[:-1])
    f.close

    pos_words_ch=[]
    f = open('./dictionary/NTUSD_positive_unicode.txt',"r",encoding="utf-8")
    for line in f.readlines():
        pos_words_ch.append(line[:-1])
    f.close

    neg_words_ch=[]
    f = open('./dictionary/NTUSD_negative_unicode.txt',"r",encoding="utf-8")
    for line in f.readlines():
        neg_words_ch.append(line[:-1])
    f.close

    data_words = list(tfidf_dataframe.columns)
    pos_words_data=[]
    neg_words_data=[]
    for words in data_words:
        if(words in pos_words_eng):
            pos_words_data.append(words)
        elif(words in neg_words_eng):
            neg_words_data.append(words)
        elif(words in pos_words_ch):
            pos_words_data.append(words)   
        elif(words in neg_words_ch):
            neg_words_data.append(words)
    
    pos_words_rank = tfidf_dataframe[pos_words_data].sum().sort_values(ascending = False)
    neg_words_rank = tfidf_dataframe[neg_words_data].sum().sort_values(ascending = False)

    n = 10
    pos_words_selected = list(pos_words_rank[0:n].index)
    neg_words_selected = list(neg_words_rank[0:n].index)

    selected_words_data = tfidf_dataframe[pos_words_selected+neg_words_selected]
    selected_words_data.fillna(value=0, inplace=True)
    
    return selected_words_data
