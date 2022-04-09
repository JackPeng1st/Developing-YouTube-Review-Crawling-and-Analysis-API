from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def word_cloud(group_df, file_name):
    word_list = []
    doc_word_list = list(group_df['words'])
    for words in doc_word_list:
        for word in words:
            word_list.append(word)

    font_path = 'TaipeiSansTCBeta-Regular.ttf' 
    plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
    wc1 = WordCloud(font_path,width=1600, height=800)
    wc1.generate(' '.join(word_list))
    #print(wc1.words_)
    wc1.to_file(file_name+".jpg")
    plt.imshow(wc1) 

def bar_chart(group_df, file_name):
	word_list = []
	doc_word_list = list(group_df['words'])
	for words in doc_word_list:
		for word in words:
			word_list.append(word)
			
	words_freq = pd.Series(word_list).value_counts()
	df = words_freq[0:10]
	plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
	plt.figure(figsize=(30,12),dpi=100)
	plt.bar(df.index,
			df.values, 
			width=0.5, 
			bottom=None, 
			align='center', 
			)
	for index,data in enumerate(df.values):
		plt.text(x=index , y =data , s=f"{data}" , fontdict=dict(fontsize=20))
	plt.xticks(rotation=45,fontsize=25)
	plt.yticks(fontsize=25)
	title='top 10 words'
	plt.title(title,fontsize=25)
	plt.savefig(file_name+".jpg")