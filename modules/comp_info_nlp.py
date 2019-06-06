# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:05:45 2019

@author: JannikHartmann
"""

import os
wd = "C:/Users/JannikHartmann/Desktop/ECB DG-S SAT"

# set directory to code rep
os.chdir(wd)

df_name = "12_04_2161_filled_Raluca_export-2019-03-29.csv"

import pandas as pd
df = pd.read_csv(df_name, dtype=object, index_col=0)

num_results = 5

import nltk
#nltk.download()
# download fails

#tokens_df = df.iloc[:717,:].copy()

# SELECT Canada only
tokens_df = df[df["cntry"]=='CA'].copy()



# prblem: mussing value in summary 1, index 90 probalby caused by 
# highlighted snippet (search result service by google)
# at this stage: could be replaced by some stopword?

#if tokens_df.isnull().sum().sum()!=0:
#tokens_df = tokens_df.fillna('no')
# nans can be better handled by .astype(str)

for j in range(num_results):
    name = 'summary_'+str(j+1)
    tokens_df[name] = [a.split() for a in tokens_df[name].astype(str)]


# function to lowercase words
def to_lowercase(text):
    lower_text = [w.lower() for w in text]
    return lower_text

# to lowercase    
for j in range(num_results):
    name = 'summary_'+str(j+1)
    tokens_df[name] = [to_lowercase(a) for a in tokens_df[name]]

#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
# import stopwords manually as download is noot possible
stop_words = {'a','about','above','after','again','against','ain','all','am',
              'an','and','any','are','aren',"aren't",'as','at','be','because',
              'been','before','being','below','between','both','but','by','can',
              'couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',
              "doesn't",'doing','don',"don't",'down','during','each','few','for',
              'from','further','had','hadn',"hadn't",'has','hasn',"hasn't",
              'have','haven',"haven't",'having','he','her','here','hers',
              'herself','him','himself','his','how','i','if','in','into','is',
              'isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me',
              'mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself',
              'needn',"needn't",'no','nor','not','now','o','of','off','on',
              'once','only','or','other','our','ours','ourselves','out','over',
              'own','re','s','same','shan',"shan't",'she',"she's",'should',
              "should've",'shouldn',"shouldn't",'so','some','such','t','than',
              'that',"that'll",'the','their','theirs','them','themselves','then',
              'there','these','they','this','those','through','to','too','under',
              'until','up','ve','very','was','wasn',"wasn't",'we','were','weren',
              "weren't",'what','when','where','which','while','who','whom','why',
              'will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd",
              "you'll","you're","you've",'your','yours','yourself','yourselves'
              }

# add custom stopwords, e.g. company, corporate, news, dates, ...?


# stopw_word removal filter function
def stopword_filter(text):
    filtered_text = [w for w in text if not w in stop_words]
    return filtered_text

# remove stopwords
for j in range(num_results):
    name = 'summary_'+str(j+1)
    tokens_df[name] = [stopword_filter(a) for a in tokens_df[name]]


# stemming?
def stemmer(word_vec):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    word_vec = [ps.stem(w) for w in word_vec]
    return word_vec

for j in range(num_results):
    name = 'summary_'+str(j+1)
    tokens_df[name] = [stemmer(a) for a in tokens_df[name]]



# wordcloud per s11, s12x, -4
# get all tokens in one column
tokens_df["summary_all"] = tokens_df["summary_1"] + tokens_df["summary_2"] + tokens_df["summary_3"] + tokens_df["summary_4"] + tokens_df["summary_5"]

# create lists of unlisted lists; filtered by s11, s12x, -4
import itertools

list_na = list(itertools.chain.from_iterable(list(tokens_df[tokens_df["instttnl_sctr"]=='-4']["summary_all"])))
list_s11 = list(itertools.chain.from_iterable(list(tokens_df[tokens_df["instttnl_sctr"]=='S11']["summary_all"])))

set_12x = {'S121','S122','S123','S124','S125','S126','S127','S128','S129'}
list_s12x = list(itertools.chain.from_iterable(list(tokens_df[tokens_df["instttnl_sctr"].isin(set_12x)]["summary_all"])))


# show twenty most frequent words by s11, s12x, -4
from collections import Counter

list_na_counts = Counter(list_na)
list_s11_counts = Counter(list_s11)
list_s12x_counts = Counter(list_s12x)

list_na_counts.most_common(20)
list_s11_counts.most_common(20)
list_s12x_counts.most_common(20)

#-> maybe exclude {'canada','canadian','company', 'inc','corporation'}
#stop_words.update({'corporate', 'company', 'news'})


# word clouds function

def show_wordcloud(data, title = None):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        collocations=False,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

# show wordcloud for all three 
show_wordcloud(list_na)
show_wordcloud(list_s11)
show_wordcloud(list_s12x)


# ngrams; maybe not include ngrams in the first MVP
# LDA?


# cut df to s11, s12x only
tokens_df["instttnl_sctr"].value_counts()
# for GANs: disable this for GANs
tokens_df = tokens_df.loc[tokens_df["instttnl_sctr"].isin(set(list(set_12x) +['S11']))]


# tf-idf or other type of Document Term Matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def list_to_string(doc_list):
    doc_list = ' '.join(word for word in list(doc_list))
    return doc_list

# create corpus object as expected by sklearn tf-idf matrix library functions    
corpus = [list_to_string(a) for a in tokens_df["summary_all"]]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), use_idf=True, norm='l2', min_df=5, binary=True)
matrix = vectorizer.fit_transform(corpus)

matrix_feature_tokens = vectorizer.get_feature_names()

    
# assumes only s12x and s11 are existent
import numpy as np
tokens_df["y"] = np.where(tokens_df["instttnl_sctr"].isin(set_12x) , 's12x', 's11')
# for GANS use this isntead:
#tokens_df["y"] = tokens_df["instttnl_sctr"]

y_X = pd.DataFrame(matrix.toarray())
y_X.columns = matrix_feature_tokens
y_X["y"] = tokens_df["y"].values
y_X.index = tokens_df["nm_entty"].copy()
# add other features, as suggested by Raluca #check for NAs
y_X["lgl_frm"] = tokens_df["lgl_frm"].values
#y_X["lgl_frm"] = y_X["lgl_frm"].fillna('unknown')



y_X["ecnmc_actvty"] = tokens_df["ecnmc_actvty"].values  
#y_X["cty"] = tokens_df["cty"].values 

# replace NAs by -99: does not really improve performance
#y_X["ecnmc_actvty"] = y_X["ecnmc_actvty"].fillna('-99')



# add entity name
    
tokens_df["nm_entty"] = [a.split() for a in tokens_df["nm_entty"].astype(str)]
tokens_df["nm_entty"] = [to_lowercase(a) for a in tokens_df["nm_entty"]]
tokens_df["nm_entty"] = [stemmer(a) for a in tokens_df["nm_entty"]]

corpus = [list_to_string(a) for a in tokens_df["nm_entty"]]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), use_idf=True, norm='l2', 
                             min_df=10, binary=False, lowercase=True, 
                             analyzer='word')
matrix = vectorizer.fit_transform(corpus)

name_feat_tokens = vectorizer.get_feature_names()

name_feat = pd.DataFrame(matrix.toarray() * 1.0) # multiply by 2 for more weight
name_feat.columns = ['name_feat_'+name_feat_tokens[i] for i in name_feat.columns]
name_feat.index = y_X.index

y_X = y_X.join(name_feat)




#for GANs use this:
#y_X.to_csv("GANs_X_y_features")


y_X.to_csv("15_04_only_CA_S11_S12X_y_features")





