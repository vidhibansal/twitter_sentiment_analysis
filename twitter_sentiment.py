import numpy as np
import plac
import random 
from pathlib import Path
import sys
import argparse
import pandas as pd
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
from nltk.stem import WordNetLemmatizer, PosterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

train  = pd.read_csv('train.csv',encoding='latin-1')
print(train.head())

#function for removing words with a given pattern
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt= re.sub(i,'',input_txt)
        
    return (input_txt)

#for removing all the usernames 
train2 = train 
tidy_tweet = []
for tweet in train['SentimentText']:
    tweet = tweet.lower()
    tidy_tweet.append(remove_pattern(tweet,'@[\w]*'))
print(train2['tidy_tweet']= tidy_tweet)

#for removing punctuations, numbers  
train2['tidy_tweet'] = train2['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
print(train2['tidy_tweet'])

wordnet_lemmatizer = WordNetLemmatizer()
#tokenizes the words in a sentence, removes all the words with length 2 or less and lemmatizes the words

stop_words = set(stopwords.words('english'))
def my_tokenizer(s):
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if len(t) > 2]
    return tokens
    

tokens = []
for sentence in train2['tidy_tweet']:
    tokens.append(my_tokenizer(sentence))

#POS tagging (required for lemmatization)
tagged_tokens = []
for token in tokens:
    tagged_tokens.append(pos_tag(token))
print(tagged_tokens)

#Lemmatization
t1=[]
wordnet_lemmatizer = WordNetLemmatizer()
for tokens in tagged_tokens:
    for word, tag in tokens:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma += word + ' '
        else:
            lemma += wordnet_lemmatizer.lemmatize(word,wntag)+ ' '
    t1.append(lemma)
    lemma=''


for i in range(len(t1)):
    tweets.append(' '.join(t1[i]))

print (t1)
train2['tidy_tweet'] = t1

#Visualization

#Wordcloud for all the words
all_words = ' '.join([text for text in train2['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Wordcoud for negative words

negative_words = ' '.join([text for text in train2['tidy_tweet'][train2['Sentiment']==0]])

wordcloud = WordCloud(width=800, height=500).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

#positive words

positive_words = ' '.join([text for text in train2['tidy_tweet'][train2['Sentiment']==1]])

wordcloud = WordCloud(width=800, height=500).generate(positive_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


def tokens_to_vectors(token, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in token:
        i = word_index_map.get(t) #get the index of the word
        x[i]+=1  #get the corresponding frequency of the term 
    x = x/x.sum()    
    x[-1] = label
    return x

N = len(t1)
s = (N, len(word_index_map)+1)
from psutil import virtual_memory
mem = virtual_memory()
data=np.zeros(s,dtype=np.uint8)
i=0
for tt in train2['tidy_tweet'][train2['Sentiment']==1]:
    tokenize = tt.split()
    xy = tokens_to_vectors(tokenize,1)
    data[i,:] = xy
    i +=1



for tt in train2['tidy_tweet'][train2['Sentiment']==0]:
    tokenize = tt.split()
    xy = tokens_to_vectors(tokenize,0)
    data[i,:] = xy
    i+=1

np.random.shuffle(data)
X = data[:,:-1]
Y = data[:,-1]
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]