# Twitter Sentiment Analysis

## Requirements
1. nltk
2. colab

## Dataset

The original dataset, train.csv has 99899 tweets.

## Pre-processing

Pre processing usually depends on the type of data under analysis.
For twitter sentiment analysis, preprocessed steps followed are as follows:

1. Removing words containing a particular pattern
eg: tweets contained user names like @user1

2. Removing punctuations, numbers and apostrophes

3. Tokenization
4. Fixing the word length and one could also perform spell correction
 eg: converting juuuusssttttt to just

5. Removing stop words and words with length less than 2
6. Lemmatization
7. Removal of rare and most frequently occuring words
8. Some manual corrections

## Models Implemented

1. Naive Bayes (57.23 % acc for testing dataset)
2. Logistic Regression  (57.24 % acc for testing dataset)
3. LSTM 
