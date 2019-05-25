# Twitter Sentiment Analysis

## Requirements
nltk
spellchecker

## Dataset

The original dataset, train.csv has 99899 tweets.
However, for the purpose of doing text analytics only a small sample of the entire dataset is considered with 500 tweets

## Pre-processing

Pre processing usually depends on the type of data under analysis.
For twitter sentiment analysis, preprocessed steps followed are as follows:

1. Removing words containing a particular pattern
eg: tweets contained user names like @user1

2. Removing punctuations, numbers and apostrophes

3. Tokenization
4. Fixing the word length and doing spelling correction
 eg: converting juuuusssttttt to just

5. Removing stop words and words with length less than 2
6. Lemmatization
7. Some manual corrections
