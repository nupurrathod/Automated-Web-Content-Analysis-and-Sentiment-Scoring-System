import html
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import os
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
import numpy as np
import syllables

df=pd.read_excel('input.xlsx')

def scrape(url):
    response=requests.get(url)
    soup=BeautifulSoup(response.content,'html.parser')
    return soup.get_text()

text_list = []

for url in df['URL']:
    try:
        text = scrape(url)
        text_list.append(text)
        print(f'Successfully scraped {url}')
    except Exception as e:
        print(f'Error scraping {url}: {e}')

df['text'] = text_list

stop_word=set()
for filename in os.listdir(r'C:\Users\nupur\OneDrive\Desktop\NLP assignment blackcoffer\StopWords'):
    if filename.endswith('.txt'):
        with open(os.path.join(r'C:\Users\nupur\OneDrive\Desktop\NLP assignment blackcoffer\StopWords',filename)) as f:
            words=set(f.read().splitlines())
        stop_word.update(words)
text=df['text']
def remove_stop_words(text):
    word = nltk.word_tokenize(text)
    word = [word.lower() for word in word if word.lower() not in stop_word]
    return " ".join(word)
df['text'] = df['text'].apply(remove_stop_words)   
df.to_excel('Output Data Structure.xlsx', index=False)
scores=df['text']

with open(r'C:\Users\nupur\OneDrive\Desktop\NLP assignment blackcoffer\MasterDictionary\positive-words.txt') as f:
    positive_words = set(f.read().splitlines())
with open(r'C:\Users\nupur\OneDrive\Desktop\NLP assignment blackcoffer\MasterDictionary\negative-words.txt') as f:
    negative_words = set(f.read().splitlines())

sia = SentimentIntensityAnalyzer()


#POSITIVE & NEGATIVE SCORE

def get_sentiment_scores(text):
    pos_count = len([word for word in words if word in positive_words])
    neg_count = len([word for word in words if word in negative_words])
    sentiment_scores = sia.polarity_scores(text)
    pos_score = sentiment_scores['pos']
    neg_score = sentiment_scores['neg']
    return pos_score, neg_score, pos_count, neg_count

df['POSITIVE SCORE'] = 0.0
df['NEGATIVE SCORE'] = 0.0

for index, row in df.iterrows():
    text = row['text']
    pos_score, neg_score, pos_count, neg_count = get_sentiment_scores(text)
    df.at[index, 'POSITIVE SCORE'] = pos_score
    df.at[index, 'NEGATIVE SCORE'] = neg_score

#POLARITY SCORE

df['POLARITY SCORE'] = df['text'].apply(lambda x:(sia.polarity_scores(x)['pos'] - sia.polarity_scores(x)['neg'])/(sia.polarity_scores(x)['pos'] + sia.polarity_scores(x)['neg'])+0.000001)

#AVG SENTENCE LENGTH

df['AVG SENTENCE LENGTH'] = df['text'].apply(lambda x: textstat.avg_sentence_length(x))


#PERCENTAGE OF COMPLEX WORD

def get_percentage_complex_words(text):
    words = textstat.lexicon_count(text, removepunct=True)
    difficult_words = textstat.difficult_words(text)
    if words > 0:
        return (difficult_words / words) * 100
    else:
        return 0
    

df['PERCENTAGE OF COMPLEX WORDS'] = df['text'].apply(get_percentage_complex_words)

#FOG INDEX

df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])

#AVG NO OF WORDS PER SENTENCE

def avg_words_per_sentence(text):

    sentences = nltk.sent_tokenize(text)

    words_per_sentence = [len(nltk.word_tokenize(sent)) for sent in sentences]
    avg_words = sum(words_per_sentence) / len(words_per_sentence)

    return avg_words
df['AVG NUMBER OF WORDS PER SENTENCE'] = df['text'].apply(avg_words_per_sentence)

#WORD COUNT
df['WORD COUNT'] = df['text'].apply(lambda x: len(x.split()))

#SUBJECTIVE SCORE

df['SUBJECTIVE SCORE'] = df.apply(lambda row: (row['POSITIVE SCORE'] + row['NEGATIVE SCORE']) / (row['WORD COUNT'] + 0.000001), axis=1)

#SYLLABLE PER WORD

def get_syllables(word):
    return syllables.estimate(word)
def get_syllables_per_word(text):
    words = text.split()
    syllables_count = sum([get_syllables(word) for word in words])
    return syllables_count / len(words)
df['SYLLABLES PER WORD'] = df['text'].apply(get_syllables_per_word)

#COMPLEX WORD COUNT

def count_complex_words(text):
    tokens = nltk.word_tokenize(text)
    count = 0
    for word in tokens:
        syllable_count = get_syllables(word)
        if syllable_count > 2:
            count += 1
    return count
df['COMPLEX WORD COUNT'] = df['text'].apply(count_complex_words)

#PERSONAL PRONOUNS

def get_personal_pronoun_count(text):
    blob = TextBlob(text)
    personal_pronouns = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    count = 0
    for word, tag in blob.tags:
        if tag == 'PRP' and word.lower() in personal_pronouns:
            count += 1
    return count
df['PERSONAL PRONOUNS'] = df['text'].apply(get_personal_pronoun_count)

#AVG WORD LENGTH

def avg_word_length(text):
    words = text.split()
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words)

df['AVG WORD LENGTH'] = df['text'].apply(avg_word_length)

df=df.drop(['text'], axis=1)

df.to_excel('Output Data Structure.xlsx', index=False)

