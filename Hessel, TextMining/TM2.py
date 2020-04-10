import pandas as pd

train=pd.read_csv('train_E6oV3lV.csv')

#%% Number of Words
#   Het bepalen van het aantal woorden in een document.
#   De intuitie hierachter is dat negatieve berichten vaak minder
#   woorden bevatten dan positieve. Voor het bepalen van het aantal
#   woorden maken we gebruik van split.

train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
print(train[['tweet', 'word_count']].head())
#%% Number of Characters

train['char_count'] = train['tweet'].str.len()
print(train[['tweet', 'char_count']].head())
#%% Average Word length

def avg_word(sentence):
    words = sentence.split()
#   lenwords=0
#   countwords=0
#   for w in words:
#       lenwords = lenwords + len(w)
#       countwords = countwords + 1
#   return lenwords / countwords
    
    return (sum(len(word) for word in words)/len(words))

train['avg_wordlength'] = train['tweet'].apply(lambda x: avg_word(x))
print(train['avg_wordlength'].head())

#%% Count Stopwords
#   Het verwijderen van stopwoorden is een belangrijk onderdeel van NLP. 
#   maar het tellen van het aantal stopwoorden geeft ook al info.

import nltk
nltk.download()
from nltk.corpus import stopwords
stop = stopwords.words('english') 

train['stop_words'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
print(train['stop_words'].head())