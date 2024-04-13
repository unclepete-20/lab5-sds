import pandas as pd
import numpy as np
import re
import sklearn
from six.moves import cPickle as pickle
from sklearn import feature_extraction, tree, model_selection, metrics


def H_entropy (x):
    # Calculate Shannon Entropy
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ] 
    H = - sum([ p * np.log2(p) for p in prob ]) 
    return H

def firstDigitIndex( s ):
    for i, c in enumerate(s):
        if c.isdigit():
            return i + 1
    return 0

def vowel_consonant_ratio (x):
    # Calculate vowel to consonant ratio
    x = x.lower()
    vowels_pattern = re.compile('([aeiou])')
    consonants_pattern = re.compile('([b-df-hj-np-tv-z])')
    vowels = re.findall(vowels_pattern, x)
    consonants = re.findall(consonants_pattern, x)
    try:
        ratio = len(vowels) / len(consonants)
    except: # catch zero devision exception 
        ratio = 0  
    return ratio

def H_entropy (x):
    # Calculate Shannon Entropy
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ] 
    H = - sum([ p * np.log2(p) for p in prob ]) 
    return H

def vowel_consonant_ratio (x):
    # Calculate vowel to consonant ratio
    x = x.lower()
    vowels_pattern = re.compile('([aeiou])')
    consonants_pattern = re.compile('([b-df-hj-np-tv-z])')
    vowels = re.findall(vowels_pattern, x)
    consonants = re.findall(consonants_pattern, x)
    try:
        ratio = len(vowels) / len(consonants)
    except: # catch zero devision exception 
        ratio = 0  
    return ratio

# ngrams: Implementation according to Schiavoni 2014: "Phoenix: DGA-based Botnet Tracking and Intelligence"
# http://s2lab.isg.rhul.ac.uk/papers/files/dimva2014.pdf

def ngrams(word, n):
    # Extract all ngrams and return a regular Python list
    # Input word: can be a simple string or a list of strings
    # Input n: Can be one integer or a list of integers 
    # if you want to extract multipe ngrams and have them all in one list
    
    l_ngrams = []
    if isinstance(word, list):
        for w in word:
            if isinstance(n, list):
                for curr_n in n:
                    ngrams = [w[i:i+curr_n] for i in range(0,len(w)-curr_n+1)]
                    l_ngrams.extend(ngrams)
            else:
                ngrams = [w[i:i+n] for i in range(0,len(w)-n+1)]
                l_ngrams.extend(ngrams)
    else:
        if isinstance(n, list):
            for curr_n in n:
                ngrams = [word[i:i+curr_n] for i in range(0,len(word)-curr_n+1)]
                l_ngrams.extend(ngrams)
        else:
            ngrams = [word[i:i+n] for i in range(0,len(word)-n+1)]
            l_ngrams.extend(ngrams)
#     print(l_ngrams)
    return l_ngrams

def ngram_feature(domain, d, n):
    # Input is your domain string or list of domain strings
    # a dictionary object d that contains the count for most common english words
    # finally you n either as int list or simple int defining the ngram length
    
    # Core magic: Looks up domain ngrams in english dictionary ngrams and sums up the 
    # respective english dictionary counts for the respective domain ngram
    # sum is normalized
    
    l_ngrams = ngrams(domain, n)
#     print(l_ngrams)
    count_sum=0
    for ngram in l_ngrams:
        if d[ngram]:
            count_sum+=d[ngram]
    try:
        feature = count_sum/(len(domain)-n+1)
    except:
        feature = 0
    return feature
    
def average_ngram_feature(l_ngram_feature):
    # input is a list of calls to ngram_feature(domain, d, n)
    # usually you would use various n values, like 1,2,3...
    return sum(l_ngram_feature)/len(l_ngram_feature)

def clasificacion(dataframe):

    df = dataframe

    # Derivacion de caracteristicas
    df['length'] = df['domain_tld'].str.len()
    df['digits'] = df['domain_tld'].str.count('[0-9]')
    df['entropy'] = df['domain_tld'].apply(H_entropy)
    df['vowel-cons'] = df['domain_tld'].apply(vowel_consonant_ratio)
    df['firstDigitIndex'] = df['domain_tld'].apply(firstDigitIndex)

    with open('d_common_en_words' + '.pickle', 'rb') as f:
            d = pickle.load(f)

    df['ngrams'] = df['domain_tld'].apply(lambda x: average_ngram_feature([ngram_feature(x, d, 1), 
                                                                    ngram_feature(x, d, 2), 
                                                                    ngram_feature(x, d, 3)]))



    df_final = df
    df_domain = df_final['domain_tld'] 
    df_final = df_final.drop(['domain_tld'], axis=1)

    Pkl_Filename = "Pickle_RL_Model.pkl"  

    with open(Pkl_Filename, 'rb') as file:  
        Pickled_LR_Model = pickle.load(file)

    prediccion = Pickled_LR_Model.predict(df_final)

    df_final['domain_tld'] = df_domain
    df_final['isDGA'] = prediccion
    df_final = df_final.drop(['length'], axis=1)
    df_final = df_final.drop(['digits'], axis=1)
    df_final = df_final.drop(['entropy'], axis=1)
    df_final = df_final.drop(['vowel-cons'], axis=1)
    df_final = df_final.drop(['firstDigitIndex'], axis=1)
    df_final = df_final.drop(['ngrams'], axis=1)


    return df_final