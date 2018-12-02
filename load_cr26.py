#cf10k

from preprocess import *
import numpy as np

import pandas as pd

def load():

    files = ["crisis/2012_Colorado_wildfires-tweets_labeled.csv", "crisis/2012_Costa_Rica_earthquake-tweets_labeled.csv", "crisis/2012_Guatemala_earthquake-tweets_labeled.csv", "crisis/2012_Italy_earthquakes-tweets_labeled.csv", "crisis/2012_Philipinnes_floods-tweets_labeled.csv", "crisis/2012_Typhoon_Pablo-tweets_labeled.csv", "crisis/2012_Venezuela_refinery-tweets_labeled.csv", "crisis/2013_Alberta_floods-tweets_labeled.csv", "crisis/2013_Australia_bushfire-tweets_labeled.csv", "crisis/2013_Bohol_earthquake-tweets_labeled.csv", "crisis/2013_Boston_bombings-tweets_labeled.csv", "crisis/2013_Brazil_nightclub_fire-tweets_labeled.csv", "crisis/2013_Colorado_floods-tweets_labeled.csv", "crisis/2013_Glasgow_helicopter_crash-tweets_labeled.csv", "crisis/2013_LA_airport_shootings-tweets_labeled.csv", "crisis/2013_Lac_Megantic_train_crash-tweets_labeled.csv", "crisis/2013_Manila_floods-tweets_labeled.csv", "crisis/2013_NY_train_crash-tweets_labeled.csv", "crisis/2013_Queensland_floods-tweets_labeled.csv", "crisis/2013_Russia_meteor-tweets_labeled.csv", "crisis/2013_Sardinia_floods-tweets_labeled.csv", "crisis/2013_Savar_building_collapse-tweets_labeled.csv", "crisis/2013_Singapore_haze-tweets_labeled.csv", "crisis/2013_Spain_train_crash-tweets_labeled.csv", "crisis/2013_Typhoon_Yolanda-tweets_labeled.csv", "crisis/2013_West_Texas_explosion-tweets_labeled.csv"]

    data = []

    for file in files:

        df = pd.read_csv(file)
        df = df[ [' Tweet Text', ' Informativeness'] ]
        for i in range(len(df)):
            tweet = df[' Tweet Text'][i]
            label = df[' Informativeness'][i]
            data.append((tweet, label))

    return data

def get_tweets(data):
    tweets = []
    for d in data:
        tweets.append(d[0])
    # print(tweets)
    return tweets

def get_labels(data):
    labels = []
    for d in data:
        labels.append(d[1])
    return labels

def convert_labels_to_int(labels):
    ints = []
    for l in labels:
        if l=='Not related':
            ints.append(0)
        else:
            ints.append(1)

    return ints

def get_dataset(displayStats=False):

    print('\nLoading data ...\n')
    data = load()
    X = get_tweets(data)
    Y = get_labels(data)
    Z = convert_labels_to_int(Y)
    pX = preprocess(X, 'cr26', stem = True, tokenize = True, removeHandles = True, removeLinks = True, removeEmojis = True, removeSymbols = True, replaceSlang = True, verbose=False)

    # hashtags inverted index
    ht = extract_hashtags(pX)
    htD = collect_hashtags(extract_hashtags(pX))
    htc1 = 0
    for h in htD:
        if len(htD[h]) >= 5:
            htc1 += 1

    if displayStats:
        print('Number of Relevant Tweets:\t\t\t\t', sum(Z), '/', len(Z), '=', sum(Z) / len(Z) * 100, '%')
        print('Number of Non Relevant Tweets:\t\t\t\t', len(Z) - sum(Z), '/', len(Z), '=', (len(Z) - sum(Z)) / len(Z) * 100, '%')
        print('Number of distinct hashtags:\t\t\t\t', len(htD.keys()))
        print('Number of hashtags present in at least 5 tweets:\t', htc1, '=', htc1 / len(htD.keys()) * 100, '%')

    return (X, Z, (pX, htD, ht))

if __name__=='__main__':
    X, Z, (pX, htD, _) = get_dataset(displayStats = True)
