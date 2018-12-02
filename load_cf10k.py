#cf10k

from preprocess import *
import numpy as np

def load():

    data = []

    with open('cf10k.csv') as f:
        lines = f.readlines()

    for line in lines:
        tweetElements = line.strip().split(',')
        # print(tweetElements)
        id = tweetElements[0]
        label = tweetElements[1]
        keyword = tweetElements[2]
        location = tweetElements[3]
        tweet = ','.join(tweetElements[4:])

        data.append((id, label, keyword, location, tweet))

    return data

def get_tweets(data):
    tweets = []
    for d in data:
        tweets.append(d[-1])
    return tweets

def get_labels(data):
    labels = []
    for d in data:
        labels.append(d[1])
    return labels

def convert_labels_to_int(labels):
    ints = []
    for l in labels:
        if l=='Relevant':
            ints.append(1)
        elif l=='Not Relevant':
            ints.append(0)
        else:
            print(l)
            raise Exception('unknown label')
    return ints

def get_keywords(data):
    keywords = []
    for d in data:
        keywords.append(d[2])
    return keywords

def get_dataset(displayStats=False):

    print('\nLoading data ...\n')
    data = load()
    X = get_tweets(data)
    Y = get_labels(data)
    Z = convert_labels_to_int(Y)

    pX = preprocess(X, 'cf10k', stem = True, tokenize = True, removeHandles = True, removeLinks = True, removeEmojis = True, removeSymbols = True, replaceSlang = True, verbose=False)

    # hashtags inverted index
    htD = collect_hashtags(extract_hashtags(pX))
    htc1 = 0
    for h in htD:
        if len(htD[h]) >= 5:
            htc1 += 1

    # keyword inverted index
    kw = get_keywords(data)
    kwD = {}
    for i,k in enumerate(kw):
        if k not in kwD:
            kwD[k] = [i]
        else:
            kwD[k].append(i)

    # keyword count list
    kwcL = []
    for k in kwD:
        kwcL.append(len(kwD[k]))
    kwmean = np.mean(kwcL)
    kwstd = np.std(kwcL)

    if displayStats:
        print('Number of Relevant Tweets:\t\t\t\t', sum(Z), '/', len(Z), '=', sum(Z) / len(Z) * 100, '%')
        print('Number of Non Relevant Tweets:\t\t\t\t', len(Z) - sum(Z), '/', len(Z), '=', (len(Z) - sum(Z)) / len(Z) * 100, '%')
        print('Number of distinct hashtags:\t\t\t\t', len(htD.keys()))
        print('Number of hashtags present in at least 5 tweets:\t', htc1, '=', htc1 / len(htD.keys()) * 100, '%')
        print('Number of distinct keywords:\t\t\t\t', len(kwD.keys()))
        print('Mean number of tweets per keyword:\t\t\t', kwmean)
        print('Std dev of tweets per keyword:\t\t\t\t', kwstd)

    return (X, Z, (pX, htD, kwD))

if __name__=='__main__':
    X, Z, _ = get_dataset(displayStats = True)
    preprocess(X, 'cf10k', stem = True, tokenize = True, removeHandles = True, removeLinks = True, removeEmojis = True, removeSymbols = True, replaceSlang = True, verbose=False)
    # print(data)
