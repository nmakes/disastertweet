# learn.py

from sklearn.linear_model import LogisticRegression
from gensim import corpora, models
from load_cf10k import get_dataset as get_cf10k_dataset
from load_cr26 import get_dataset as get_cr26_dataset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def run_on_cf10k():

    X, Z, (pX, _, _) = get_cf10k_dataset()
    documents = [[word for word in pXline] for pXline in pX]

    dictionary = corpora.Dictionary(documents)
    n_items = len(dictionary)
    print(n_items)
    # print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in documents]
    # print(corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    ds = []
    for doc in tqdm(corpus_tfidf):
    	d = [0] * n_items
    	for index, value in doc :
    		d[index]  = value
    	ds.append(d)

    lsimodel = models.LsiModel(corpus_tfidf, id2word=dictionary)
    vectorized_corpus = lsimodel[corpus_tfidf]
    Z = np.array(Z)

    # print(vectorized_corpus[0])
    vectorized_tweets = []

    second = lambda T: T[1]

    for tw in vectorized_corpus:
        tweet = np.array(list(map(second, tw)))
        vectorized_tweets.append(tweet)

    vectorized_tweets = np.array(vectorized_tweets)

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(vectorized_tweets, Z)
    pr = []
    re = []
    f1 = []

    for i, (train_idx, test_idx) in enumerate(skf.split(vectorized_tweets, Z)):
        X_train, X_test = vectorized_tweets[train_idx], vectorized_tweets[test_idx]
        y_train, y_test = Z[train_idx], Z[test_idx]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        p, r, f = precision_score(preds, y_test), recall_score(preds, y_test), f1_score(preds, y_test)
        pr.append(p)
        re.append(r)
        f1.append(f)

        print('fold', i+1, 'precision:', p, 'recall:', r, 'f1:', f)

    print('mean precision:', np.mean(pr))
    print('std precision:', np.std(pr))
    print('mean recall:', np.mean(re))
    print('std recall:', np.std(re))
    print('mean f1:', np.mean(f1))
    print('std f1:', np.std(f1))

def run_on_cr26():

    X, Z, (pX, _, _) = get_cr26_dataset()
    documents = [[word for word in pXline] for pXline in pX]

    dictionary = corpora.Dictionary(documents)
    n_items = len(dictionary)
    # print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in documents]
    # print(corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    ds = []
    for doc in tqdm(corpus_tfidf):
    	d = [0] * n_items
    	for index, value in doc :
    		d[index]  = value
    	ds.append(d)

    lsimodel = models.LsiModel(corpus_tfidf, id2word=dictionary)
    vectorized_corpus = lsimodel[corpus_tfidf]
    Z = np.array(Z)

    # print(vectorized_corpus[0])
    vectorized_tweets = []

    second = lambda T: T[1]

    for tw in vectorized_corpus:
        tweet = np.array(list(map(second, tw)))
        vectorized_tweets.append(tweet)

    vectorized_tweets = np.array(vectorized_tweets)

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(vectorized_tweets, Z)
    pr = []
    re = []
    f1 = []

    for i, (train_idx, test_idx) in enumerate(skf.split(vectorized_tweets, Z)):
        X_train, X_test = vectorized_tweets[train_idx], vectorized_tweets[test_idx]
        y_train, y_test = Z[train_idx], Z[test_idx]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        p, r, f = precision_score(preds, y_test), recall_score(preds, y_test), f1_score(preds, y_test)
        pr.append(p)
        re.append(r)
        f1.append(f)

        print('fold', i+1, 'precision:', p, 'recall:', r, 'f1:', f)

    print('mean precision:', np.mean(pr))
    print('std precision:', np.std(pr))
    print('mean recall:', np.mean(re))
    print('std recall:', np.std(re))
    print('mean f1:', np.mean(f1))
    print('std f1:', np.std(f1))

print('\nCrowdFlower10K Dataset Results\n')
run_on_cf10k()

print('\nCrisisLex26 Dataset Results\n')
run_on_cr26()
