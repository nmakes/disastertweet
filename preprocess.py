# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize.casual import remove_handles
import numpy

# Import other files
import config # config file
import load_cf10k

# Tokenization
from nltk.tokenize import TweetTokenizer
tk=TweetTokenizer()

# Stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

# Removing Emojis
from emoji import UNICODE_EMOJI

# load special data from config file
symbols = config.symbols
slang = config.slang

def remove_emojis(tweet):

	if type(tweet)==str:
		words = tweet.split(' ')
	else:
		words = tweet
	# print('words', words)
	newTweet = []
	for word in words:
		shouldAppend = True
		for c in word:
			if c in UNICODE_EMOJI:
				shouldAppend = False
				break
		if shouldAppend:
			newTweet.append(word)

	return newTweet

def remove_links(tweet):

	tweetWords = tweet.split()
	newTweet = []

	for w in tweetWords:
		if ('http://' not in w) and ('https://' not in w):
			newTweet.append(w)

	return ' '.join(newTweet)

def remove_symbols(tweet):

	newTweet = []
	for word in tweet.split():
		# print(word)
		if word not in symbols:
			newTweet.append(word)
	return ' '.join(newTweet)

def replace_slang(tweet):
	global slang
	newTweet = []
	# print('-- REPLACESLANG', tweet)
	for i,w in enumerate(tweet):
		if w.lower() in slang.keys():
			for expansion in slang[w.lower()]:
				newTweet.append(expansion)
		else:
			newTweet.append(w)

	return newTweet

def preprocess(lines, dataset, stem = True, tokenize = True, removeHandles = True, removeLinks = True, removeEmojis = True, removeSymbols = True, replaceSlang = True, verbose = True):

	# Input: Raw lines from the dataset file
	# Output: Preprocessed tweet (or list of tokens if tokenize = True)

	tweets = []

	for i,line in enumerate(lines):

		if dataset == 'DisasterTweet':
			tweetElements = line.strip().split(",")[:-4]

			for i in range(len(tweetElements)):
				tweetElements[i] = tweetElements[i].strip()

			for k in tweetElements:
				if k=='':
					tweetElements.remove(k)

			tweet = ",".join(tweetElements)

		elif dataset == 'cf10k':
			tweet = line

		elif dataset == 'cr26':
			tweet = line

		else:

			raise Exception('No dataset selected')
			tweet = None

		if verbose:
			print('original:', tweet)

		if removeHandles:
			tweet = remove_handles(tweet)

			if verbose:
				print('removeHandles:', tweet)

		if removeLinks:
			tweet = remove_links(tweet)

			if verbose:
				print('removeLinks:', tweet)

		if stem:
			tweet = [stemmer.stem(word) for word in tweet.split() if (word not in symbols)]
			tweet = " ".join(tweet)

			if verbose:
				print('stem:', tweet)

		if tokenize:
			tweet = tk.tokenize(tweet)

			if verbose:
				print('tokenize', tweet)

		if removeEmojis:
			tweet = remove_emojis(tweet)

			if verbose:
				print('removeEmojis:', tweet)

		if removeSymbols:

			for w in tweet:
				if w in symbols:
					tweet.remove(w)

			if verbose:
				print('removeSymbols:', tweet)

		if replaceSlang:

			if type(tweet)==str:
				tweet = tk.tokenize(tweet)

			tweet = replace_slang(tweet)

			if verbose:
				print('replaceSlang:', tweet)

		# ---
		if verbose:
			print('\n')
		# ---

		tweets.append(tweet)

	return tweets

def extract_hashtags(tweets):
	hashtags = []

	if type(tweets[0])==str:
		for i,tweet in tweets:
			tweets[i] = tk.tokenize(tweet)

	for tweet in tweets:
		tempHT = []
		for w in tweet:
			if w[0]=='#' and len(w) > 1:
				tempHT.append(w)
		hashtags.append(tempHT)

	return hashtags

def collect_hashtags(hashtags):
	hashtagsDict = {}

	for i,ht in enumerate(hashtags):
		if ht!=[]:
			for h in ht:
				if h.lower() not in hashtagsDict:
					hashtagsDict[h.lower()] = [i]
				else:
					hashtagsDict[h.lower()].append(i)

	return hashtagsDict

if __name__=='__main__':

	# # for DisasterTweet
	# place = 'california'
	# dtWhich = 'california_fire'
	# aff = 'unaffected'
	# filename = 'DisasterTweet/' + dtWhich + '/' + dtWhich + '_' + aff + '_filtered_hash.txt'

	filename = 'cf10k.csv'

	with open(filename) as f:
		lines = f.readlines()
	preptweets = preprocess(lines, 'cf10k', stem = True, tokenize = True, removeHandles = True, removeLinks = True, removeEmojis = True, removeSymbols = True, replaceSlang = True, verbose=False)
	hashtagsPerTweet = extract_hashtags(preptweets)
	hashtagsInvertedCollection = collect_hashtags(hashtagsPerTweet)
	L = []
	for k in sorted(hashtagsInvertedCollection):
		L.append([k, len(hashtagsInvertedCollection[k])])

	def sorterfunc(elem):
		return -elem[1]

	L = sorted(L, key=sorterfunc)
	for line in L:
		print(line)

	# print(slang)

	# for tweet in preptweets:
	# 	print(tweet)

# tfidf = TfidfVectorizer(tokenizer=tk.tokenize, stop_words="english")
# tfs = tfidf.fit_transform(tweets)
# wordsids =  tfidf.get_feature_names()
# print (tfs).shape
