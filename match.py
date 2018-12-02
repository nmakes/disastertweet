import load_cr26
from nltk.stem.porter import *
from tqdm import tqdm

stemmer = PorterStemmer()

data = load_cr26.get_dataset()
tweets = data[2][0]
labels = data[1]
hashtags = data[2][2]
hashtagsID = data[2][1]

accuracies =[]
keywords=['thunderstorm', 'debri', 'wildfir', 'wreckag', 'typhoon', 'terror', 'outbreak', 'rescuer', 'arson', 'storm',
'dust storm', 'bomb', 'sandstorm', 'hurrican', 'fire', 'nuclear disast', 'raze', 'terrorist', 'mass murder', 'evacu',
'disast', 'earthquak', 'hostag', 'survivor', 'massacr', 'violent storm', 'polic', 'injur', 'sinkhol', 'crash', 'accid',
'derail', 'drought', 'flood', 'fire truck', 'bioterror', 'attack', 'landslid', 'displac', 'rainstorm', 'armi', 'cyclon',
 'tornado', 'hail', 'collis', 'hailstorm', 'damag', 'dead', 'ambul', 'fatal']

retrieved_indexes = []
relevant_retrieved_indexes = []

newhashtags=[]

def match(keywords,tweets):

	totalrel=0
	totalret=0
	for keyword in tqdm(keywords):
		relevant = 0
		total = 0
		for x,tweet in enumerate(tweets):
			flag = True
			for k in keyword.split():
				if k not in tweet:
					flag=False
			if(flag):
				total = total+1
				totalret+=1
				retrieved_indexes.append(x)
				if(labels[x]==1):
					totalrel+=1
					relevant=relevant+1
					relevant_retrieved_indexes.append(x)
					if(len(hashtags[x])>0):
						for elem in hashtags[x]:
							newhashtags.append(elem)
		# print (keyword,relevant,total)
		accuracies.append([keyword,relevant/(total+1)])

	# print("\n FINAL ACC = ",totalrel,totalret,totalrel/totalret)

	return accuracies

def fun(pair):
	return  -pair[1]

r = match(keywords,tweets)
r = sorted(r,key=fun)
topk=[]
for record in r[:50]:
	# print (record)
	topk.append(record[0])

print('Top 50 Keywords:')
print (topk)

print('\nResults of keyword based matching on CrisisLexT26 dataset')
print ("Retrieved : " , len(retrieved_indexes))
print ("Relevant : ",len(relevant_retrieved_indexes))
print("Precision : ", len(relevant_retrieved_indexes)/len(retrieved_indexes))
print("Recall : " , len(relevant_retrieved_indexes)/25070)

freqhash =[]

newhashtags=set(newhashtags)
for hashtag in newhashtags:
	freqhash.append([len(hashtagsID[hashtag]),hashtag])

freqhash = sorted(freqhash)
top50hashes=freqhash[-50:]

def matchbyhash(hashtags,tweets):
	totalrel=0
	totalret=0
	for tag in tqdm(hashtags):
		relevant = 0
		total = 0
		for x,tweet in enumerate(tweets):
			# print (x,tweet)
			if(tag[1] in tweet):
				total = total+1
				totalret+=1
				retrieved_indexes.append(x)
				if(labels[x]==1):
					totalrel+=1
					relevant=relevant+1
					relevant_retrieved_indexes.append(x)
		# print (tag,relevant,total)
	# rint("\n FINAL ACC while matching hashtags = ",totalrel,totalret,totalrel/(totalret+1))

matchbyhash(top50hashes,tweets)

relevant_retrieved_indexes=list(set(relevant_retrieved_indexes))
retrieved_indexes=list(set(retrieved_indexes))
print('\nResults of keyword & hashtags based matching on CrisisLexT26 dataset')
print ("Retrieved : " , len(retrieved_indexes))
print ("Relevant : ",len(relevant_retrieved_indexes))
print("Precision : ", len(relevant_retrieved_indexes)/len(retrieved_indexes))
print("Recall : " , len(relevant_retrieved_indexes)/25070)
