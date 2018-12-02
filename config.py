import re

# Define stop words
# stop_words = ["@","*","\"","\'",".","???","️","?", "'", '"']
symbols = "!@#$%^&*()-_=+{}[]:;\'\",<.>?/\\|~`\t\n"

# Load slang
slang = {}
with open('slang.txt') as f:
    slangs = f.readlines()
for line in slangs:
    line = line.split()
    slang[ line[0] ] = line[1:]

# DisasterTweetFiles
DisasterTweetFiles = ['california_fire', 'iowa_stf', 'iowa_stf_2', 'iowa_storm', 'jersey_storm', 'michigan_storm', 'napa_earthquake', 'newyork_storm', 'oklahoma_storm', 'texas_storm', 'vermont_storm', 'virginia_storm', 'washington_mudslide', 'washington_storm', 'washington_wildfire']
tempList = []
for f in DisasterTweetFiles:
    tempList.append( 'DisasterTweet' + '/' + f + '/' + f + '_affected_filtered_hash.txt' )
    tempList.append( 'DisasterTweet' + '/' + f + '/' + f + '_unaffected_filtered_hash.txt' )
    tempList.append( 'DisasterTweet' + '/' + f + '/' + 'hashtags' )
# print(tempList)
