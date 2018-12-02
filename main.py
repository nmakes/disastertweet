import subprocess

print('Select an option:')
print('1. View CrowdFlower10K statistics')
print('2. View CrisisLex26 statistics')
print('3. View CrowdFlower10K hashtags data')
print('4. Run Learning Based Disaster Tweet Retrieval')
print('5. Run Matching Based Disaster Tweet Retrieval')
print('6. Exit')
opt = input('>> ')

if opt=='1':
    subprocess.call(['python3', 'load_cr26.py'])
if opt=='2':
    subprocess.call(['python3', 'load_cf10k.py'])
if opt=='3':
    print('[hashtag, number of tweets containing hashtag]')
    subprocess.call(['python3', 'preprocess.py'])
if opt=='4':
    subprocess.call(['python3', 'learn.py'])
if opt=='5':
    subprocess.call(['python3', 'match.py'])
