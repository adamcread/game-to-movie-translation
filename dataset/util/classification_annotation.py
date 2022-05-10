# loop through all files in val
import os

classes = ['trainA', 'trainB']
root = '../patches/extracted/'

matches = []
for i, c in enumerate(classes):
    for image in os.listdir(root+c):
        matches.append(f"{c}/{image} {i}")
    
with open('../annotation/test.txt', 'w+') as fp:
    fp.write('\n'.join(matches))