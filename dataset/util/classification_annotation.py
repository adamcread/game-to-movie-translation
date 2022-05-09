# loop through all files in val
import os

classes = ['full-body-sitting', 'full-body-standing', 'half-body', 'head-only', 'other']
root = '../patches/classified/val/'

matches = []
for i, c in enumerate(classes):
    for image in os.listdir(root+c):
        matches.append(f"{c}/{image} {i}")
    
with open('../annotation/val.txt', 'w+') as fp:
    fp.write('\n'.join(matches))