import random
import shutil
import json
from scipy.special import softmax
import numpy as np

file_names = open('../annotation/test.txt', 'r').readlines()
results = json.load(open('../annotation/classification_results.json', 'r'))
root = '../patches/extracted'
dest = '../patches/classified/test'

res_dict = {'full-body-standing': [], 'full-body-sitting': [], 'half-body': [], 'head-only': [], 'other': []}
for i in range(len(file_names)):
    print(f"{i} out of {len(file_names)}")
    f = file_names[i].split()[0]
    res = results['pred_class'][i]
    score = max(results['class_scores'][i])

    if 'trainB' in f:
        res_dict[res] = res_dict[res] + [(f.split('/')[-1], score)]

    # shutil.copyfile(f"{root}/{f}", f"{dest}/{res}/{'_'.join(f.split('/'))}")

# ! TRAIN A: 2517, 626, 5665, 1265, 275
# ! TRAIN B: 4676, 1591, 14821, 6394, 1624

# * 1.85776, 2.5415, 2.616, 5.0545, 5.9505
# 2,000/4,000
# 600/1,200
# 5,000/10,000
# 1,000/2,000
# 250/500

# how to select images from these patch proportions?
props = {'full-body-standing': 2000, 'full-body-sitting': 600, 'half-body': 5000, 'head-only': 1000, 'other': 250}
sampled_files = []
for c in res_dict.keys():
    probabilities = softmax([x[1] for x in res_dict[c]])
    fz = np.random.choice(['_'.join(x[0].split('_')[:-1])+'.png' for x in res_dict[c]], size=props[c]*2, p=probabilities)
    sampled_files.extend(fz)


print(len(sampled_files))
with open('../annotation/trainB.txt', 'w+') as fp:
    fp.write('\n'.join(sampled_files))