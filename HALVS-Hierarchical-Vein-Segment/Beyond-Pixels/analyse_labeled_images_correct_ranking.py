import pickle5 as pickle
import os
import numpy as np
PATH = './labeled_images_correct_ranking/0/final_dict.pickle'
DICT = pickle.load(open(PATH, "rb"))
#print(DICT)
print(DICT.keys())
print(DICT[2].keys())
print(DICT[2][0.95, 1].keys())
print(DICT[2][0.95, 1]['total'])
print(DICT[2][0.95, 1][7])

x = sorted(DICT[2][0.95, 1][7].items(), key=lambda x:x[1], reverse=True)
print(x)
with open('test.txt', 'w') as f:
    for line in x:
        f.write(str(line[0])+' : '+str(line[1])+'\n')