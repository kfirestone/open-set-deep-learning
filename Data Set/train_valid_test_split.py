import os
import shutil
import random
import math

#Create unknown class split (20% to validation, 80% test)

for i, cls in enumerate(os.listdir('Unknown Classes/')):
    random.seed(i)
    num_examples = len(os.listdir('Unknown Classes/' + cls))
    valid_samples = random.sample(range(num_examples), math.floor(0.2*num_examples))

    for j, img in enumerate(os.listdir('Unknown Classes/' + cls)):
        if j in valid_samples:
            os.makedirs('split_data_set/unknown_classes/valid/' + cls, exist_ok = True)
            shutil.copyfile('Unknown Classes/' + cls + '/' + img, 'split_data_set/unknown_classes/valid/' + cls + '/' + img)
        else:
            os.makedirs('split_data_set/unknown_classes/test/' + cls, exist_ok = True)
            shutil.copyfile('Unknown Classes/' + cls + '/' + img, 'split_data_set/unknown_classes/test/' + cls + '/' + img)

#Create known class split (70% train, 15% validation, 15% test)

for i, cls in enumerate(os.listdir('Known Classes/')):
    random.seed(i)
    num_examples = len(os.listdir('Known Classes/' + cls))
    valid_test_idxs = random.sample(range(num_examples), math.floor(0.3*num_examples))
    valid_idxs = valid_test_idxs[0:len(valid_test_idxs)//2]
    test_idxs = valid_test_idxs[len(valid_test_idxs)//2:]

    for j, img in enumerate(os.listdir('Known Classes/' + cls)):
        if j in valid_idxs:
            os.makedirs('split_data_set/known_classes/valid/' + '/' + cls, exist_ok = True)
            shutil.copyfile('Known Classes/' + cls + '/' + img, 'split_data_set/known_classes/valid/' + cls + '/' + img)
        elif j in test_idxs:
            os.makedirs('split_data_set/known_classes/test/' + '/' + cls, exist_ok = True)
            shutil.copyfile('Known Classes/' + cls + '/' + img, 'split_data_set/known_classes/test/' + cls + '/' +  img)
        else:
            os.makedirs('split_data_set/known_classes/train/' + '/' + cls, exist_ok = True)
            shutil.copyfile('Known Classes/' + cls + '/' + img, 'split_data_set/known_classes/train/' + cls + '/' + img)