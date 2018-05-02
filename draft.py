import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
import h5py


for i in h5py.File('data/model_output.h5').items():
	print(i)

# img_path = 'data/pic_train'
# label_path = 'data/labels_train.csv'
# with h5py.File('data/data.h5') as data:
# 	del data['train_mean']
# 	del data['train_std']
