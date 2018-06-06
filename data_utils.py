# coding=utf-8
__author__ = 'wangqc'

import os
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from sklearn.model_selection import train_test_split as data_split

logging.basicConfig(level=logging.INFO, format='%(name)-12s %(asctime)s %(levelname)-8s %(message)s')
logger_tc = logging.getLogger('tc')

# tools for data process
class DataUtils:
	def __init__(self):
		self.img_train_path = os.path.realpath('data/pic_train')
		self.img_test_path = os.path.realpath('data/pic_test')
		self.label_train_path = os.path.realpath('data/labels_train.csv')
		self.label_test_path = os.path.realpath('data/labels_test.csv')
		self.data_path = os.path.realpath('data/data.h5')
		self.num_classes = 2

	# extract features from images and labels for csv, image resolution as 299 * 299, mode as 'train', 'cross_val', 'test'
	def data_extract(self, mode):
		with h5py.File(self.data_path) as data:
			if 'X_%s' % mode in data:
				logger_tc.info('%s data exists.' % mode)
				return
		img_path, label_path = (self.img_test_path, self.label_test_path) if mode == 'test' \
			else (self.img_train_path, self.label_train_path)
		features = np.concatenate(
			[img_to_array(load_img(os.path.join(img_path, img), target_size=(299, 299)))[np.newaxis]
			 for img in sorted(os.listdir(img_path))], axis=0)
		labels = np.genfromtxt(label_path, dtype=int, delimiter=',', filling_values=1)
		logger_tc.info('feature shape: %s\tlabel shape: %s' % (features.shape, labels.shape))

		# if mode is 'test', there is no need to split data
		if mode == 'test':
			with h5py.File(self.data_path) as data:
				data.create_dataset('X_test', data=features), data.create_dataset('y_test', data=labels)

		# if mode is 'train' or 'cross_val', need to split data into train and validation data set
		else:
			# split positive and negative samples and split train and validation respectively to keep the proportion
			pos_idx, neg_idx = np.where(labels[:, 1] == 1)[0], np.where(labels[:, 1] == 0)[0]
			logger_tc.info('postive sample: %s\tnegative sample: %s' % (len(pos_idx), len(neg_idx)))

			# split train and validation by 9 : 1 and merge positive and negative samples in train and val data respectively
			X_pos_train, X_pos_valid, y_pos_train, y_pos_valid = data_split(features[pos_idx], labels[pos_idx], test_size=0.2)
			X_neg_train, X_neg_valid, y_neg_train, y_neg_valid = data_split(features[neg_idx], labels[neg_idx], test_size=0.2)
			X_train, X_valid = np.concatenate((X_pos_train, X_neg_train)), np.concatenate((X_pos_valid, X_neg_valid))
			y_train, y_valid = np.concatenate((y_pos_train, y_neg_train)), np.concatenate((y_pos_valid, y_neg_valid))
			logger_tc.info('train features shape: %s\tvalid features shape: %s' % (X_train.shape, X_valid.shape))
			logger_tc.info('train labels shape: %s\tvalid labels shape: %s' % (y_train.shape, y_valid.shape))
			# if mode is 'train', data need to be saved in data.h5
			if mode == 'train':
				with h5py.File(self.data_path) as data:
					data.create_dataset('X_train', data=X_train), data.create_dataset('y_train', data=y_train)
					data.create_dataset('X_valid', data=X_valid), data.create_dataset('y_valid', data=y_valid)
			# if mode is 'cross_val', function need to return data
			else:
				return X_train, y_train, X_valid, y_valid

	# augment data 8 times via rotate each image by 0°, 90°, 180°, 270° and take symmetry
	def augmentation(self, data):
		aug_data = data
		sym = 0
		while sym < 2:
			for i in range(1, 4):
				aug_data = np.concatenate((aug_data, np.rot90(data, i, axes=(1, 2))), axis=0)
			if sym < 1:
				data = np.transpose(data, (0, 2, 1, 3))
				aug_data = np.concatenate((aug_data, data), axis=0)
			sym += 1
		return aug_data

	def data_augment(self):
		with h5py.File(self.data_path) as data:
			if 'X_train_a' in data:
				logger_tc.info('Augmented data exists.')
				return

		with h5py.File(self.data_path) as data:
			X_train_a, X_valid_a = self.augmentation(data['X_train'][:]), self.augmentation(data['X_valid'][:])
			y_train_a, y_valid_a = np.tile(data['y_train'][:], (8, 1)), np.tile(data['y_valid'][:], (8, 1))
			logger_tc.info(
				'train aug features shape: %s\tvalid aug features shape: %s' % (X_train_a.shape, X_valid_a.shape))
			logger_tc.info(
				'train aug labels shape: %s\tvalid aug labels shape: %s' % (y_train_a.shape, y_valid_a.shape))

			# save augmented data to data_aug.h5
			data.create_dataset('X_train_a', data=X_train_a), data.create_dataset('y_train_a', data=y_train_a)
			data.create_dataset('X_valid_a', data=X_valid_a), data.create_dataset('y_valid_a', data=y_valid_a)

	# randomly check sample and or check one sample its augmented copies; t to be 'train', 'valid', 'test'
	def image_sampling(self, mode='train', check_aug=False):
		with h5py.File(self.data_path) as data:
			X_mode, y_mode = 'X_%s' % mode, 'y_%s' % mode
			# if sampling is to check augmented data, randomly pick one index and find its augmented copies
			if check_aug:
				X_mode, y_mode = X_mode + '_a', y_mode + '_a'
				size = len(data[y_mode][:]) // 8
				idx = np.random.randint(0, size - 1)
				sample = [idx + size * i for i in range(8)]
			# if sampling is not to check augmented data, randomly pick 8 indexes
			else:
				sample = sorted(np.random.choice(len(data[y_mode][:]), 8, replace=False))
			sample_label = data[y_mode][sample]
			logger_tc.info('sample: [%s]' % ','.join(map(str, sample_label[:, 0])))
			logger_tc.info('labels: [%s]' % ','.join(map(str, sample_label[:, 1])))
			plt.figure(figsize=(100, 50))
			for i, img in enumerate(data[X_mode][sample]):
				plt.subplot(2, 4, i + 1)
				plt.title('sample %s' % sample_label[i, 0], fontsize=20)
				plt.imshow(np.uint8(img))
			plt.show()

	# mode = 'train' or 'test'
	def data_preprocess(self, mode):
		# data preprocess
		# zero-center(minus mean) and normalization(divide std);
		# The mean must be computed only over the training data and then subtracted equally from all splits (train/val/test)
		if mode == 'train':
			with h5py.File(self.data_path) as data:
				X_train, X_valid = data['X_train_a'][:], data['X_valid_a'][:]
				if 'train_mean' in data:
					train_mean, train_std = data['train_mean'][:], data['train_std'][:]
				else:
					train_mean, train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
					data.create_dataset('train_mean', data=train_mean), data.create_dataset('train_std', data=train_std)
				y_train, y_valid = data['y_train_a'][:], data['y_valid_a'][:]
			# y_train need to be one-hot coded to be fit during the training
			return (X_train - train_mean) / train_std, np_utils.to_categorical(y_train[:, 1], self.num_classes),\
				   (X_valid - train_mean) / train_std, y_valid
		else:
			with h5py.File(self.data_path) as data:
				return (data['X_test'][:] - data['train_mean'][:]) / data['train_std'][:], data['y_test'][:]


if __name__ == '__main__':
	du = DataUtils()
	# du.data_extract('train')
	# du.data_augment()
	# du.image_sampling(mode='valid', check_aug=True)
	du.image_sampling(mode='valid')
	# du.data_preprocess('train')
