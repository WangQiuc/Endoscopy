# coding=utf-8
__author__ = 'wangqc'

import numpy as np
from itertools import permutations

from keras.applications import ResNet50, Xception
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras import backend as K, regularizers
from sklearn import metrics as metrics

import os
import h5py
import logging

from data_utils import DataUtils

logging.basicConfig(level=logging.INFO, format='%(name)-12s %(asctime)s %(levelname)-8s %(message)s')
logger_tc = logging.getLogger('tc')


# utilize transfer learning to train the tumor classifier by adding FC layers on the top of a pre-trained classifier
class TransferLearner:
	def __init__(self):
		self.resnet50_weights = os.path.realpath('models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.xception_weights = os.path.realpath('models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.model_path = os.path.realpath('data/model.h5')
		self.model_output_path = os.path.realpath('data/model_output.h5')
		self.num_classes = 2
		self.transfer_classifiers = {'resnet50': (ResNet50, self.resnet50_weights),
		                             'xception': (Xception, self.xception_weights)}
		self.du = DataUtils()

	# transfer learn from pre-trained cnn to extract the features; mode as 'train', 'valid', 'cross_val' or 'test'
	def _get_transfer(self, X, mode, classifier):
		path = '%s_%s' % (classifier, mode)
		with h5py.File(self.model_output_path) as model_output:
			if path in model_output:
				return model_output[path][:]

		# load pre-trained cnn without top, output 2 classes as 0: negative; 1: positive
		clf, weights = self.transfer_classifiers[classifier]
		base = clf(False, weights, Input(shape=(299, 299, 3)), classes=self.num_classes)
		f = K.function([base.layers[0].input, K.learning_phase()], [base.layers[-1].output])
		# split whole data set to 100 per batch due to memory issue
		transfer_output = [f([X[i:i + 100], 0])[0] for i in range(0, len(X), 100)]
		transfer_output = np.concatenate(transfer_output, axis=0)
		if mode != 'cross_val':
			with h5py.File(self.model_output_path) as model_output:
				model_output.create_dataset(path, data=transfer_output)
		return transfer_output


	# if cross_val is on, inputs should be X_train, y_train, X_valid, y_valid which have been preprocessed
	def train(self, classifier, cross_val=False, *inputs):
		if not cross_val:
			# data preprocess
			X_train, y_train, X_valid, y_valid = self.du.data_preprocess('train')
			# get pre-trained cnn's result as extracted features
			logger_tc.info('start transfer learning')
			transfer_train_output = self._get_transfer(X_train, 'cross_val', classifier)
			transfer_valid_output = self._get_transfer(X_valid, 'cross_val', classifier)
		else:
			X_train, y_train, X_valid, y_valid = inputs
			logger_tc.info('start transfer learning')
			transfer_train_output = self._get_transfer(X_train, 'train', classifier)
			transfer_valid_output = self._get_transfer(X_valid, 'valid', classifier)

		# parameter tuning
		b_score, b_r1, b_r2, b_pred = 0, 0, 0, np.array([])
		logger_tc.info('parameter tuning')
		# for r1, r2 in [(5e-4, 1e-6)]:
		# for r1, r2 in [(np.random.choice(np.arange(50, 100), 2) / 1000.) for _ in range(3)]:
		for r1, r2 in permutations([1e-1, 1e-2, 1e-3], 2):
			# input extracted features to 2 FC layers (2048 -(RELU)-> 1024 -(SOFTMAX)-> 2) to get result
			input_tensor = Input(shape=(1, 1, 2048))
			X = Flatten()(input_tensor)
			X = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(r1))(X)
			predictions = Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(r2))(X)
			model = Model(inputs=input_tensor, outputs=predictions)
			model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
			model.fit(transfer_train_output, y_train, epochs=20, batch_size=128)
			# validate
			pred = model.predict(transfer_valid_output, batch_size=32)
			y_pred = np.zeros(len(pred), dtype=int)
			y_pred[pred[:, 1] > pred[:, 0]] = 1
			score = metrics.accuracy_score(y_valid[:, 1], y_pred)
			if score > b_score:
				b_score, b_r1, b_r2, b_pred = score, r1, r2, pred
			logger_tc.info('cur: acc %s, r1 %s, r2 %s; best: acc %s, r1 %s, r2 %s' % (score, r1, r2, b_score, b_r1, b_r2))
		# if cross_val:
		# 	return b_score
		# with h5py.File(self.model_output_path) as model_output:
		# 	if '%s_valid_pred' % classifier not in model_output:
		# 		model_output.create_dataset('%s_valid_pred' % classifier, data=b_pred)

	def cross_validation(self, classifier, fold=5):
		du = DataUtils()
		scores = []
		for i in range(fold):
			logger_tc.info('cross validation fold %s start' % i)
			X_train, y_train, X_valid, y_valid = du.data_extract('cross_val')
			X_train, X_valid, y_train, y_valid=\
				du.augmentation(X_train), du.augmentation(X_valid), np.tile(y_train, (8, 1)), np.tile(y_valid, (8, 1))
			train_mean, train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
			X_train, X_valid = (X_train - train_mean) / train_std, (X_valid - train_mean) / train_std
			scores += self.train(classifier, True, X_train, y_train, X_valid, y_valid),
			logger_tc.info('cross validation fold %s end\n\n' % i)
		logger_tc.info('%s %s fold cross val: avg acc: %s, std: %s') % (classifier, fold, np.mean(scores), np.std(scores))


	def test(self):
		pass


if __name__ == '__main__':
	tl = TransferLearner()
	tl.train('resnet50')
	# tl.cross_validation('resnet50')
