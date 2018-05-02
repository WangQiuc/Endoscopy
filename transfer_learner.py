# coding=utf-8
__author__ = 'wangqc'

import numpy as np
import matplotlib.pyplot as plt

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

	# transfer learn from pre-trained cnn to extract the features; mode as 'train_output', 'valid_output' or 'test_output'
	def _get_transfer(self, X, mode, classifier):
		path = '%s_%s' % (classifier, mode)
		with h5py.File(self.model_output_path) as model_output:
			if path in model_output:
				transfer_output = model_output[path][:]
			else:
				# load pre-trained cnn without top, output 2 classes as 0: negative; 1: positive
				clf, weights = self.transfer_classifiers[classifier]
				base = clf(False, weights, Input(shape=(299, 299, 3)), classes=self.num_classes)
				f = K.function([base.layers[0].input, K.learning_phase()], [base.layers[-1].output])
				# split whole data set to 100 per batch due to memory issue
				transfer_output = [f([X[i:i + 100], 0])[0] for i in range(0, len(X), 100)]
				transfer_output = np.concatenate(transfer_output, axis=0)
				model_output.create_dataset(path, data=transfer_output)
			return transfer_output

	def train(self, transfer_classifier):
		# data preprocess
		X_train, X_valid, y_train, y_valid = self.du.data_preprocess('train')

		# train
		# first get pre-trained cnn's result as extracted features
		transfer_train_output = self._get_transfer(X_train, 'train', transfer_classifier)
		# then input extracted features to 2 FC layers (2048 -(RELU)-> 1024 -(SOFTMAX)-> 2) to get result
		input_tensor = Input(shape=(1, 1, 2048))
		X = Flatten()(input_tensor)
		X = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.00005))(X)
		predictions = Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.000001))(X)
		model = Model(inputs=input_tensor, outputs=predictions)
		model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
		logger_tc.info('start training')
		model.fit(transfer_train_output, y_train, epochs=30, batch_size=128)

		# valid
		transfer_valid_output = self._get_transfer(X_valid, 'valid', transfer_classifier)
		logger_tc.info('start testing')
		pred = model.predict(transfer_valid_output, batch_size=32)
		y_pred = np.zeros(len(pred), dtype=int)
		y_pred[pred[:, 1] > pred[:, 0]] = 1
		logger_tc.info('validation accuracy: %s' % metrics.accuracy_score(y_valid[:, 1], y_pred))
		with h5py.File(self.model_output_path) as model_output:
			if '%s_valid_pred' % transfer_classifier not in model_output:
				model_output.create_dataset('%s_valid_pred' % transfer_classifier, data=pred)

	def test(self):
		pass


if __name__ == '__main__':
	tl = TransferLearner()
	tl.train('resnet50')
