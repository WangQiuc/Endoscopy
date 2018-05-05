# coding=utf-8
__author__ = 'wangqc'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.applications import ResNet50, Xception
from keras.layers import Input
from keras import backend as K
from sklearn import metrics as metrics

import os
import h5py
import logging

from transfer_learner import TransferLearner

logging.basicConfig(level=logging.INFO, format='%(name)-12s %(asctime)s %(levelname)-8s %(message)s')
logger_tc = logging.getLogger('tc')


# tools for classification analysis
class Analyser:
	def __init__(self):
		self.resnet50_weights = os.path.realpath('models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.xception_weights = os.path.realpath('models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.data_path = os.path.realpath('data/data.h5')
		self.model_output_path = os.path.realpath('data/model_output.h5')
		self.model_path = os.path.realpath('data/model.h5')
		self.num_classes = 2
		self.transfer_classifiers = {'resnet50': (ResNet50, self.resnet50_weights),
									 'xception': (Xception, self.xception_weights)}

	# output roc curve
	def roc_curve(self, y_set, yp_set):
		with h5py.File(self.data_path) as data, h5py.File(self.model_output_path) as model_output:
			y, y_prob = data[y_set][:][:, 1], model_output[yp_set][:][:, 1]
		fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
		auc_score = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %.2f)' % auc_score)
		plt.xlabel('False Positive Rate'), plt.ylabel('True Poistive Rate'), plt.legend(loc='lower right')
		plt.show()

	# visualize first layers of the cnn
	def bottom_layer_visualizer(self, classifier):
		with h5py.File(self.data_path) as data:
			train_mean, train_std = data['train_mean'][:], data['train_std'][:]
			idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 1)
			sample = data['X_train'][:][idx]
			x = ((sample - train_mean) / train_std)[np.newaxis]
		clf, weights = self.transfer_classifiers[classifier]
		base = clf(False, weights, Input(shape=(299, 299, 3)), classes=self.num_classes)
		f = K.function([base.layers[0].input, K.learning_phase()], [base.layers[2].output])
		bottom_layer = f([x, 0])[0]
		print(bottom_layer)

	# shift an occluded area over a positive image and collect positive probability on different occluded position
	# generate a hot map based on the collections to see which position positive probability is more sensitive to
	def hotmap(self, classifier):
		tl = TransferLearner()
		with h5py.File(self.data_path) as data:
			idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 3, replace=False)
			logger_tc.info('sample images %s' % ', '.join(map(str, data['y_valid'][:][idx, 0])))
			samples = data['X_train'][:][idx]
		pic_len, occ_len = samples.shape[1], 60
		hot_maps = []
		for i in range(3):
			occ_samples = []
			sample = samples[i]
			for row in range(0, pic_len - occ_len, occ_len // 6):
				for col in range(0, pic_len - occ_len, occ_len // 6):
					# add an 60 * 60 occluded area on the sample picture and move 10 pixels per step
					occ_sample = np.copy(sample)
					occ_sample[row: row + occ_len, col: col + occ_len] = 0
					occ_samples.append(occ_sample[np.newaxis])
			occ_samples = np.concatenate(occ_samples, axis=0)
			hot_maps.append(tl.predict(occ_samples, 'resnet50')[:, 1].reshape((pic_len // (occ_len // 6) - 5, -1)))
		plt.figure(figsize=(100, 50))
		for i in range(3):
			plt.subplot(2, 3, i + 1)
			plt.imshow(np.uint8(samples[i]))
			plt.subplot(2, 3, i + 4)
			sns.heatmap(hot_maps[i])
		plt.show()


if __name__ == '__main__':
	al = Analyser()
	# al.roc_curve('y_valid_a', 'resnet50_valid_pred')
	al.hotmap('resnet50')
