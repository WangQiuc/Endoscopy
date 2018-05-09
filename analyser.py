# coding=utf-8
__author__ = 'wangqc'

import os
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications import ResNet50, Xception
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from sklearn import metrics as metrics

from transfer_learner import TransferLearner

logging.basicConfig(level=logging.INFO, format='%(name)-12s %(asctime)s %(levelname)-8s %(message)s')
logger_tc = logging.getLogger('tc')


# tools for classification analysis
class Analyser:
	def __init__(self):
		self.img_path = os.path.realpath('data/pic_train')
		self.resnet50_weights = os.path.realpath('models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.xception_weights = os.path.realpath('models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.data_path = os.path.realpath('data/data.h5')
		self.model_output_path = os.path.realpath('data/model_output.h5')
		self.model_path = os.path.realpath('data/model.h5')
		self.num_classes = 2
		self.transfer_classifiers = {'resnet50': (ResNet50, self.resnet50_weights),
		                             'xception': (Xception, self.xception_weights)}
		self.layer_stages = {'resnet50': [3, 36, 78, 140, 172],
		                     'xception': []}

	# output roc curve
	def roc_curve(self, y_set, yp_set):
		with h5py.File(self.data_path) as data, h5py.File(self.model_output_path) as model_output:
			y, y_prob = data[y_set][:][:, 1], model_output[yp_set][:][:, 1]
		fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
		auc_score = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %.2f)' % auc_score)
		plt.xlabel('False Positive Rate'), plt.ylabel('True Poistive Rate'), plt.legend(loc='lower right')
		plt.show()

	# visualize specific layer of the cnn, ResNet50 have 5 stages {1,2,3,4,5}
	def layer_visualizer(self, classifier, stage=0, source=None):
		# pick one specific image or randomly pick one from validation set
		with h5py.File(self.data_path) as data:
			train_mean, train_std = data['train_mean'][:], data['train_std'][:]
			if source:
				sample = img_to_array(load_img(os.path.join(self.img_path, source), target_size=(299, 299)))[np.newaxis]
				sample_id = int(source.replace('.jpg', ''))
			else:
				idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 1)
				sample, sample_id = data['X_valid'][:][idx], data['y_valid'][:][idx, 0][0]
		# show original img
		img = plt.imshow(np.uint8(sample[0]))
		img.axes.grid(False)
		plt.show()
		x = ((sample - train_mean) / train_std)
		layer = self.layer_stages[classifier][stage - 1]
		clf, weights = self.transfer_classifiers[classifier]
		model = clf(False, weights, Input(shape=(299, 299, 3)), classes=self.num_classes)
		f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
		bottom_layer = f([x, 0])[0][0]
		# use number of dimensions to determine the layout of subplot
		dimensions, divider = bottom_layer.shape[-1], 2 ** (stage // 2 + 3)
		plt.figure(figsize=(100, 100), facecolor=(0.2, 0.2, 0.2))
		for i in range(bottom_layer.shape[-1]):
			plt.subplot(divider, dimensions // divider, i + 1)
			sns.heatmap(bottom_layer[:, :, i], cmap='binary', xticklabels=False, yticklabels=False, cbar=False)
		plt.suptitle('sample %s' % sample_id, color='white')
		plt.show()

	# shift an occluded area over a positive image and collect the positive probabilities with different area occluded
	# plot the collection on a heatmap to see which positions are more important to a positive classification
	def heatmap(self, classifier, source=None):
		tl = TransferLearner()
		# pick 3 specific images or randomly pick 3 images from validation set
		if source:
			samples = [img_to_array(load_img(os.path.join(self.img_path, img), target_size=(299, 299))) for img in source]
			sample_ids = [int(img.replace('.jpg', '')) for img in source]
		else:
			with h5py.File(self.data_path) as data:
				idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 3, replace=False)
				samples, sample_ids = data['X_valid'][:][idx], data['y_valid'][:][idx, 0]
		pic_len, occ_len = len(samples[0]), 60
		heat_maps = []
		model = tl.train(classifier, 'model')
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
			heat_maps.append(
				tl.predict(occ_samples, 'resnet50', model)[:, 1].reshape((pic_len // (occ_len // 6) - 5, -1)))
		plt.figure(figsize=(100, 50))
		for i in range(3):
			plt.subplot(2, 3, i + 1)
			plt.title('sample %s' % sample_ids[i])
			img = plt.imshow(np.uint8(samples[i]))
			img.axes.grid(False)
			plt.subplot(2, 3, i + 4)
			sns.heatmap(heat_maps[i], cmap='magma', xticklabels=False, yticklabels=False)
		plt.show()


if __name__ == '__main__':
	al = Analyser()
	# al.roc_curve('y_valid_a', 'resnet50_valid_pred')
	# al.layer_visualizer('resnet50', stage=1, source='0149.jpg')
	al.heatmap('resnet50', source=['0149.jpg', '0219.jpg', '0312.jpg'])
