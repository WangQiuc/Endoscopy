# coding=utf-8
__author__ = 'wangqc'

import os
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications import ResNet50, Xception
from keras.models import load_model
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
		self.img_train_path = os.path.realpath('data/pic_train')
		self.img_test_path = os.path.realpath('data/pic_test')
		self.num_classes = 2
		self.resnet50_weights = os.path.realpath('models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.xception_weights = os.path.realpath('models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		self.data_path = os.path.realpath('data/data.h5')
		self.model_output_path = os.path.realpath('data/model_output.h5')
		self.model_path = {'resnet50': os.path.realpath('data/model_resnet50.h5'),
						   'xception': os.path.realpath('data/model_xception.h5')}
		self.transfer_classifiers = {'resnet50': (ResNet50, self.resnet50_weights),
		                             'xception': (Xception, self.xception_weights)}
		self.layer_stages = {'resnet50': [3, 36, 78, 140, 172, -1],
		                     'xception': []}

	# output roc curve
	def roc_curve(self, y_set, yp_set):
		with h5py.File(self.data_path) as data, h5py.File(self.model_output_path) as model_output:
			y, y_prob = data[y_set][:][:, 1], model_output[yp_set][:][:, 1]
		fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
		auc_score = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %.2f)' % auc_score)
		plt.xlabel('False Positive Rate'), plt.ylabel('True Poistive Rate'), plt.legend(loc='lower right')
		plt.savefig('output/roc_curve.jpg')
		plt.show()

	# visualize specific layer of the cnn, ResNet50 have 5 stages {1,2,3,4,5}
	def layer_visualizer(self, classifier, stage, mode='test', source=None):
		# pick one specific image or randomly pick one from validation set
		img_path = self.img_test_path if mode == 'test' else self.img_train_path
		if source:
			sample = img_to_array(load_img(os.path.join(img_path, source), target_size=(299, 299)))[np.newaxis]
		else:
			with h5py.File(self.data_path) as data:
				idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 1)
				sample, sample_id = data['X_valid'][:][idx], data['y_valid'][:][idx, 0][0]
		# # show original img
		# img = plt.imshow(np.uint8(sample[0]))
		# img.axes.grid(False)
		# plt.show()
		# # preprocess sample data
		with h5py.File(self.data_path) as data:
			x = (sample - data['train_mean'][:]) / data['train_std'][:]
		# build model architecture ending with different layers
		layer = self.layer_stages[classifier][stage - 1]
		clf, weights = self.transfer_classifiers[classifier]
		model = clf(False, weights, Input(shape=(299, 299, 3)), classes=self.num_classes)
		f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
		bottom_layer = f([x, 0])[0][0]

		# if stage is 0 which to be final layer, return vector instead of heatmap
		if not stage:
			return bottom_layer

		# use number of dimensions to determine the layout of subplot
		dimensions, divider = bottom_layer.shape[-1], 2 ** (stage // 2 + 3)
		plt.figure(figsize=(50, 50 * divider * divider // dimensions), facecolor=(0.2, 0.2, 0.2))
		for i in range(bottom_layer.shape[-1]):
			plt.subplot(divider, dimensions // divider, i + 1)
			sns.heatmap(bottom_layer[:, :, i], cmap='binary', xticklabels=False, yticklabels=False, cbar=False)
		plt.savefig('output/layer_sample_stage%s.jpg' % stage, facecolor=(0.2, 0.2, 0.2))
		# plt.show()

	# shift an occluded area over a positive image and collect the positive probabilities with different area occluded
	# plot the collection on a heatmap to see which positions are more important to a positive classification
	def heatmap(self, classifier, mode='test', load=True, source=None):
		tl = TransferLearner()
		img_path = self.img_test_path if mode == 'test' else self.img_train_path
		# pick 3 specific images or randomly pick 3 images from validation set
		if source:
			samples = [img_to_array(load_img(os.path.join(img_path, img), target_size=(299, 299))) for img in source]
			sample_ids = [int(img.replace('.jpg', '')) for img in source]
		else:
			with h5py.File(self.data_path) as data:
				idx = np.random.choice(np.where(data['y_valid'][:][:, 1] == 1)[0], 3, replace=False)
				samples, sample_ids = data['X_valid'][:][idx], data['y_valid'][:][idx, 0]
		pic_len = len(samples[0])
		heat_maps = []
		# load a trained model or train a new model
		model = load_model(self.model_path[classifier]) if load else tl.train(classifier, 'model')
		for i in range(3):
			occ_samples = []
			sample = samples[i]
			for row in range(0, pic_len - 59, 6):
				for col in range(0, pic_len - 59, 6):
					# add an 60 * 60 occluded area on the sample picture and move 6 pixels per step
					occ_sample = np.copy(sample)
					occ_sample[row: row + 60, col: col + 60] = 0
					occ_samples.append(occ_sample[np.newaxis])
			occ_samples = np.concatenate(occ_samples, axis=0)
			# preprocess sample data
			with h5py.File(self.data_path) as data:
				occ_samples = (occ_samples - data['train_mean'][:]) / data['train_std'][:]
			# gather the positive probability of each occluded sample img
			heat_maps.append(
				tl.predict(occ_samples, 'resnet50', model)[:, 1].reshape(((pic_len - 60) // 6 + 1, -1)))

		plt.figure(figsize=(20, 10))
		for i in range(3):
			plt.subplot(2, 3, i + 1)
			plt.title('sample %s' % sample_ids[i], fontsize=20)
			img = plt.imshow(np.uint8(samples[i]))
			img.axes.grid(False)
			plt.subplot(2, 3, i + 4)
			sns.heatmap(heat_maps[i], cmap='magma', xticklabels=False, yticklabels=False)
		plt.savefig('output/heatmap_sample.jpg')
		plt.show()



if __name__ == '__main__':
	al = Analyser()
	# al.roc_curve('y_valid_a', 'resnet50_valid_pred')
	# print(al.layer_visualizer('resnet50', mode='train', stage=0, source='0996.jpg'))
	for i in range(5):
		al.layer_visualizer('resnet50', stage=i+1, mode='train', source='0903.jpg')
	# al.heatmap('resnet50', load=False, source=['0968.jpg', '0687.jpg', '0995.jpg'])

