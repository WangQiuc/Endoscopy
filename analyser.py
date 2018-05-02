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


# tools for classification analysis
class Analyser:
	def __init__(self):
		self.data_path = os.path.realpath('data/data.h5')
		self.model_output_path = os.path.realpath('data/model_output.h5')
		self.num_classes = 2

	# output roc curve
	def roc_curve(self, y_set, yp_set):
		with h5py.File(self.data_path) as data, h5py.File(self.model_output_path) as model_output:
			y, y_prob = data[y_set][:][:, 1], model_output[yp_set][:][:, 1]
		fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
		auc_score = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=2, label='ROC Curve (area = %.2f)' % auc_score)
		plt.xlabel('False Positive Rate'), plt.ylabel('True Poistive Rate'), plt.legend(loc='lower right')
		plt.show()

	# visualize weight at the bottom (layer next to the input) of the cnn
	def weight_visualizer(self):
		pass


if __name__ == '__main__':
	al = Analyser()
	al.roc_curve('y_valid_a', 'resnet50_valid_pred')
