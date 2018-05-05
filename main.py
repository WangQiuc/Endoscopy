# coding=utf-8
__author__ = 'wangqc'

from data_utils import DataUtils
from transfer_learner import TransferLearner
from analyser import Analyser

import logging

logging.basicConfig(level=logging.INFO, format='%(name)-12s %(asctime)s %(levelname)-8s %(message)s')
logger_tc = logging.getLogger('tc')


class TumorClassifier:
	def __init__(self):
		self.data_utils = DataUtils()
		self.transfer_learner = TransferLearner()
		self.analyser = Analyser()


if __name__ == '__main__':
	tc = TumorClassifier()
	# tc.data_utils.data_extract('train')
	# tc.data_utils.data_augment()
	# tc.data_utils.image_sampling(check_aug=True)
	# tc.data_utils.image_sampling()
	# tc.transfer_learner.train('resnet50')
	tc.analyser.roc_curve('y_valid_a', 'resnet50_valid_pred')
	tc.analyser.hotmap('resnet50')
