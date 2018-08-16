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
	# tc.data_utils.image_sampling(mode='valid', check_aug=True)
	# tc.data_utils.image_sampling(mode='valid')
	# tc.transfer_learner.train('resnet50')
	# tc.transfer_learner.cross_validation('resnet50')
	# tc.analyser.roc_curve('y_valid_a', 'resnet50_valid_pred')
	# tc.analyser.layer_visualizer('resnet50', stage=3, mode='train', source='0149.jpg')
	# tc.analyser.heatmap('resnet50', mode='train', source=['0120.jpg', '0219.jpg', '0222.jpg'])
	# tc.data_utils.data_extract('test')
	# model = tc.transfer_learner.train('resnet50', 'model')
	# tc.transfer_learner.test('resnet50', model)
	# tc.transfer_learner.test('resnet50')
	# tc.analyser.roc_curve('y_test', 'resnet50_test_pred')
	# for i in range(5):
	# 	tc.analyser.layer_visualizer('resnet50', stage=i+1, source='0996.jpg')
	tc.analyser.heatmap('resnet50', load=False, source=['0593.jpg', '0942.jpg', '0971.jpg'])
