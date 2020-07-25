from abc import abstractmethod

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from bagging.underbagging import UnderBagging
from utils.check_measure import gm_measure

class ChunkBase:
	def __init__(self):
		self.ensemble = list()
		self.chunk_count = 0
		self.train_count = 0
		self.w = np.array([])
		self.buf_data = np.array([])
		self.buf_label = np.array([])

	@abstractmethod
	def _update_chunk(self, data, label):
		pass

	def _predict_base(self, test_data, prob_output=False):
		if len(self.ensemble) == 0:
			pred = np.zeros(test_data.shape[0])
		else:
			pred = np.zeros([test_data.shape[0], len(self.ensemble)])
			for i in range(len(self.ensemble)):
				if prob_output:
					pred[:, i] = self.ensemble[i].predict_proba(test_data)[:, 1]
				else:
					pred[:, i] = self.ensemble[i].predict(test_data)

		return pred

	def update(self, single_data, single_label):
		pred = self.predict(single_data.reshape(1, -1))

		if self.buf_label.size < self.chunk_size:
			self.buf_data = np.r_[self.buf_data.reshape(-1, single_data.shape[0]), single_data.reshape(1, -1)]
			self.buf_label = np.r_[self.buf_label, single_label]
			self.train_count += 1

		if self.buf_label.size == self.chunk_size or self.train_count == self.data_num:
			self._update_chunk(self.buf_data, self.buf_label)
			self.buf_data = np.array([])
			self.buf_label = np.array([])

		return pred

	def update_chunk(self, data, label):
		pred = self.predict(data)
		self._update_chunk(data, label)

		return pred

	def predict(self, test_data):
		all_pred = np.sign(self._predict_base(test_data))
		
		if len(self.w) != 0:
			pred = np.sign(np.dot(all_pred, self.w))
		else:
			pred = all_pred

		return pred

	def calculate_err(self, all_pred, label):
		ensemble_size = all_pred.shape[1]
		err = np.zeros(ensemble_size)
		
		for i in range(ensemble_size):
			if self.err_func == 'gm':
				err[i] = 1 - gm_measure(all_pred[:, i], label)

			elif self.err_fun == 'f1':
				err[i] = 1 - f1_score(label, all_pred[:, i])

		return err
