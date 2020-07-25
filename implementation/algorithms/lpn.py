import numpy as np

from bagging.underbagging import UnderBagging
from chunk_base.chunk_base import ChunkBase

class LearnppNIE(ChunkBase):
	def __init__(self, data_num, chunk_size, T=5, a=0.5, b=10, err_func='gm'):
		ChunkBase.__init__(self)
		self.T = T
		self.data_num = data_num
		self.chunk_size = chunk_size
		self.a = a
		self.b = b
		self.err_func = err_func
		self.beta = np.array([[0.0]])

	def _update_chunk(self, data, label):
		model = UnderBagging(T=self.T, auto_r=True)
		model.train(data, label)
		self.ensemble.append(model)
		self.chunk_count += 1
		all_pred = np.sign(self._predict_base(data))

		if self.chunk_count > 1:
			pred = np.dot(all_pred[:, :-1], self.w)
		else:
			pred = np.zeros_like(label)

		pred = np.sign(pred)

		err = self.calculate_err(all_pred, label)

		if err[-1] > 0.5:
			model = UnderBagging(T=self.T, auto_r=True)
			model.train(data, label)
			self.ensemble[-1] = model
			all_pred = np.sign(self._predict_base(data))
			err = self.calculate_err(all_pred, label)
			if err[-1] > 0.5:
				err[-1] = 0.5

		err[err > 0.5] = 0.5

		if self.chunk_count == 1:
			self.beta[0, 0] = err / (1 - err)
		else:
			self.beta = np.pad(self.beta, ((0, 1), (0, 1)), 'constant', constant_values=(0))
			self.beta[:self.chunk_count, self.chunk_count - 1] = err / (1 - err)

		self.w = np.zeros(self.chunk_count)
		
		for k in range(self.chunk_count):
			omega = np.array(range(1, self.chunk_count - k + 1))
			omega = 1 / (1 + np.exp(-self.a * (omega - self.b)))
			omega /= np.sum(omega)
			beta_hat = np.sum(omega * self.beta[k, k:self.chunk_count])
			self.w[k] = np.log(1 / beta_hat)

		return pred
