import numpy as np

from ..utils import *


class Sigmoid():
	def __init__(self):
		pass
	def run(self, a):
		return 1.0 / (1.0 + np.exp(-a))

	def pass_gradients(self, out_tensor, in_tensor):
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor.set_gradients(out_tensor_grad * out_tensor.data * (1.0 - out_tensor.data))


class Matmul():
	def __init__(self):
		pass

	def run(self, a, b):
		return np.dot(a, b)

	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor_1.set_gradients(np.dot(out_tensor_grad, in_tensor_2.data.T))
		in_tensor_2.set_gradients(np.dot(in_tensor_1.data.T, out_tensor_grad))

class Sum():
	def __init__(self):
		pass
	def run(self, a):
		return np.sum(a)

	def pass_gradients(self, out_tensor, in_tensor):
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		assert is_scalar(out_tensor_grad)

		in_tensor.set_gradients(np.full(in_tensor.data.shape, out_tensor_grad))
