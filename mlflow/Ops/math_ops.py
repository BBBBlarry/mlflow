import numpy as np

from ..utils import *


class Add():
	def __init__(self):
		pass
	def run(self, a, b):
		return np.add(a, b)

	# takes the the output variable tensor and input variable tensors
	# set the grads of the input variable tensors
	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		# in case of bias adding
		if len(in_tensor_1.shape) < len(out_tensor_grad.shape):
			in_tensor_1.set_gradients(np.sum(out_tensor_grad, axis = 0))
		else:
			in_tensor_1.set_gradients(out_tensor_grad)

		if len(in_tensor_2.shape) < len(out_tensor_grad.shape):
			in_tensor_2.set_gradients(np.sum(out_tensor_grad, axis = 0))
		else:
			in_tensor_2.set_gradients(out_tensor_grad)
		

class Sub():
	def __init__(self):
		pass
	def run(self, a, b):
		return np.add(a, -b)

	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor_1.set_gradients(out_tensor_grad)
		in_tensor_2.set_gradients(-out_tensor_grad)

class Mul():
	def __init__(self):
		pass
	def run(self, a, b):
		return a * b

	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert is_scalar(in_tensor_1.data) or is_scalar(in_tensor_2.data)
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor_1.set_gradients(np.dot(out_tensor_grad, in_tensor_2.data.T))
		in_tensor_2.set_gradients(np.dot(in_tensor_1.data.T, out_tensor_grad))


class Div():
	def __init__(self):
		pass
	def run(self, a, b):
		return a / b

	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert is_scalar(in_tensor_2.data)
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor_1.set_gradients(np.dot(out_tensor_grad, (1.0 / in_tensor_2.data).T))
		in_tensor_2.set_gradients(None)
		#TODO(blarry): implemenet
		#print "WARNING: Passing grad to the second input tensor is not implemented."

class Pow():
	def __init__(self):
		pass
	def run(self, b, p):
		return np.power(b, p) 

	def pass_gradients(self, out_tensor, in_tensor_1, in_tensor_2):
		assert is_scalar(in_tensor_2.data)
		assert not out_tensor.get_gradients() is None

		out_tensor_grad = out_tensor.get_gradients()
		in_tensor_1.set_gradients(in_tensor_2.data * np.power(in_tensor_1.data, in_tensor_2.data - 1.0))
		in_tensor_2.set_gradients(None)
		#TODO(blarry): implemenet
		#print "WARNING: Passing grad to the second input tensor is not implemented."
