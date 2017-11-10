from Ops.math_ops import *


class Tensor(object):
	def __init__(self, data, graph):
		self.data = np.array(data)
		self.graph = graph
		self.graph.add_edge(self)

	@property
	def shape(self):
		return self.data.shape

	def run(self, op_res = None):
		return self.data

	###### gradient based optimization ######
	#TODO(blarry): figure out a better way to pass 
	# 			   gradients from Tensors
	def pass_gradients(self, out_tensor, in_tensor = None):
		pass

	def set_gradients(self, value):
		pass

	def get_gradients(self):
		return None

	###### operator override ######
	def __add__(self, other):
		other = self.__num_to_constant(other)

		variable_node = Variable(None, self.graph)
		op_node = Add()

		self.graph.add_edge(op_node, self)
		self.graph.add_edge(op_node, other)
		self.graph.add_edge(variable_node, op_node)

		return variable_node
	
	def __sub__(self, other):
		other = self.__num_to_constant(other)

		variable_node = Variable(None, self.graph)
		op_node = Sub()

		self.graph.add_edge(op_node, self)
		self.graph.add_edge(op_node, other)
		self.graph.add_edge(variable_node, op_node)

		return variable_node

	def __mul__(self, other):
		other = self.__num_to_constant(other)

		variable_node = Variable(None, self.graph)
		op_node = Mul()

		self.graph.add_edge(op_node, self)
		self.graph.add_edge(op_node, other)
		self.graph.add_edge(variable_node, op_node)

		return variable_node

	def __div__(self, other):
		other = self.__num_to_constant(other)

		variable_node = Variable(None, self.graph)
		op_node = Div()

		self.graph.add_edge(op_node, self)
		self.graph.add_edge(op_node, other)
		self.graph.add_edge(variable_node, op_node)

		return variable_node

	def __pow__(self, other):
		other = self.__num_to_constant(other)

		variable_node = Variable(None, self.graph)
		op_node = Pow()

		self.graph.add_edge(op_node, self)
		self.graph.add_edge(op_node, other)
		self.graph.add_edge(variable_node, op_node)

		return variable_node


	###### private helper methods ######
	def __num_to_constant(self, a):
		if isinstance(a, int) or isinstance(a, float):
			a = Constant(a, self.graph)
		return a



class Variable(Tensor):
	def __init__(self, data, graph):
		super(Variable, self).__init__(data, graph)
		# when data is not None, we assume 
		# this variable is optimizable
		if not data is None:
			graph.add_opt_var(self)

		# the gradient used for optimization
		self.gradients = self.data

	###### gradient based optimization ######
	def set_default_gradients(self):
		self.gradients = np.full(self.shape, 1.0)

	def set_gradients(self, value):
		self.gradients = value

	def get_gradients(self):
		return self.gradients

	###### Basic variable node methods ######
	def update(self, value):
		self.data = value

	def run(self, op_res = None):
		# is a result of an op
		if not op_res is None:
			self.data = op_res
		return self.data

class Constant(Tensor):
	pass
