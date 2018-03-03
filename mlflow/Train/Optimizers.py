import numpy as np


class Optimizer(object):
	def __init__(self, graph):
		self.graph = graph
		self.opt_vars = graph.get_opt_vars()

	# the default mode of running this op will be minimization
	# TODO(Blarry): add maximation
	def minimize(self, var):
		# add self to graph
		self.graph.add_edge(self, var)
		self.cost_node = var
		return self

	#TODO(blarry): dummy_cost is here to fit into the
	# deferred execution scheme, should be fixed
	def run(self, dummy_cost):
		self.cost_node.set_default_gradients()
		self.graph.fill_gradients(self.cost_node)

		#TODO(blart): parallelize this
		for var in self.opt_vars:
			grad = self.get_gradients(var)
			self.opt(var, grad)

	# no default optimzation strategy
	# TODO(Blarry): add a warning
	def opt(self, var, grad):
		pass

	def get_gradients(self, opt_var):
		return opt_var.gradients


class GradientDescentOptimizer(Optimizer):
	def __init__(self, learning_rate, graph):
		super(GradientDescentOptimizer, self).__init__(graph)
		self.learning_rate = learning_rate

	# optimzation stage
	def opt(self, var, grad):
		assert not grad is None
		assert grad.shape == var.data.shape
		var.update(var.data - grad * self.learning_rate)


class AdaGradOptimizer(Optimizer):
	def __init__(self, learning_rate, graph, eps=1e-6):
		super(AdaGradOptimizer, self).__init__(graph)
		self.learning_rate = learning_rate
		self.eps = 1e-6
		self.hist_grad = dict.fromkeys(self.opt_vars)

	# optimzation stage
	def opt(self, var, grad):
		assert not grad is None
		assert grad.shape == var.data.shape
		assert var in self.hist_grad

		# first run, initialize historical grad (to 0)
		if self.hist_grad[var] is None:
			self.hist_grad[var] = np.zeros_like(grad)

		# update historical gradients
		self.hist_grad[var] += np.power(grad, 2)
		# calculate the adjusted gradient
		adj_grad = grad / (self.eps + np.sqrt(self.hist_grad[var]))

		var.update(var.data - adj_grad * self.learning_rate)


### This optimizer is implemented just for fun,
### it's goal is to introduce a new regularization
### technique.
### Be cautious about using it until I test
### it's effectiveness.
class RandomGradientOptimizer(Optimizer):
	def __init__(self, learning_rate, skip_rate, graph):
		super(RandomGradientOptimizer, self).__init__(graph)
		self.learning_rate = learning_rate
		self.skip_rate = min(max(skip_rate, 0.0), 1.0)

	# optimzation stage
	def opt(self, var, grad):
		assert not grad is None
		assert grad.shape == var.data.shape

		# only optimize a few variables based on skip rate
		if np.random.random() > self.skip_rate:
			var.update(var.data - grad * self.learning_rate)
