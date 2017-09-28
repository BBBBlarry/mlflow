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

	# optimzation strage
	def opt(self, var, grad):
		assert not grad is None
		assert grad.shape == var.data.shape
		var.update(var.data - grad * self.learning_rate)
