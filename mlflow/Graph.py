from .Train import Optimizer

class Graph:
	def __init__(self, adj = {}, logging = True): 
		self.adj = adj
		self.optimizable_variables = []

	def print_me(self):
		print repr(self.adj)

	def add_edge(self, vertex, node = None):
		if not vertex in self.adj:
			self.adj[vertex] = []

		if not node is None:
			if not node in self.adj[vertex]:
				self.adj[vertex] += [node]
			else:
				if logging:
					print "Edge already exist: from " + repr(vertex) + " to " + repr(node)

	def get_dependency(self, vertex):
		if not vertex in self.adj:
			return None
		else:
			return self.adj[vertex]

	def in_graph(self, vertex):
		return vertex in self.adj

	# deferred execution
	def exe(self, node):
		# running an op/tensor will use the DEG scheme
		if not self.in_graph(node):
			res = node.run()
		else:
			dependencies = self.get_dependency(node)
			data_flow = []

			for d in dependencies:
				data_flow += [self.exe(d)]

			res = node.run(*data_flow)

		if isinstance(node, Optimizer):
			return node
		else:
			return res

	def fill_gradients(self, node):
		dependencies = self.get_dependency(node)
		for d in dependencies:
			d.pass_gradients(node, *self.get_dependency(d))
			self.fill_gradients(d)


	############################################
	# add an optimizable variable to the graph, 
	# this will allow the optimizer later on to
	# be attached to them
	def add_opt_var(self, opt_var):
		self.optimizable_variables.append(opt_var)
	# return the optimizable variables
	def get_opt_vars(self):
		return self.optimizable_variables
