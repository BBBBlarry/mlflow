from copy import deepcopy
from .Train import Optimizer
import pickle
import os

class Graph:
	def __init__(self, adj = {}, logging = True, optimizable_variables = [], tags = {}, from_pickle=False, pickle_name = "model", pickle_path = "./models/"): 
		# initialize from scratch
		if not from_pickle:
			self.initialize_graph(adj, logging, optimizable_variables, tags)
		else:
			# init from pickle
			self.deserialize(pickle_name, pickle_path)


	def initialize_graph(self, adj, logging, optimizable_variables, tags):
		self.adj = adj
		self.optimizable_variables = optimizable_variables
		self.tags = tags
		self.logging = logging


	def print_me(self):
		print repr(self.adj)

	def add_edge(self, vertex, node = None):

		if not vertex in self.adj:
			self.adj[vertex] = []

		if not node is None:
			if not node in self.adj[vertex]:
				self.adj[vertex] += [node]
			else:
				if self.logging:
					print "Edge already exist: from " + repr(vertex) + " to " + repr(node)

	# add the name of the node to a dictionary
	def tag_node(self, node, tag):
		# do nothing to none
		if node is not None:
			if tag in self.tags:
				# duplicate name
				if logging:
					print "Bad node tag: " + tag + " is already used."
			else:
				# add name for future reference
				self.tags[tag] = node
	
	# get a node by its name
	def get_node(self, tag):
		if tag in self.tags:
			return self.tags[tag]
		else:
			if logging:
				print "Bad node name: " + tag + " is not found in graph."
			return None

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

	# return a copy of this graph
	def copy(self):
		return Graph(adj = deepcopy(self.adj), logging = True, \
			optimizable_variables = deepcopy(self.optimizable_variables))


	##### for serialization #####
	def serialize(self, name, path="./models/"):
		if len(self.tags) == 0:
			if self.logging:
				print "Warning: serialization of non-tagged graph."

		full_path = path+name+".model"
		if not os.path.exists(os.path.dirname(full_path)):
			try:
				os.makedirs(os.path.dirname(full_path))
			except OSError as exc: # race condition
				if exc.errno != errno.EEXIST:
					raise

		with open(full_path, 'w') as to_file:
			pickle.dump(self, to_file)

	def deserialize(self, name, path="./models/"):
		full_path = path+name+".model"
		if not os.path.exists(os.path.dirname(full_path)):
			raise 

		with open(full_path, 'r') as from_file:
			data = from_file.read()
			g = pickle.loads(data)
			self.initialize_graph(g.adj, g.logging, g.optimizable_variables, g.tags)

	############################################
	# add an optimizable variable to the graph, 
	# this will allow the optimizer later on to
	# be attached to them
	def add_opt_var(self, opt_var):
		self.optimizable_variables.append(opt_var)
	# return the optimizable variables
	def get_opt_vars(self):
		return self.optimizable_variables
