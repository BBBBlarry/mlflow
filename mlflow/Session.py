from Exceptions import *
from Graph import Graph
from Ops.graph_ops import Placeholder


class Session:
	def __init__(self, graph):
		self.graph = graph

	def run(self, node, feed_dict = {}):
		if not self.graph.in_graph(node):
			raise NotInGraphException("Variable \"" + repr(node) + "\" not in graph.")
		# feed values frist
		self.__feed(feed_dict)

		return self.graph.exe(node)

	# feed values to placeholder variables
	def __feed(self, feed_dict):
		for placeholder in feed_dict:
			# type check
			if not isinstance(placeholder, Placeholder):
				raise FeedDictError(repr(placeholder) + " not a placeholder.")
			# assign value
			placeholder.assign(feed_dict[placeholder])
