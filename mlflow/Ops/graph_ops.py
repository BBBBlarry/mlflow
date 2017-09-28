from ..Exceptions.graph_exceptions import *


class Placeholder():
	def __init__(self, variable_node, size):
		self.__variable_node = variable_node
		self.size = size

	@property
	def data(self):
		return self.__variable_node.data

	# placeholders do not have gradients
	def set_gradients(self, value):
		pass

	# assign value to its child
	def assign(self, value):
		# TODO(blarry): add exception
		if value.shape != tuple(self.size):
			raise PlaceholderSizeMismatchError("Placeholder size mismatch: expect size " + repr(self.size))
		self.__variable_node.update(value)

	def run(self, value):
		# pass value from its child variable
		return value

	def pass_gradients(self, out_tensor, in_tensor):
		# placeholder will not pass gradient for no dependencies
		# will need to be updated
		pass
