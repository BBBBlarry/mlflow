from Ops import *
from Tensor import Variable


###### graph ops ######
def placeholder(size, graph):
	variable_node = Variable(None, graph)
	op_node = Placeholder(variable_node, size)

	graph.add_edge(op_node, variable_node)

	return op_node

###### math ops ######
###################################
### All current math ops are    ###
### overloaded within Variables ###
### and Constants. 				###
###################################

###### linear algebra ######
def sum(a, graph):
	variable_node = Variable(None, graph)
	op_node = Sum()

	graph.add_edge(op_node, a)
	graph.add_edge(variable_node, op_node)

	return variable_node
	
def matmul(a, b, graph):
	op_node = Matmul()
	output_node = Variable(None, graph)

	graph.add_edge(op_node, a)
	graph.add_edge(op_node, b)
	graph.add_edge(output_node, op_node)

	return output_node

def sigmoid(a, graph):
	op_node = Sigmoid()
	output_node = Variable(None, graph)

	graph.add_edge(op_node, a)
	graph.add_edge(output_node, op_node)

	return output_node
