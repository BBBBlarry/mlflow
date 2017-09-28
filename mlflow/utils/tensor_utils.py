import numpy as np


def is_scalar(tensor):
	# must be numpy type
	assert type(tensor).__module__ == np.__name__
	return tensor.shape == (1,) or tensor.shape == ()
