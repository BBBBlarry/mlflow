from .context import mlflow as mf

## test if the packages and modules are present
def test_components():
	assert mf.Graph
	assert mf.Session
	assert mf.Tensor
	assert mf.Ops
	assert mf.Train
	assert mf.Exceptions
	assert mf.utils

	