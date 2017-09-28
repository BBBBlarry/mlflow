# MLflow v0.01

**IMPORTANT: MLFLOW v0.01 IS STILL UNDER DEVELOPMENT AND IS NOT THOROUGHLY TESTED. PLEASE DO NOT USE IT FOR ANY APPLICATION PURPOSES.**

MLflow is causal machine learning library for python which is structured and operate similar to Google'e [Tensorflow](https://github.com/tensorflow/tensorflow). 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

You should have already installed [Python 2](https://www.python.org/download/releases/2.7.2/) and [NumPy](http://www.numpy.org/) on your machine before you can run mlflow. 

### Installing

After you download the code from this repo, you can use the makefile to build and install.  

```bash
# build
make build
# install
make install
```

### Example Code

Let's walk through how to create a basic one-layer neural network using mlflow. This code also be found under [/examples](/examples).

```python
# import dependencies
import mlflow as mf
import numpy as np
from tqdm import tqdm

# hyperparams
n_samples = 10
itr = 1000

# data set
x = np.ones((n_samples, 10))
y = np.ones((n_samples, 5))

# create a graph
# this is a crucial step for using mlflow
g = mf.Graph()

# input data placeholder
x_placeholder = mf.placeholder([n_samples, 10], graph = g)

# 1-layer NN 
W = mf.Variable(np.random.random((10, 5)), graph = g)
b = mf.Variable(np.zeros((5)), graph = g) 

# activation
z = mf.matmul(x_placeholder, W, graph = g) + b
pred = mf.sigmoid(z, graph = g)

# cost function
y_placeholder = mf.placeholder([n_samples, 5], graph = g)
cost = mf.sum(pow((pred - y_placeholder), 2.0), graph = g) / (2 * n_samples)

# to run the graph you need to create a mlflow session first
# create session
sess = mf.Session(g)
# run graph
res = sess.run(pred, feed_dict = {x_placeholder: x})
print res.shape

# show the initial cost
cost_res = sess.run(cost, feed_dict = {x_placeholder: x, y_placeholder: y})
print cost_res

# training
# use a gradient descent optimizer with a learning rate of 0.1
opt = mf.Train.GradientDescentOptimizer(0.1, graph = g).minimize(cost)
# show progress as we go
for i in tqdm(range(itr)):
	sess.run(opt, feed_dict = {x_placeholder: x, y_placeholder: y})
	
# show the final cost
print sess.run(cost, feed_dict = {x_placeholder: x, y_placeholder: y})
```

## Running the tests

**There are no good tests written for mlflow yet.**

To run the automated tests, type in the following command:

```bash
make test
```

## Versioning

[SemVer](http://semver.org/) is used for versioning. 

## Author

* **Blarry Wang** - *MLflow* - [BBBBlarry](https://github.com/BBBBlarry)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details