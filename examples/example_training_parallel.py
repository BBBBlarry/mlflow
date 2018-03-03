import mlflow as mf
import numpy as np
from tqdm import tqdm

# for testing consistency
np.random.seed(42)

### hyperparams
n_samples = 10
itr = 1000

### data set
x = np.random.random((n_samples, 10))
y = np.zeros((n_samples, 10))
# find the integer sum of the input and one-hot encode into y
target_idx = np.floor(np.sum(x, axis = 1)).astype(int)
y[np.arange(y.shape[0]), target_idx] = 1

### graph
GRAPH_OC = {
	"FILLGRADIENTS_PARA_FLAG": True,
	"FILLGRADIENTS_PARA_CUTOFF": 1
}
g = mf.Graph(optimization_configs = GRAPH_OC)

### input data place holder
x_placeholder = mf.placeholder([n_samples, 10], graph = g)

### NN layer
W = mf.Variable(np.random.random((10, 10)), graph = g)
b = mf.Variable(np.zeros((10)), graph = g)

### activation
z = mf.matmul(x_placeholder, W, graph = g) + b
pred = mf.sigmoid(z, graph = g)

### cost function
y_placeholder = mf.placeholder([n_samples, 10], graph = g)
cost = mf.sum(pow((pred - y_placeholder), 2.0), graph = g) / (2 * n_samples)

### run the graph
sess = mf.Session(g)
res = sess.run(pred, feed_dict = {x_placeholder: x})
print res.shape
cost_res = sess.run(cost, feed_dict = {x_placeholder: x, y_placeholder: y})
print cost_res


### training step
# compare the optimization methods
#opt = mf.Train.GradientDescentOptimizer(0.8, graph = g).minimize(cost)
opt = mf.Train.AdaGradOptimizer(0.8, graph = g).minimize(cost)

for i in tqdm(range(itr)):
	sess.run(opt, feed_dict = {x_placeholder: x, y_placeholder: y})

print sess.run(cost, feed_dict = {x_placeholder: x, y_placeholder: y})
