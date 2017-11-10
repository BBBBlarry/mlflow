import mlflow as mf
import numpy as np
from tqdm import tqdm

# for testing consistency
np.random.seed(42)

### hyperparams
n_samples = 10
itr = 1000

### data set
x = np.ones((n_samples, 10))
y = np.ones((n_samples, 5))

### graph
g = mf.Graph()

### input data place holder
x_placeholder = mf.placeholder([n_samples, 10], graph = g)

### NN layer
W = mf.Variable(np.random.random((10, 5)), graph = g)
b = mf.Variable(np.zeros((5)), graph = g) 

### activation
z = mf.matmul(x_placeholder, W, graph = g) + b
pred = mf.sigmoid(z, graph = g)

### cost function
y_placeholder = mf.placeholder([n_samples, 5], graph = g)
cost = mf.sum(pow((pred - y_placeholder), 2.0), graph = g) / (2 * n_samples)

### run the graph
sess = mf.Session(g)
res = sess.run(pred, feed_dict = {x_placeholder: x})
cost_res = sess.run(cost, feed_dict = {x_placeholder: x, y_placeholder: y})


### training step
# compare the optimization methods
#opt = mf.Train.GradientDescentOptimizer(0.1, graph = g.minimize(cost)
opt = mf.Train.AdaGradOptimizer(0.1, graph = g).minimize(cost)

for i in tqdm(range(itr)):
	sess.run(opt, feed_dict = {x_placeholder: x, y_placeholder: y})

print "Prediction: \n", sess.run(pred, feed_dict = {x_placeholder: x, y_placeholder: y})


## tagging!
# since you won't have access to the node objects you created here after 
# you serialize and deserialize, you have to tag the nodes you want to be 
# saved so that they can be called in the future!

g.tag_node(opt, "opt")
g.tag_node(pred, "prediction")
g.tag_node(x_placeholder, "x_placeholder")
g.tag_node(y_placeholder, "y_placeholder")

g.serialize("my_model")