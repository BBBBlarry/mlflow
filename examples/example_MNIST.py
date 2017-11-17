import mlflow as mf
import numpy as np
from tqdm import tqdm

# to run this example, you have to install python-mnist and scikit learn
# use "pip install python-mnist"
from mnist import MNIST
# use "pip install sklearn"
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# for testing consistency
np.random.seed(42)

### data set
mndata = MNIST('./dataset')
x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

# make them numpy arrays
x_train, _y_train, x_test, _y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# hyperparams
l_rate = 0.5
s_rate = 0.2
training_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 50
n_batches = training_size / batch_size
epochs = 10
image_size = x_train.shape[1]
n_labels = 10

# onehot encode labels
y_train = np.zeros((training_size, n_labels))
y_test = np.zeros((test_size, n_labels))
# standarize data
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

for y_idx in range(training_size):
	hot_idx = _y_train[y_idx]
	y_train[y_idx, hot_idx] = 1

for y_idx in range(test_size):
	hot_idx = _y_test[y_idx]
	y_test[y_idx, hot_idx] = 1

print "Data preparation compelete..."



# get the graph from pickle
g = mf.Graph()

### input data place holder
x_placeholder = mf.placeholder([batch_size, image_size], graph = g)
y_placeholder = mf.placeholder([batch_size, n_labels], graph = g)

### NN layer 1
W1 = mf.Variable(np.random.random((image_size, 200)), graph = g)
b1 = mf.Variable(np.zeros((200)), graph = g) 

### activation
z1 = mf.matmul(x_placeholder, W1, graph = g) + b1
a1 = mf.sigmoid(z1, graph = g)

### NN layer 2
W2 = mf.Variable(np.random.random((200, n_labels)), graph = g)
b2 = mf.Variable(np.zeros((n_labels)), graph = g) 

### activation
z2 = mf.matmul(a1, W2, graph = g) + b2
pred = mf.sigmoid(z2, graph = g)


### cost function
cost = mf.sum(pow((pred - y_placeholder), 2.0), graph = g) / (2 * batch_size)

# optimizer
opt = mf.Train.AdaGradOptimizer(l_rate, graph = g).minimize(cost)


# create session and run prediction!
sess = mf.Session(g)
for e in range(epochs):
	print "Epoch " + str(e)
	print "Cost: ", sess.run(cost, feed_dict = {x_placeholder: x_train[:batch_size], y_placeholder: y_train[:batch_size]})
	for b in tqdm(range(n_batches)):
		sess.run(opt, feed_dict = {x_placeholder: x_train[b * batch_size:(b + 1) * batch_size], y_placeholder: y_train[b * batch_size:(b + 1) * batch_size]})
	x_train, y_train = shuffle(x_train, y_train, random_state=42)

print "50 Sample Train Cost: \n", sess.run(cost, feed_dict = {x_placeholder: x_train[:batch_size], y_placeholder: y_train[:batch_size]})
print "50 Sample Test Cost: \n", sess.run(cost, feed_dict = {x_placeholder: x_test[:batch_size], y_placeholder: y_test[:batch_size]})



