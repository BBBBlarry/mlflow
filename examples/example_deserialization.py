import mlflow as mf
import numpy as np

### data set
x = np.ones((10, 10))
y = np.ones((10, 5))

# get the graph from pickle
g = mf.Graph(from_pickle = True, pickle_name = "my_model")

# we can make predictions using the model!
pred = g.get_node("prediction")
x_placeholder = g.get_node("x_placeholder")
y_placeholder = g.get_node("y_placeholder")

# create session and run prediction!
sess = mf.Session(g)
print "Prediction: \n", sess.run(pred, feed_dict = {x_placeholder: x, y_placeholder: y})
