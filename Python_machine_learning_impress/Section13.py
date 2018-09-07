
# coding: utf-8

# In[ ]:


#Tensor flow >> GIL (global inter prilock)
#Low layer API >> Layers, Keras
import tensorflow as tf
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name="x")
    w = tf.Variable(2.0, name="weight")
    b = tf.Variable(0.7, name="bias")
    z = w * x + b
    init = tf.global_variables_initializer()
    
with tf.Session(graph=g) as sess:
    sess.run(init)
    for t in [1.0, 0.6, -1.8]:
        print("x={:>4.1f} --> z={:>4.1f}".format(t, sess.run(z, feed_dict={x:t})))


# In[ ]:


#place holder >> variables
#feed_dict >> feeding(=provideing) dictionary
with tf.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x: [1.0, 2.0, 3.0]}))


# In[ ]:


import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="input_x")
    x2 = tf.reshape(x, shape=(-1, 6), name="x2")
    
    xsum = tf.reduce_sum(x2, axis=0, name="col_sum")
    xmean = tf.reduce_mean(x2, axis=0, name="col_mean")
with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)
    print("input shape:", x_array.shape)
    print("Reshaped:\n", sess.run(x2, feed_dict={x: x_array}) )
    print("Column_sum:\n", sess.run(xsum, feed_dict={x: x_array}))
    print("Column_means:\n", sess.run(xmean, feed_dict={x: x_array}))


# In[ ]:


import tensorflow as tf
import numpy as np

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

class TfLinreg(object):
    
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            #building tensor flow
            self.build()
            self.init_op = tf.global_variables_initializer()
            
    def build(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim),
                               name="x_input")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None),
                               name="y_input")
        print(self.X)
        print(self.y)
        #weight matrics and bias vector
        w = tf.Variable(tf.zeros(shape=(1)), name="weight")
        b = tf.Variable(tf.zeros(shape=(1)), name="bias")
        print(w)
        print(b)

        self.z_net = tf.squeeze(w * self.X + b, name="z_net")
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net, name="sqr_errors")
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name="mean_cost")

        optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate,
                    name="GradientDecent")
        self.optimizer = optimizer.minimize(self.mean_cost)


# In[ ]:


lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)


# In[ ]:


def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    sess.run(model.init_op)
    #initialize variables >> w, b
    training_costs = []
    for i in range(num_epochs):
        n, cost = sess.run([model.optimizer, model.mean_cost],
                          feed_dict={model.X:X_train, model.y:y_train})
        print(n)
        training_costs.append(cost)
    return training_costs


# In[ ]:


sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(range(1, len(training_costs)+1), training_costs)
plt.tight_layout()
plt.xlabel("Epoch")
plt.ylabel("Training_cost")
plt.show()


# In[ ]:


def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net, feed_dict={model.X:X_test})
    return y_pred


# In[ ]:


plt.scatter(X_train, y_train, marker="s", s=50, label="training_data")
plt.plot(range(X_train.shape[0]), predict_linreg(sess, lrmodel, X_train),
        color="grey", marker="o", markersize=6, linewidth=3, label ="LinReg Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#layers api , karas api
import os
import struct
import numpy as np
def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "{}-labels.idx1-ubyte".format(kind))
    images_path = os.path.join(path, "{}-images.idx3-ubyte".format(kind))
    
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.0) - 0.5) ** 2
    return images, labels


# In[ ]:


X_train, y_train = load_mnist(".", kind="train")
print("Rows: {:d}, Columns: {:d}".format(X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist(".", kind="t10k")
print("Rows: {:d}, Columns: {:d}".format(X_test.shape[0], X_test.shape[1]))
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val
del X_train, X_test
print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)


# In[ ]:


import tensorflow as tf
n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name="tf_x")
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name="tf_y")
    y_onehot = tf.one_hot(indices=tf_y,depth=n_classes)
    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name="layer1")
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name="layer2")
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name="layer3")
    predictions = {
        "classes" : tf.argmax(logits, axis=1, name="predicted_classes"),
        "probablities" : tf.nn.softmax(logits, name="softmax_tensor")
    }


# In[ ]:


with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=cost)
    init_op = tf.global_variables_initializer()
    


# In[ ]:


def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)
    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
        
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i: i+batch_size, :], y_copy[i: i+batch_size])


# In[ ]:


sess = tf.Session(graph=g)
sess.run(init_op)
#50 epochs in training
training_costs = []
for epoch in range(50):
    training_loss = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)
    for batch_X, batch_y, in batch_generator:
        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(" -- Epoch {:2d} AVG. training loss : {:.4f}".format(epoch+1, np.mean(training_costs)))


# In[ ]:


feed =  {tf_x : X_test_centered}
y_pred = sess.run(predictions["classes"], feed_dict=feed)
print("Test Accuracy: {:.2f}".format(100 * np.sum(y_pred == y_test) / y_test.shape[0]))


# In[ ]:


#Lets use Karas!
#instead of code below, you can use it.
#from kares.datasets import mnist
#(train_imges, train_labels), (test_images, test_labels) = mnist.load_data()
X_train,  y_train = load_mnist("./", kind="train")
print("Rows : {:d}, Cols : {:d}".format(X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist("./", kind="t10k")
print("Rows : {:d}, Cols : {:d}".format(X_test.shape[0], X_test.shape[1]))

#Standardization & Reguralization
mean_vals = np.mean(X_train, axis=0)
print(mean_vals.shape)
std_val = np.std(X_train)
print(std_val.shape, std_val)
X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val

del X_train, X_test
print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)


# In[ ]:


import tensorflow as tf
import keras 
np.random.seed(123)
tf.set_random_seed(123)
y_train_onehot = keras.utils.to_categorical(y_train)
print("First 3 labels:", y_train[:3])
print("\n First 3 labels in onehote: \n", y_train_onehot[:3])


# In[ ]:


#lets implementation neural network
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=50, input_dim=X_train_centered.shape[1],
                            kernel_initializer="glorot_uniform",
                             bias_initializer="zeros", activation="tanh"))
model.add(keras.layers.Dense(units=50, input_dim=50, 
                            kernel_initializer="glorot_uniform",
                            bias_initializer="zeros", activation="tanh"))
model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=50,
                            kernel_initializer="glorot_uniform",
                            bias_initializer="zeros", activation="softmax"))
#model compiling
sgd_optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9)
model.compile(optimizer=sgd_optimizer, loss="categorical_crossentropy")
#Glorot initialization >> famous method
#keras.optimizer.SGD class 


# In[ ]:


history = model.fit(X_train_centered, y_train_onehot, batch_size=64,
                   epochs=50, verbose=1, validation_split=0.1)
#verbose >> output comments


# In[ ]:


y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print("first 3 predicttions :",y_train_pred[:3])


# In[ ]:


y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print("First 3 predictions:", y_train_pred[:3])
print("Training accuracy: {:.2f}".format(train_acc * 100))
y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print("First 3 predictions:", y_test_pred[:3])
print("Test accuracy: {:.2f}".format(test_acc * 100))


# In[ ]:


#logistic function takes much time, so hyperbolic tangent function is very useful
import numpy as np 
X = np.array([1, 1.4, 2.5])
w = np.array([0.4, 0.3, 0.5])
def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print("P(y=1|x) = {:.3f}".format(logistic_activation(X, w)))
#Possibility of class 1


# In[ ]:


W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.5, 1.0, 0.3],
              [0.4, 1.1, 0.5, 0.8]])
#A[0][0] must be 1.0
A = np.array([[1.0, 0.1, 1.5, 0.5]])

Z = np.dot(W, A[0])
y_probables = logistic(Z)
print("Net Input:", Z)
print("OUT put units", y_probables)


# In[ ]:


y_class = np.argmax(Z, axis=0)
print("Predicted Class Label: {:d}".format(y_class))


# In[ ]:


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print("Probabilities:", y_probas)
np.sum(y_probas)


# _hyperbolic tangent_
# $(exp(x) - exp(-x)) / (exp(x) + (exp(-x)))$

# In[ ]:


import matplotlib.pyplot as plt
def tanh(z):
    e_p = np.exp(z)
    e_n = np.exp(-z)
    return (e_p - e_n) / (e_p + e_n)

z = np.arange(-5, 5, 0.0005)
logistic_act = logistic(z)
tanh_act = tanh(z)
plt.ylim = ([-1.5, 1.5])
plt.xlabel("Net input $z$")
plt.ylabel("activation $/phi(z)$")
for n in [1, 0.5, 0, -0.5, -1]:
    plt.axhline(n, color="black", linestyle=":")
plt.plot(z, tanh_act, linewidth=3, linestyle="--", label="tanh")
plt.plot(z, logistic_act, linewidth=3, label="logistic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[ ]:


tanh_act = np.tanh(z)


# In[ ]:


from scipy.special import expit
log_act = expit

