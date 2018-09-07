
# coding: utf-8

# ## Convolutional Neural Network
# ### Elements of CNN
# - salient feature
# - CNN construct feature hierarchy and caliculate feature map
# - CNN concepts 
#     - loosely coupled
#     - sharing paramators
# - CNN layers
#     * convolutional layer
#     * subsampling layer (== pooling layer)
#     * fully connected layer
#     
# ### discrete convolution
# * padding
#     * stride >> one of hyperparamator that is number of shifts
# * zerp padding
#     - full mode
#     - same mode
#     - vaild mode
#     
# $ output = \lfloor \frac {n+2p-m}{s} \rfloor + 1 $

# In[ ]:


import numpy as np

def conv1d (x, w, p=0, s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
        res = []
        for i in range(0, int(len(x) / s), s):
            res.append(np.sum(x_padded[i: i + w_rot.shape[0]] * w_rot))
        return np.array(res)
    
x = [1, 3, 5, 2, 8, 5, 5, 4]
w = [4, 2, 0, 5]
get_ipython().run_line_magic('time', 'print("conv1d Implementation:", conv1d(x, w, p=2, s=1))')
get_ipython().run_line_magic('time', 'print("Numpy results:", np.convolve(x, w, mode="same"))')


# In[ ]:


import numpy as np
import scipy.signal

def conv2d(X, W, p=(0, 0), s=(1, 1)):
    W_rot = np.array(W)[::-1, ::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2 * p[0]
    n2 = X_orig.shape[1] + 2 * p[1]
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]: p[0]+X_orig.shape[0],
            p[1]: p[1]+X_orig.shape[1]] = X_orig
    res = []
    for i in range(0, int((X_padded.shape[0] - W_rot.shape[0]) / s[0] + 1), s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - W_rot.shape[1]) / s[1]) + 1, s[1]):
            X_sub = X_padded[i: i+W_rot.shape[0], j: j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))
    return (np.array(res))
                             
X = [[5, 2, 5, 9], [7, 5, 2, 8], [8, 2, 0, 6], [7, 1, 0, 0]]
W = [[1, 2, 8], [7, 1, 0], [7, 6, 5]]
print("conv2d Implementation:\n", conv2d(X, W, p=(1, 1), s=(1, 1)))
print("scipy results: \n", scipy.signal.convolve2d(X, W, mode="same"))


# ## pooling
# - max-pooling
# - mean-pooling
# > pooling size

# ## multiple input channel
# ### output channel >> feature map
# ### regularization >> droplut

# ## multiple CNN architecture
# - input layer
# - cnn1
# - pooling1
# - cnn2
# - pooling2
# - full-connected-layer1
# - full-connected-layer2 & softmax-layer

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
X_data, y_data = load_mnist("./", kind="train")
print("rows: {:d}, cols: {:d}".format(X_data.shape[0], X_data.shape[1]))
X_test, y_test = load_mnist("./")
print("rows: {:d}, cols: {:d}".format(X_test.shape[0], X_test.shape[1]))
X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]
print("training:", X_train.shape, y_train.shape)
print("valid:", X_valid.shape, y_valid.shape)
print("test:", X_test.shape, y_test.shape)


# In[ ]:


#dealing data by mini-batch
def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):
    
    ind = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]
        
    for i in range(0, X.shape[0], batch_size):
        yield (X[i: i+batch_size, :], y[i: i+batch_size])
        


# In[ ]:


mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals) / std_val


# ### conv_layer
# - input_tensor
# - name (scope_name)
# - kernel_size
# - n_output_channels

# In[ ]:


import tensorflow as tf
import numpy as np

def conv_layer(input_tensor, name, kernel_size, n_output_channels,
              padding_mode="SAME", strides=(1, 1, 1,1)):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        #input_tensor = ( batch_size, width, hieght, channel)
        n_input_channels = input_shape[-1]
        weights_shape = (list(kernel_size) + [n_input_channels, n_output_channels])
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, name="net_pre-activation")
        conv = tf.nn.relu(conv, name="activation")
        print(conv)
        
        return conv
        


# In[ ]:


g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name="conv_layer", n_output_channels=32, kernel_size=(3, 3))

del g, x


# ### convlayer
# - input_tensor
# - name ( scope_name )
# - kernel_size
# - n_output_channels

# In[ ]:


def fc_layer(input_tensor, name, n_output_units, activation_fc=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        #np.prod >> return the product of array elements
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
        #if tensor is vector, reshape it from 1d to 2d
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name="_weights", shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name="_biases", initializer=tf.zeros(shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        if activation_fc is None:
            return layer
        layer = activation_fc(layer, name="activation")
        print(layer)
        return layer


# In[ ]:


g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    fc_layer(x, name="fc_test", n_output_units=32, activation_fc=tf.nn.relu)

del g, x


# In[ ]:


def print_build(n):
    print("\n Building {:d} layer:".format(n))


# In[ ]:


def build_cnn():
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
    tf_y = tf.placeholder(tf.int32 , shape=[None], name="tf_y")
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name="ts_x_reshaped")
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name="tf_y_onehot")
    print_build(1)
    h1 = conv_layer(tf_x_image, name="conv_1", kernel_size=(5, 5),
                    padding_mode="VALID", n_output_channels=32)
    #max pooling
    h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2,1],
                            strides=[1, 2, 2, 1], padding="SAME")
    print_build(2)
    h2 = conv_layer(h1, name="conv_2", kernel_size=(5, 5), 
                   padding_mode="VALID", n_output_channels=64)
    h2_pool = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding="SAME")
    print_build(3)
    h3  = fc_layer(h2, name="fc_3", n_output_units=1024, activation_fc=tf.nn.relu)
    #dropout regulation
    keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name="dropout_layer")
    #full connected layer 2
    print_build(4)
    h4 = fc_layer(h3_drop, name="fc_4", n_output_units=10, activation_fc=None)
    #prediction
    predictions = {
        "probabilities":  tf.nn.softmax(h4, name="probabilities"),
        "labels":  tf.cast(tf.argmax(h4, axis=1), tf.int32, name="labels")
    }
    #tensor board
    cross_entropy_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot),
    name="cross_entropy_loss")
    #opimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name="train_op")
    #prediction accuracy
    correct_predictions = tf.equal(predictions["labels"], tf_y, name="correct_preds")
    #tf.cast >> cast a tensor to new dtype
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


# In[ ]:


# g = tf.Graph()
# learning_rate = 1e-4
# with g.as_default():
#     build_cnn()
# with tf.Session(graph=g) as sess:
#     sess.run(tf.global_variables_initializer())
#     file_writer = tf.summary.FileWriter(logdir=".\logs\\", graph=g)
#     #command in working directory `tensorboard --logdir logs/`


# In[ ]:


def save(saver, sess, epoch, path=".\model\\"):
    if not os.path.isdir(path):
        os.makedirs(path)
        
    print("savining model in {}".format(path))
    saver.save(sess, os.path.join(path, "cnn-model.ckpt"), global_step=epoch)

def load(saver, sess, path, epoch):
    print("Loading model form {}".format(path))
    saver.restore(sess, os.path.join(path, "cnn-model.ckpt-{:d}".format(epoch)))
    
def train(sess, training_set, validation_set=None, initialize=True,
         epochs=20, shuffle=True, dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss=[]
    #initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())
        
    #shuffle by batch_generator
    np.random.seed(random_seed)
    
    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle, batch_size=1024)
        avg_loss = 0.0
#         print(X_data.shape[0] / 64)
        for i, (batch_x, batch_y) in enumerate(batch_gen):
#             if i < 50:
            feed = {"tf_x:0":batch_x, "tf_y:0":batch_y, "fc_keep_prob:0": dropout}
           # print("batch",i)
            loss, _ = sess.run(["cross_entropy_loss:0", "train_op"], feed_dict=feed)
            avg_loss += loss
#             else:
#                 break
        training_loss.append(avg_loss / (i + 1))
        print("Epoch : {:02d}, Training avg loss: {:.3f}".format(epoch, avg_loss), end=" ")
        if validation_set is not None:
            feed = {"tf_x:0": validation_set[0],
                   "tf_y:0": validation_set[1], 
                   "fc_keep_prob:0": 1.0}
            valid_acc = sess.run("accuracy:0", feed_dict=feed)
            print("Validation Acc: {:.3f}".format(valid_acc))
        else:
            print()

def predict(sess, X_test, return_proba=False):
    feed = {"tf_x:0":X_test, "fc_keep_prob": 1.0}
    if return_proba:
        return sess.run("probilities:0", feed_dict=feed)
    else:
        return sess.run("labels:0", feed_dict=feed)


# In[ ]:


learning_rate = 1e-4
random_seed = 123
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    build_cnn()
    saver = tf.train.Saver()


# In[ ]:


with tf.Session(graph=g) as sess:
    train(sess, training_set=(X_train_centered, y_train),
         validation_set=(X_valid_centered, y_valid),
         initialize=True, random_seed=123)
    save(saver, sess, epoch=20)


# In[ ]:


del g 


# In[ ]:


g2 = tf.Graph()
with g2.as_default():
    tf.set_random_seed(random_seed)
    build_cnn()
    saver = tf.train.Saver()
with tf.Session(graph=g2) as sess:
    load(saver,sess, epoch=20, path=".\model\\")
    preds = predict(sess, X_test_centered, retrun_proba=False)
    print("Test Accuracy:{:.3f}".foramt(100 * np.sum(preds == y_test) /len(y_test)))


# In[ ]:


np.set_printoptions(precision=2, suppress=True)

with tf.Session(graph=g2) as sess:
    load(saver, sess, epoch=20, path=".\model\\")
    print(predict(sess, X_test_centered[:10], return_proba=False))
    print(predict(sess, X_test_centered[:10], return_proba=True))


# In[ ]:


with tf.Session(graph=g2) as sess:
    load(saver, sess, epoch=20, path=".\model\\")
    
    train(sess, training_set=(X_train_centered, y_train),
          validation_set = (X_valid_centered, y_valid),
          initialize=False, epochs=20, random_seed=123)
    
    save(saver, sess, epoch=40, path=".\model\\")
    preds = predict(sess, X_test_centered, return_proba=False)
    print("Test accurcy:{:.3f}".format(100 * np.sum(preds == y_test)) / len(y_test))


# ## Layers API

# In[ ]:


import tensorflow as tf
import numpy as np

class ConvCNN(object):
    def __init__(self, batch_size=64, epochs=20, learning_rate=1e-4, dropout_rate=0.5,
                shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
        self.sess = tf.Session(graph=g)
        
    def build(self):
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
        tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")
        is_train = tf.placeholder(tf.bool, shape=(), name="is_train")
        
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name="tf_x_2dimage")
        #tensor = [batch_size, width, height, n_channels]
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, 
                                name="input_y_onehot")
        h1 = tf.layers.conv2d(tf_x_image, kernel_size=(5, 5), filters=32,
                             activation=tf.nn.relu)
        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2, 2),
                                       strides=(2, 2))
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(5, 5), filters=64, 
                             activation=tf.nn.relu)
        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(2, 2),
                                      strides=(2, 2))
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 1024, activation=tf.nn.relu)
        
        #drop regulation
        h3_drop = tf.layers.dropout(h3, rate=self.dropout_rate, training=is_train)
        
        h4 = tf.layers.dense(h3_drop, 10, activation=tf.nn.relu)
        
        predictions = {
            "probabilities": tf.nn.softmax(h4, name="probabilities"),
            "labels": tf.cast(tf.argmax(h4, axis=1), tf.int32, name="labels")
        }
        #cost function & optimizing
        cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=h4, labels=tf_y_onehot),
        name="cross_entropy_loss")
        
        #optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name="train_op")
        #correct rate
        correct_predictions = tf.equal(predictions["labels"], tf_y,
                                      name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                 name="accuracy")
        
    def save(self, epoch, path=".\\tflayers-molel\\"):
        if not os.path.isdir(path):
            os.mkdirs(path)
        print("Saving model in {}".format(path))
        self.saver.save(self.sess, op.path.join(path, "model.ckpt"),
                       global_step=epoch)
        
    def load(self, epoch, path):
        print("Loading model from {}".format(path))
        self.saver.restore(self.sess, os.path.join(path, "model.ckpt-{:d}".format(epoch)))
    
    def train(self, training_set, validation_set=None, initialize=True):
        if initialize:
            self.sess.run(self.init_op)
        
        self.train_cost = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        
        for epoch in range(1, self.epochs+1):
            batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {"tf_x:0": batch_x, "tf_y:0": batch_y,
                       "is_train:0": True}
                loss, _ = self.sess.run(["cross_entropy_loss:0", "train_op"],
                                       feed_dict=feed)
                avg_loss += loss
            
            print("Epoch {:2d}  Training_avg: {:.3f}".format(epoch, avg_loss), end=" ")
            if valid_set is not None:
                feed = {"tf_x:0": batch_x, "tf_y:0": batch_y,
                       "is_train:0": False}
                valid_acc = self.sess.run("accuracy:0", feed_dict=feed)
                print("Valid accuracy: {.3f}".format(valid_acc))
            else:
                print()
        
    def predict(self, X_test, return_proba=False):
        feed = {"tf_x:0": X_test, "is_train:0": False}
        if return_proba:
            return self.sess.run("probabilities:0", feed_dict=feed)
        else:
            return self.sess.run("labels:0", feed_dict=feed)
    


# In[ ]:


cnn =  ConvCNN(random_seed=123)
cnn.train(training_set = (X_train_centered, y_train),
         validation_set = (X_valid_centered, y_valid))
cnn.save(epoch=20)


# In[ ]:


del cnn


# In[ ]:


cnn2 = ConvCNN(random_seed=123)
cnn2.load(epoch=20, path =".\\tflayers-model\\")
print(cnn2.predict(X_test_centered[:10, :]))

          


# In[ ]:


preds = cnn2.predict(X_test_centered)
print("Test accuracy: {.2f}".format(100 * np.sum(y_test == preds) / len(y_test)))

