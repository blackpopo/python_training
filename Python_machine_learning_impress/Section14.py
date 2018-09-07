
# coding: utf-8

# Features of Tensoer flow
# 
# 1. _computeation praph_
# 
# 1. _rank_
# 
# 1. _tensor_

# In[ ]:


import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])
    
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    #get rank & shape
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print("Shapes:",s1, s2, s3)
    
with tf.Session(graph=g) as sess:
    print("Ranks:",r1.eval(), r2.eval(), r3.eval())
    


# # How to construct Graph & compile it
# 
# 2. make an instance of empty calcuration graph
# 2. append node to graph
# 2. execute graph
#     2. start new session
#     2. initialize variables
#     2. execute graph in this session

# In[ ]:


g = tf.Graph()
with g.as_default():
    a = tf.constant(1, name="a")
    b = tf.constant(2, name="b")
    c = tf.constant(3, name="c")
    z = 2 * (a - b) + c
    


# In[ ]:


#Execute Graph
with tf.Session(graph=g) as sess:
    print("2*(a-b)+c =",sess.run(z))


# In[ ]:


#feeding data >> placeholder 
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[], name="tf_a")
    tf_b = tf.placeholder(tf.int32, shape=[], name="tf_b")
    tf_c = tf.placeholder(tf.int32, shape=[], name="tf_c")
    r_1 = tf_a - tf_b
    r_2 = 2 * r_1
    z = r_2 + tf_c
    
#execute this graph
with tf.Session(graph=g) as sess:
    feed = {tf_a: 1, tf_b: 2, tf_c: 3}
    print("z:", sess.run(z, feed_dict=feed))


# In[ ]:


with tf.Session(graph=g) as sess:
    feed = {tf_a: 1, tf_b: 2}
    print("r1:", sess.run(r_1, feed_dict=feed))
    print("r2:", sess.run(r_2, feed_dict=feed))
    feed = {tf_c: 3, tf_a: 1, tf_b: 2}
    print("r1:", sess.run(r_1, feed_dict=feed))
    print("r2:", sess.run(r_2, feed_dict=feed))
    


# ## variables of Tensorflow
# - define in calicuration graph
# - initialize in session
# - use "variable scope
# - reuse existing variables

# ## define variables
# * tf.Variable(<\init> , name="<\variable>")
# * tf.get_variable("<\variable>", ...)
# > tf.get_variable is better than tf.Variable

# In[ ]:


import tensorflow as tf
import numpy as np;
g1 = tf.Graph();
with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4],[5, 6, 7, 8]]), name="w")
    print(w)


# In[ ]:


with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    


# In[ ]:


import tensorflow as tf
g2 = tf.Graph()

with g2.as_default():
    w1 = tf.Variable(1, name="w1")
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(2, name="w2")


# In[ ]:


with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print("w1:", sess.run(w1))
    


# In[ ]:


with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print("w2", sess.run(w2))


# ## variable scope

# In[ ]:


import tensorflow as tf

g = tf.Graph()

with g.as_default():
    with tf.variable_scope("net_A"):
        with tf.variable_scope("layer_1"):
            w1 = tf.Variable(tf.random_normal(shape=(10, 4)), name="weights")
        with tf.variable_scope("layer_2"):
            w2 = tf.Variable(tf.random_normal(shape=(20, 20)), name="weights")
    with tf.variable_scope("net_B"):
        with tf.variable_scope("layer_1"):
            w3 = tf.Variable(tf.random_normal(shape=(10, 4)), name="wights")
print(w1)
print(w2)
print(w3)


# In[ ]:


import tensorflow as tf
#helper function
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name="weights", shape=(data_shape[1], n_classes),
                             dtype=tf.float32)
    bias = tf.get_variable(name="bias", initializer=tf.zeros(shape=n_classes))
    logits = tf.add(tf.matmul(data, weights), bias, name="logits")
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)), name="w1")
    b1 = tf.Variable(tf.zeros(shape=n_hidden), name="b1")
    hidden = tf.add(tf.matmul(data, w1), b1, name="hidden_pre-activation")
    hidden = tf.nn.relu(hidden, name="hidden_activation")
    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])), name="w2")
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]), name="b2")
    output = tf.add(tf.matmul(hidden, w2), b2, name="output")
    return output, tf.nn.sigmoid(output)


# In[ ]:


#Construct Cal Graph
batch_size=64
g = tf.Graph()
with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="tf_X")
    #Construct generator
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)

    #Construct classifier
    with tf.variable_scope("classifier") as scope:
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
        #reuse generator
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size))
        init_op = tf.global_variables_initializer()


# In[ ]:


g = tf.Graph()
with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="tf_X")
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
    with tf.variable_scope("classifier"):
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(batch_size))
    with tf.variable_scope("classifier", reuse=True):
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size) )
        init_op =tf.global_variables_initializer()


# ## Construct Regressionmodel
# - input **x**   placeholder is tf_X
# - input **y**   placeholder is tf_y
# - modelparamator **w**     variable weight
# - modelparamator **b**     variable bias
# - modeloutput  $\hat{y}$    output y_hat 

# In[ ]:


import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(123)
    tf_X = tf.placeholder(shape=(None), dtype=tf.float32, name="tf_X")
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name="tf_y")
    
    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25), name="weight")
    bias = tf.Variable(0.0, name="bias")
    
    y_hat = tf.add(weight * tf_X, bias, name="y_hat")
    
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name="cost")
    
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name="train_op")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None)
        y.append(r)
    return x, 1.726 * x -0.84 + np.array(y)

x, y = make_random_data()
plt.plot(x, y,"o")
#marker="o" => lines are added
plt.show()


# In[ ]:


x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[:100]

n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    #model training
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], feed_dict={tf_X: x_train, tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print("Epoch {:4d}: {:.4f}".format(e, c))


# In[ ]:


plt.plot(training_costs)
plt.show()


# In[ ]:


#execute objects with its name
n_epochs = 500
training_costs = []
#session start
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs):
        c, _ = sess.run(["cost:0", "train_op"], feed_dict={"tf_X:0": x_train, "tf_y:0": y_train})
        training_costs.append(c)
        if not e % 50:
            print("Epochs {:4d}: {:.4f}".format(e, c))


# In[ ]:


#Saving models >> apeend new node called saver
with g.as_default():
    saver = tf.train.Saver()

n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], feed_dict={tf_X: x_train, tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print("Epoch: {:4d}: {:.4f}".format(e, c))
    
    saver.save(sess, "./trained-model")


# In[ ]:


# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph("./trained-model.meta")


# In[ ]:


import tensorflow as tf
import numpy as np

g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver =  tf.train.import_meta_graph("./trained-model.meta")
    new_saver.restore(sess, "./trained-model")
    pred = sess.run("y_hat:0", feed_dict={"tf_X:0": x_test})


# In[ ]:


import matplotlib.pyplot as plt
x_arr = np.arange(-2, 4, 0.1)
g2 = tf.Graph()

with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph("./trained-model.meta")
    new_saver.restore(sess, "./trained-model")
    y_arr = sess.run("y_hat:0", feed_dict={"tf_X:0": x_arr})

plt.figure()
plt.plot(x_train, y_train, "bo")
plt.plot(x_test, y_test, "bo", alpha=0.3)
plt.plot(x_arr, y_arr.T[:, 0], "-r", lw=3)
plt.show()
    


# In[ ]:


#Tensor operator
import tensorflow as tf
import numpy as np
g = tf.Graph()

with g.as_default():
    arr = np.array([[1.0, 2.0, 3.0, 3.5],
                    [4.0, 5.0, 6.0, 6.5],
                    [7.0, 8.0, 9.0, 9.5]])
    T1 = tf.constant(arr, name="T1")
    print(T1)
    s = T1.get_shape()
    print("Shape of T1:", s)
    T2 = tf.Variable(tf.random_normal(shape=s))
    print(T2)
    T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0], )))
    print(T3)


# In[ ]:


with g.as_default():
    T4 = tf.reshape(T1, shape= [1,1, -1], name="T4")
    print(T4)
    T5 = tf.reshape(T1, shape=[1, 3, -1], name="T5")
    print(T5)


# In[ ]:


with tf.Session(graph=g) as sess:
    print(sess.run(T4))
    print()
    print(sess.run(T5))


# In[ ]:


with g.as_default():
    T6 = tf.transpose(T5, perm=[2, 1, 0], name="T6")
    print(T6)
    T7 = tf.transpose(T5, perm=[0, 2, 1], name="T7")
    print(T7)


# In[ ]:


with g.as_default():
    t5_split = tf.split(T5, num_or_size_splits=2, axis=2, name="T8")
    print(t5_split)
    t5_split1 = tf.split(T5, num_or_size_splits=4, axis=2, name="T8")
    print(t5_split1)


# In[ ]:


g = tf.Graph()
with g.as_default():
    t1 = tf.ones(shape=(5, 1), dtype=tf.float32, name="t1")
    t2 = tf.zeros(shape=(5, 1), dtype=tf.float32, name="t2")
    print(t1)
    print(t2)

with g.as_default():
    t3 = tf.concat([t1, t2], axis=0, name="t3")
    print(t3)
    t4 = tf.concat([t1, t2], axis=1, name="t4" )
    print(t4)


# In[ ]:


with tf.Session(graph=g) as sess:
    print(t3.eval())
    print()
    print(t4.eval())
#eval()  >> evaluation


# ### use controlflow in constructing Graph

# In[ ]:


import tensorflow as tf
x, y = 1.0, 2.0
g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name="tf_y")
    if x < y:
        res = tf.add(tf_x, tf_y, name="result_add")
    else:
        res = tf.subtracit(tf_x, tf_y, name="rusult_sub")
    print("Object:", res)

with tf.Session(graph=g) as sess:
    print("x < y :{} -> result:{}".format((x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y})))
    x, y = 2.0, 1.0
    print("x < y :{} -> result:{}".format((x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y})))


# In[ ]:


import tensorflow as tf
x, y= 1.0, 2.0
g = tf.Graph()

with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name="tf_y")
    res = tf.cond(tf_x < tf_y,
                 lambda: tf.add(tf_x, tf_y, name="result_add"),
                 lambda: tf.subtract(tf_x, tf_y, name="result_sub"))
    print("Object:", res)
    
with tf.Session(graph=g) as sess:
    print("x < y: {} -> result: {}".format((x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y})))
    x, y = 2.0, 1.0
    print("x < y: {} -> result: {}".format((x < y), res.eval(feed_dict={"tf_x:0": x, "tf_y:0": y})))


# In[ ]:


if (x < y):
    result = 1
else:
    result = 0


# In[ ]:


f1 = lambda: tf.constant(1)
f2 = lambda: tf.constant(0)
result = tf.case([(tf.less(x, y), f1)], default=f2)


# In[ ]:


i = tf.constant(0)
threshold = 100
c = lambda i: tf.less(i, 100)
b = lambda i: tf.add(i, 1)
t = tf.while_loop(cond=c, body=b, loop_vars=[i])


# ### visualize caliculation graph with TensorBoard

# In[ ]:


batch_size = 64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="tf_X")
    #construct generator
    with tf.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
        
    with tf.variable_scope("classifier"):
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size))


# In[ ]:


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(logdir=".\logs\\", graph=g)


# ## use this command in working directory
# `tensorboard --logdir logs\`
