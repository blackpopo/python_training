
# coding: utf-8

# # Recurrent Neural Network
# ### Sequence >> not IID(Independent and Identically Distributed)
# ### Backpropagation Through Time >> One of RNN model
# #### problems in BPTT
# - vanishing gradient $|w_{hh}| < 1$
# - exploding gradient $|w_{hh}| > 1$
# #### Solutions in BPTT
# - T-BPTT(Truncated Backpropagation Through Time)
# - LSTM (Long and Short Term Memory)

# ### UNITS in LSTM
# - memory cell >> that has cell state and three gates
#     - forget gate >> only sigmoid
#     - input gate (& input node ) >> tanh and sigmoid
#     - output gate >> tanh and sigmoid

# In[ ]:


import pyprind 
import pandas as pd
from string import punctuation
import re
import numpy as np
df = pd.read_csv("movie_data.csv",encoding="utf-8")


# In[ ]:


#Don't use "set" in finding unique word, instead use Counter 
from collections import Counter
counts = Counter()
pbar = pyprind.ProgBar(len(df["review"]), title="counting words occurrences")
for i, review in enumerate(df["review"]):
    text = "".join([c if c not in punctuation else " "+c+" " for c in review]).lower()
    df.loc[i, "review"] = text
    pbar.update()
    counts.update(text.split())
#pyprind.ProgBar >> tracking progress


# In[ ]:


word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:5])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}
#enumerate(iter, start, end)
mapped_reviews = []
pber = pyprind.ProgBar(len(df["review"]), title="map reviews to ints")
for review in df["review"]:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()


# In[ ]:


sequence_length = 200
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)
for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]
    


# In[ ]:


X_train = sequences[:25000, :]
y_train = df.loc[:25000, "sentiment"].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, "sentiment"].values


# In[ ]:


#helper function >> mini batch
np.random.seed(123)

def create_batch_generator(x, y=None, batch_size=64):
    n_batches = len(x) // batch_size
    x = x[:n_batches * batch_size]
    if y is not None:
        y = y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        if y is not None:
            yield x[ii: ii+batch_size], y[ii: ii+batch_size]
        else:
            yield x[ii: ii+batch_size]


# In[ ]:


# n_words = len(word_counts)
# embedding_size = 200
# embedding = tf.Variable(
# tf.random_uniform(shape=(n_words, embedding_size), minval=-1, maxval=1))
# #call tf.nn.embedding_lookup function
# embed_x = tf.nn.embedding_lookup(embedding, tf_x)


# ## Constructiong RNN model
# - constructor
# - build method >> input_data, input_label, keep_dropout
# - train method >> training
# - predict methd >> new seesion and prediction new_data

# In[ ]:


import tensorflow as tf

class SentimentRNN(object):
    def __init__(self, n_words, seq_length=200, lstm_size=256, num_layers=1, 
                batch_size=64, learning_rate=0.0001, embed_size=200):
        self.n_words = n_words
        self.seq_length = seq_length
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_size = embed_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            

    def build(self):
        tf_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name="tf_x")
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size), name="tf_y")
        tf_keep_prob = tf.placeholder(tf.float32, name="tf_keep_prob")
        embedding = tf.Variable(tf.random_uniform((self.n_words, self.embed_size),
                                                 minval=-1, maxval=1), name="embedding")
        embed_x = tf.nn.embedding_lookup(embedding, tf_x, name="embed_x")
        #lstm cells
        cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
        output_keep_prob = tf_keep_prob)
        for _ in range(self.num_layers)])

        #Define initial State
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        print(" << initial State >> ", self.initial_state)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, embed_x, 
                                                           initial_state = self.initial_state)
        #lstm_outputs = [batch_size, max_time, cells.output_size]
        print("\n << lstm output >> ", lstm_outputs)
        print("\n << final state >> ", self.final_state)

        #full connected layer
        logits = tf.layers.dense(inputs=lstm_outputs[: ,-1],
                                units=1, activation=None, name="logits")
        logits = tf.squeeze(logits, name="logits_squeezed")
        #tf.squeeze >> Removes dimensions of size 1 from the shape of a tensor
        print("\n << logits      >> ", logits)
        y_proba = tf.nn.sigmoid(logits, name="probabilities")
        predictions = {
            "probabilities": y_proba,
            "labels": tf.cast(tf.round(y_proba), tf.int32, name="labels")
        }
        print("\n << predictions >> ", predictions)
        cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name="cost")

        #Define Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name="train_op")

    def train(self, X_train, y_train, num_epochs):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)
            iteration = 1
            for epoch in range(num_epochs):
                state = sess.run(self.initial_state)

                if epoch == 0:
                    self.saver.save(sess, "model/sentiment-{:d}.ckpt".format(epoch))
                    
                for batch_x, batch_y in create_batch_generator(
                                            X_train, y_train, self.batch_size):
                    feed = {"tf_x:0": batch_x, "tf_y:0": batch_y,
                           "tf_keep_prob:0": 0.5, self.initial_state: state}
                    loss, _, state = sess.run([
                        "cost:0", "train_op", self.final_state
                    ], feed_dict = feed)

                    if iteration % 20 == 0:
                        print("Epoch: {:d}/{:d}  Iteration: {:d} | Training loss: {:.5f}".format(
                        epoch + 1, num_epochs, iteration, loss))
                    iteration +=1
                    
                if (epoch + 1) % 10 == 0:
                    self.saver.save(sess, "model/sentiment-{:d}.ckpt".format(epoch))

    def predict(self, X_data, return_proba=False):
        preds = []
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(".\model\\"))
        test_state = sess.run(self.initial_state)
        for ii, batch_x in enumerate(create_batch_generator(
                        X_data, None, batch_size=self.batch_size), 1):
            feed = {"tf_x:0": batch_x, "tf_keep_prob:0": 1.0,
                   self.initial_state: test_state}
            if return_proba:
                pred, test_state = sess.run(["probabilities:0", self.final_state],
                                           feed_dict=feed)
            else:
                pred, test_state = sess.run(["labels:0", self.final_state],
                                           feed_dict=feed)

            preds.append(pred)
        return np.concatenate(preds)


# In[ ]:


n_words = max(list(word_to_int.values())) + 1

rnn = SentimentRNN(n_words=n_words, seq_length=sequence_length, embed_size=256,
                  lstm_size=128, num_layers=1, batch_size=100, learning_rate=0.001)


# In[ ]:


rnn.train(X_train, y_train, num_epochs=40)


# In[ ]:


preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print("Test Accurcy: {:.3f}".format(np.sum(preds == y_true) / len(y_true)))


# In[ ]:


proba = rnn.predict(X_test, return_proba=True)


# In[ ]:


#Language model of RNN

