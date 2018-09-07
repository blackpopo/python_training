
# coding: utf-8

# In[ ]:


#Backpropagation McCulloch-Pitts neuron
#multilayer feedforward neural network
#hidden layer, bias unit, one-versus-all, 
import os 
import struct
import numpy as np
# def load_mnist(path, kind="train"):
#     labels_path = os.path.join(path, "{}-labels.idx1-ubyte".format(kind))
#     images_path = os.path.join(path, "{}-images.idx3-ubyte".format(kind))
#     #read files
#     with open(labels_path, "rb") as lbpath:
#         magic, n = struct.unpack(">II", lbpath.read(8))
#     #magic number, number of items
#         labels = np.fromfile(lbpath, dtype=np.uint8)
    
#     with open(images_path, "rb") as imgpath:
#         magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
#         images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
#         images = ((images / 255.0) - 0.5) * 2
    
#     return images, labels

# #magic, n  = struct.unpack(">II", lbpath.read(8))
# #magic number >> file protocol 
# # >II >> edian and no sign integral　(read from head)


# In[ ]:


# X_train, y_train = load_mnist("", kind="train")
# print("Rows: {}, Columns: {}".format(X_train.shape[0], X_train.shape[1]))


# In[ ]:


# X_test, y_test = load_mnist("", kind="t10k")
# print("Rows: {}, Columns: {}".format(X_test.shape[0], X_test.shape[1]))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap="Greys")
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()


# In[ ]:


# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(25):
#     #print(X_train[y_train == 7][i])
#     img = X_train[y_train == 7][i].reshape(28,28)
#     ax[i].imshow(img, cmap="Greys")

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()


# In[ ]:


# import numpy as np
# np.savez_compressed("mnist_scaled.npz", X_train=X_train,
#                    y_train=y_train, X_test=X_test, y_test=y_test)


# In[ ]:


mnist = np.load("mnist_scaled.npz")


# In[ ]:


mnist.files


# In[ ]:


X_train = mnist["X_train"]


# In[ ]:


X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]


# In[ ]:


#from neuralnet import NeuralNetMLP


# In[ ]:


class NeuralNetMLP(object):
    #n_hidden >> number of hidden layers
    #l2 >> l2 regulariaion paramator
    #shuffle >> shuffle every epoch or not
    #eval >> evalation dictionary
    #epochs >> number of training
    #eta >> 7th Greek alphabet
    def __init__(self, n_hidden=30, l2=0.0, epochs=100, eta=0.01,
                shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        
    def _onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for ind, val in enumerate(y.astype(int)):
            onehot[val, ind] = 1
        return onehot.T
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    #np.clip(val, min, max)
    
    def _forward(self, X):
        z_h = np.dot(X, self.w_h) + self.b_h
        #total hidden_layer
        a_h = self._sigmoid(z_h)
        #activate hidden layer
        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out
    
    def _compute_cost(self, y_enc, output):
        #y_enc >> encoded y by onehot encoder
        #output activeted outlayer
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.0) +
                             np.sum(self.w_out ** 2.0)))
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 -term2) + L2_term
        return cost
    
    def predict(self, X):
        #prediction
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        #sample features for validation << X_valid, y_valid
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]
    #initialize weights
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = np.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b_out = np.zeros(n_output)
        self.w_out = np.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {"cost": [], "train_accuracy": [], "valid_accuracy": []}
        #dictionary for evaluation
        y_train_enc = self._onehot(y_train, n_output)
        #training in epoch number times
        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            #indices >> plural index (not indexes)
            if self.shuffle:
                self.random.shuffle(indices)
            for start_ind in range(0,
                                  indices.shape[0] - self.minibatch_size+1,
                                  self.minibatch_size):
                batch_ind = indices[start_ind: start_ind + self.minibatch_size]
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_ind])
                #backpropagation
                sigma_out = a_out - y_train_enc[batch_ind]
                sigmoid_derivation_h = a_h * (1.0 - a_h)
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                          sigmoid_derivation_h)
                grad_w_h = np.dot(X_train[batch_ind].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)
                #regularization and update weights
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            #evaluation in every iteration
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])
            import sys 
            sys.stderr.write("  {:strlen}/{} | Cost: {:2f} | Train/Valid acc. {:2f} / {:2f}".format(i+1,
                                                            self.epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()
            
            self.eval_["cost"].append(cost)
            self.eval_["train_accuracy"].append(train_acc)
            self.eval_["valid_accuracy"].append(valid_acc)
            
        return self


# In[ ]:


n_epochs = 200
nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=n_epochs,
                      eta=0.0005, minibatch_size=100, shuffle=True, seed=1)


# In[ ]:


nn.fit(X_train=X_train[:55000], y_train=y_train[:55000],
      X_valid=X_train[55000:], y_valid=y_train[55000:])


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_["cost"])
plt.xlabel("Cost")
plt.ylabel("Epochs")
plt.show()


# In[ ]:


plt.plot(range(nn.epochs), nn.eval_["train_accuracy"], label="training")
plt.plot(range(nn.epochs), nn.eval_["valid_accuracy"], label="validation",
        linestyle="--")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()


# In[ ]:


y_test_pred = nn.predict(X_test)
#score of model accuracy in test data set
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
print("training accuracy: {:2f}".format(acc * 100))


# In[ ]:


miscal_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscal_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscal_img[i].reshape(28,28)
    ax[i].imshow(img, cmap="Greys", interpolation="nearest")
    ax[i].set_title("{:d} ) t:{:d} p:{:d}".format(i+1, correct_lab[i], miscal_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

