
# coding: utf-8

# In[ ]:


# Sentiment Analysis
# opinion Mining
# import tarfile
# with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
#     tar.extractall()
# 一度の実行で十分


# In[ ]:


import pyprind
import pandas as pd
import os
# basepath = "aclImdb"
# labels = {"pos":1, "neg":0}
# pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
# for s in ("test", "train"):
#      for l in ("pos", "neg"):
#          path = os.path.join(basepath, s, l)
#          for file in os.listdir(path):
#              with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
#                  txt = infile.read()
#              df = df.append([[txt, labels[l]]], ignore_index = True)
# df.columns = ["review", "sentiment"]


# In[ ]:


import numpy as np
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# #DataFrame の row をランダムにシャッフルする。
# df.to_csv("movie_data.csv", index=False, encoding="utf-8")
df = pd.read_csv("movie_data.csv", encoding="utf-8")
df.head(3)


# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    "The sun is shining", "The weather is sweet",
    "The sun is shining, the weather is sweet, and one and one is two"])
bag = count.fit_transform(docs)


# In[ ]:


print(count.vocabulary_)


# In[ ]:


print(bag.toarray())


# In[ ]:


df.loc[0, "review"][-50:]


# In[ ]:


import re 
def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text) 
    text = (re.sub("[\W]+"," ", text.lower()) + "".join(emoticons).replace("-", ""))
    return text


# In[ ]:


preprocessor(df.loc[0, "review"][-50:])


# In[ ]:


preprocessor("</a>This is :) is :(a test :-)!") 


# In[ ]:


df["review"] = df["review"].apply(preprocessor)


# In[ ]:


def tokenizer(text):
    return text.split()
tokenizer("runners like running and thus they run")


# In[ ]:


#単語を原型に変換する　>> nltk module
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter("runners like running and thus they run")


# In[ ]:


import nltk 
nltk.download("stopwords")


# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words("english")
[w for w in tokenizer_porter("runners like running and run a lot ")[-10:] if w not in stop]


# In[ ]:


X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{"vect__ngram_range":[(1, 1)], 
              "vect__stop_words":[stop, None],
              "vect__tokenizer":[tokenizer, tokenizer_porter],
              "clf__penalty":["l1", "l2"],
              "clf__C":[1.0, 10.0, 100.0]},
             
              {"vect__ngram_range":[(1, 1)],
              "vect__stop_words":[stop, None],
              "vect__tokenizer":[tokenizer, tokenizer_porter],
              "vect__use_idf":[False],
              "vect__norm":[None],
              "clf__penalty":["l1", "l2"],
              "clf__C":[1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([("vect", tfidf), 
                    ("clf", LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring="accuracy",
                          cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)


# In[ ]:


print("best parameter set: {}".format(gs_lr_tfidf.best_params_)


# In[ ]:


print("CV accuracy: {} ".format(gs_lr_tfidf.best_score_)
clf = gs_lr_tidf.best_estimator_
print("best_accuracy:{}".format(clf.score(X_test, y_test)))


# In[ ]:


import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words("english")
def tokenizer(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


# In[ ]:


def stream_docs(path):
    with open(path, "r", encoding="utf-8") as csv:
        next(csv)
        for line in csv:
              text, label = line[:-3], int(line[-2])
              yield text, label
    


# In[ ]:


next(stream_docs(path="movie_data.csv"))


# In[ ]:


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
            return None, None
    return docs, y


# In[ ]:


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error="ignore", n_features=2**21, 
                        preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss="log", random_state=1, max_iter=1)
doc_stream = stream_docs(path="movie_data.csv")


# In[ ]:


import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=100)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# In[ ]:


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print("accuracy: {}".format(clf.score(X_test, y_test)))


# In[ ]:


clf = clf.partial_fit(X_test, y_test)


# In[ ]:


import pandas as pd
df = pd.read_csv("movie_data.csv", encoding="utf-8")
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
X = count.fit_transform(df["review"].values)


# In[ ]:


#print(X[:10])


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components =10, random_state=123, learning_method="batch")
X_topics = lda.fit_transform(X)


# In[ ]:


lda.components_.shape


# In[ ]:


n_top_words = 5
feature_names = count.get_feature_names()
for topic_ind, topic in enumerate(lda.components_):
    print("topic {}".format(topic_ind + 1))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words:-1]]))


# In[ ]:


horror = X_topics[:, 5].argsort()[::-1]
for iter_ind, movie_ind in enumerate(horror[:3]):
    print("\nHorror moive #{}".format(iter_ind))
    print(df["review"][movie_ind][:300],"...")

