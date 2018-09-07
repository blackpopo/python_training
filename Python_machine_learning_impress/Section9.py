
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv("movie_data.csv", encoding="utf-8")
df.loc[0, "review"][-50:]


# In[ ]:


import re
def preprocessor(text):
    text = re.sub("<[^.]>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = (re.sub("[\W]+", " ", text.lower()) + "".join(emoticons).replace("-", ""))
    return text
df["review"] = df["review"].apply(preprocessor)
df.loc[0, "sentiment"]


# In[ ]:


X_train = df.loc[:25000, "review"].values
y_train = df.loc[:25000, "sentiment"].values
X_test = df.loc[25000:, "review"].values
y_test = df.loc[25000:, "sentiment"].values


# In[ ]:


print(X_train[0], y_train[0])


# In[ ]:


# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
def tokenizer(text):
    return text.split()
# tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
#                        ngram_range=(1, 1), stop_words=None, tokenizer=tokenizer) 
# pipe_lr  = make_pipeline(tfidf, LogisticRegression(random_state=0, penalty="l2", C=10.0))
# pipe_lr.fit(X_train, y_train)


# In[ ]:


def stream_docs(path):
    with open(path, "r", encoding="utf-8") as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


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
vect = HashingVectorizer(decode_error="igonre", n_features=2**21, preprocessor=None,tokenizer=tokenizer)
clf = SGDClassifier(loss="log", random_state=1, n_iter=1)
doc_stream = stream_docs(path="movie_data.csv")


# In[ ]:


import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# In[ ]:


# X_test = tfidf.fit_transform(X_test)
# clf.score(X_test, y_test)


# In[ ]:


import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words("english")


# In[ ]:


#Let's make an application!!
#model persistence >> シリアライズ
import pickle
import os
desti = os.path.join("movieclassifier", "pkl_objects")
if not os.path.exists(desti):
    os.makedirs(desti)
    #protocol >> python 3.x x=4
pickle.dump(stop, open(os.path.join(desti, "stopwords.pkl"), "wb"), protocol=4)
#protocol >> python 3.x 以上のもので互換性を持たせる。この場合はx=4。
pickle.dump(clf, open(os.path.join(desti, "classifier.pkl"), "wb"), protocol=4)


# In[ ]:


#vectorizer.py として movieclassifier directory に保存する。
# from sklearn.feature_extraction.text import HashingVectorizer
# import re
# import os
# import pickle
 
# current_dir = os.path.dirname(__file__)
# stop = pickle.load(open(os.path.join(current_dir,"movieclassifier", "pkl_objects", "stopwords.pkl"), "rb"))
# def tokenizer(text):
#     text = re.sub("<[^>]*>", "", text)
#     emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
#     text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", " ")
#     tokenized = [w for w in text.split() if w not in stop]
#     return tokenized

# vect = HashingVectorizer(decode_error="igonore", n_features=2**21, preprocessor=None, tokenizer=tokenizer)


# In[ ]:


import pickle 
import re
import os 
from vectorizer import vect
clf = pickle.load(open(os.path.join("movieclassifier", "pkl_objects", "classifier.pkl"), "rb"))


# In[ ]:


import numpy as np
label = {0:"negative", 1:"positive"}
example = ["I love you forever"]
X = vect.transform(example)
print("Prediction: {} \nProbability {}".format(label[clf.predict(X)[0]], np.max(clf.predict_proba(X)*100)))


# In[ ]:


import sqlite3
import os
if os.path.exists("reviews.sqlite"):
    os.remove("reviews.sqlite")
conn = sqlite3.connect
c = conn.cursor()
c.execute("CREATE TABLLE review_db (review TEXT, sentiment INTEGER, data TEXT)")
example1 = "I love the movie very much"
c.execute("INSERT INTO rebiew_db (review, sentiment, data) VALUES"
         "(?, ?, DATETIME('now'))", (example1, 1))

