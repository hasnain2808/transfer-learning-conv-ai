from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import pickle

from keras.models import load_model

def loadData(filename):
    file_ptr = open(filename, 'rb')
    loaded_obj = pickle.load(file_ptr)
    return loaded_obj

embeddings_index  = loadData("classifier_model/glove.840B.300d.pkl")

# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

# # train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
# val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
# val_y = np.array(val_df["target"][:3000])


model = load_model('classifier_model/my_modelcpu.h5')
zz = ['I like to sleep',"that's cool other cultures are nice", "where is Geneva cats?", "What public figure defended New York in Januar"]
valDF = pd.DataFrame()
valDF['question_text'] = zz

# prediction part
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 0]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

# test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
test_df = valDF

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())

y_te = (np.array(all_preds) > 0.5).astype(np.int)
print(y_te)
print(valDF['question_text'])