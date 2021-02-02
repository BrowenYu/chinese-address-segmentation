import os
import itertools
import pickle
import random
import collections
import numpy as np
import tensorflow as tf
from tokenizer import Tokenizer

window = 3 # 窗口大小
minlen = 2 # 句子最小长度
mintf = 10 # 最小词频
processes = 7 # 并行分词进程数

def preprocess(content):
    # 文章的预处理，这里暂不处理
    return content

def load_sentences(filepath, shuffle=True):
    readList = []
    with open(filepath, 'rb') as f:
        readList=pickle.load(f)
    samples = []
    for item in readList:
        samples.append("".join(item[0]))
    if shuffle:
        random.shuffle(samples)
    return samples

file = "./static_model/vocabs.json"
tokenizer = Tokenizer(mintf, processes)
if os.path.exists(file):
    # X = load_sentences('./train_all_1209.pkl')
    tokenizer.load_vocab_from_file(file)
    # tokenizer.load(file, X)
# else:
    # X = load_sentences('./train_all_1209.pkl')
    # print("tokenize...")
    # tokenizer.fit_in_parallel(X)
    # tokenizer.save(file)

words = tokenizer.words
word2id = tokenizer.word2id
id2word = {j:i for i,j in word2id.items()}
vocab_size = len(word2id)


def readDataFiles(path):
    readList = []
    with open(path, 'rb') as f:
        readList=pickle.load(f)
    cat_sample_20411 = []
    for item in readList:
        cat_sample_20411.append("".join(item[0]))
    return cat_sample_20411

if __name__ == "__main__":
    a, b = word2id, id2word
    c = 1
    # 测试
    # i = 0
    # for (a, b), c in iter(dl):
        # print(a.shape, b.shape, c.shape)
        # i+=1
        # print(i)
        # break
    # DataGenerator(epochs=10)
    # path = './train_all_1209.pkl'
    # temp = readDataFiles(path)
    # a = 1

