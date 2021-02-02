import json
# from https://github.com/allenwind/count-in-parallel
from parallel import count_in_parallel_from_generator

def get_chars_count(vocabs, X):
    word2id = []
    words = []
    return word2id, words

class Tokenizer:
    """支持并行的Tokenizer"""

    def __init__(self, mintf=16, processes=7):
        self.word2id = {}
        self.words = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.mintf = mintf
        self.processes = processes
        self.filters = set("!\"#$%&'()[]*+,-./，。！@·……（）【】<>《》?？；‘’“”")

    # def fit_in_parallel(self, X):
    #     words = count_in_parallel_from_generator(
    #         tokenize=tokenize,
    #         generator=X,
    #         processes=self.processes
    #     )

    #     # 过滤低频词和特殊符号
    #     words = {i: j for i, j in words.items() \
    #              if j >= self.mintf and i not in self.filters}
    #     # 0:MASK
    #     # 1:UNK
    #     # 建立字词ID映射表
    #     for i, c in enumerate(words, start=2):
    #         self.word2id[c] = i
    #     self.words = words

    def __len__(self):
        return len(self.word2id) + 2

    def transform(self, X):
        wids = []
        for sentence in X:
            wid = self.transform_one(sentence)
            wids.append(wid)
        return wids

    def transform_one(self, sentence):
        wid = []
        # for word in jieba.lcut(sentence, HMM=False):
        #     w = self.word2id.get(word, self.UNKNOW)
        #     wid.append(w)
        a = 1 / 0
        return wid

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                [self.word2id, self.words],
                fp,
                indent=4,
                ensure_ascii=False
            )

    def load(self, file, X):
        with open(file, "r") as fp:
            vocab = json.load(fp)
            word_vocab = vocab['word_vocab']['idx_to_token']
            # word_vocab = word_vocab[2:]
            # self.words = word_vocab[2:]
            for i, w in enumerate(word_vocab):
                self.word2id[w] = i
        for sentence in X:
            for c in sentence:
                if c in word_vocab:
                    if c not in self.words:
                        self.words[c] = 1
                    else:
                        self.words[c] += 1
                else:
                    if word_vocab[1] not in self.words:
                        self.words[word_vocab[1]] = 1
                    else:
                        self.words[word_vocab[1]] += 1
        vocab['word_vocab']['words_num'] = self.words
        with open(file, "w") as fp:
            json.dump(vocab, fp)

    def load_vocab_from_file(self, file):
        with open(file, "r") as fp:
            vocab = json.load(fp)
            word_vocab = vocab['word_vocab']['idx_to_token']
            # word_vocab = word_vocab[2:]
            # self.words = word_vocab[2:]
            for i, w in enumerate(word_vocab):
                self.word2id[w] = i
            self.words = vocab['word_vocab']['words_num']