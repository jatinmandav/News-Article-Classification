import pandas as pd
from gensim.models import FastText
import nltk
import numpy as np
import random
from tqdm import tqdm

class ReadData:
    def __init__(self, path_file, embedding_config, train_val_split=0.2, data_shape=(400, 256), sentence_pair=False):
        if path_file != None:
            self.path_file = path_file
            self.train_val_split = train_val_split
            self.read_file()

        self.embedding_config = embedding_config
        self.load_embedding_model()

        if sentence_pair:
            self.max_len = data_shape[0][0]
        else:
            self.max_len = data_shape[0]

        self.sentence_pair = sentence_pair
        self.label_map = {0: 'politics', 1: 'technology', 2:'entertainment', 3: 'business'}

    def read_file(self):
        self.data = pd.read_excel(self.path_file, sheet_name=None)['Sheet1']

        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data.columns = ['context', 'label']

        self.train_data = self.data.head(int(len(self.data)*(1-self.train_val_split)))
        self.train_x = list(self.train_data.context)
        self.train_y = list(self.train_data.label)

        self.val_data = self.data.tail(int(len(self.data)*(self.train_val_split)))
        self.val_x = list(self.val_data.context)
        self.val_y = list(self.val_data.label)

        self.train_size = len(self.train_data)
        self.val_size = len(self.val_data)

        print('Training Size: {}, Validation Size: {}'.format(self.train_size, self.val_size))

    def load_embedding_model(self):
        if self.embedding_config['type'] == 'fasttext':
            self.embedding = FastText.load(self.embedding_config['path'])

        print('Embedding Loaded')

    def sent2vec(self, sent, max_len=400, dim=256):
        vector = []
        for i, word in enumerate(sent):
            if i < self.max_len:
                vec = self.embedding[str(word)]
                vector.append(vec)

        vectors = []
        vectors += list(vector)
        padding_len = max_len - len(vectors)
        for _ in range(padding_len):
            vectors.append(np.zeros(dim))

        return np.array(vectors)

    def get_next_batch(self, start_index, batch_size=64):
        batch_x, batch_x2, batch_y = [], [], []
        for i, sent in enumerate(self.train_x[start_index:start_index+batch_size]):
            tokenized = nltk.word_tokenize(sent.lower())
            x = self.sent2vec(tokenized)

            if self.sentence_pair:
                for index in range(4):
                    batch_x.append(x)
                    batch_x2.append(self.embedding[self.label_map[index]])

                    if index == int(self.train_y[start_index+i]):
                        batch_y.append(1.)
                    else:
                        batch_y.append(0.)
            else:
                batch_x.append(x)
                one_hot = np.zeros(4)
                one_hot[int(self.train_y[start_index+i])] = 1.
                batch_y.append(one_hot)

        if self.sentence_pair:
            #x, y = np.hstack([batch_x, batch_x2]), np.array(batch_y)
            x, y = [np.array(batch_x), np.array(batch_x2)], np.array(batch_y)
        else:
            x, y = np.array(batch_x), np.array(batch_y)

        return x, y

    def get_next_val_batch(self, start_index, batch_size=64):
        batch_x, batch_x2, batch_y = [], [], []
        for i, sent in enumerate(self.val_x[start_index:start_index+batch_size]):
            tokenized = nltk.word_tokenize(sent.lower())
            x = self.sent2vec(tokenized)

            if self.sentence_pair:
                for index in range(4):
                    batch_x.append(x)
                    batch_x2.append(self.embedding[self.label_map[index]])

                    if index == int(self.val_y[start_index+i]):
                        batch_y.append(1.)
                    else:
                        batch_y.append(0.)
            else:
                batch_x.append(x)
                one_hot = np.zeros(4)
                one_hot[int(self.val_y[start_index+i])] = 1.
                batch_y.append(one_hot)

        if self.sentence_pair:
            #x, y = np.hstack([batch_x, batch_x2]), np.array(batch_y)
            x, y = [np.array(batch_x), np.array(batch_x2)], np.array(batch_y)
        else:
            x, y = np.array(batch_x), np.array(batch_y)

        return x, y

    def generator(self, batch_size=64):
        while True:
            no_batches = int(self.train_size/batch_size)
            for i in range(no_batches):
                start_index = i*batch_size
                batch_x, batch_x2, batch_y = [], [], []
                for i, sent in enumerate(self.train_x[start_index:start_index+batch_size]):
                    tokenized = nltk.word_tokenize(sent.lower())
                    x = self.sent2vec(tokenized)

                    if self.sentence_pair:
                        for index in range(4):
                            batch_x.append(x)
                            batch_x2.append(self.embedding[self.label_map[index]])

                            if index == int(self.train_y[start_index+i]):
                                batch_y.append(1.)
                            else:
                                batch_y.append(0.)
                    else:
                        batch_x.append(x)
                        one_hot = np.zeros(4)
                        one_hot[int(self.train_y[start_index+i])] = 1.
                        batch_y.append(one_hot)

                if self.sentence_pair:
                    x, y = [np.array(batch_x), np.array(batch_x2)], np.array(batch_y)
                else:
                    x, y = np.array(batch_x), np.array(batch_y)

                yield x, y

    def read_val(self):
        val_x, val_x2, val_y = [], [], []
        i = 0
        for sent in tqdm(self.val_x):
            tokenized = nltk.word_tokenize(sent.lower())
            x = self.sent2vec(tokenized)

            if self.sentence_pair:
                for index in range(4):
                    val_x.append(x)
                    val_x2.append(self.embedding[self.label_map[index]])

                    if index == int(self.val_y[i]):
                        val_y.append(1.)
                    else:
                        val_y.append(0.)
            else:
                val_x.append(x)
                one_hot = np.zeros(4)
                one_hot[int(self.val_y[i])] = 1.
                val_y.append(one_hot)

            i += 1

        if self.sentence_pair:
            x, y = [np.array(val_x), np.array(val_x2)], np.array(val_y)
        else:
            x, y = np.array(val_x), np.array(val_y)

        return x, y

if __name__ == '__main__':
    embedding = {'type': 'fasttext', 'path': 'fasttext-embedding/skipgram-256-news-classification.fasttext'}
    reader = ReadData('Participants_Data_News_category/Data_Train.xlsx', embedding, sentence_pair=True)

    x, y = reader.get_next_batch(0)
    print(x[0].shape, x[1].shape, y.shape)

    #generator = reader.generator()

    #for x, y in generator:
    #    print(x[0].shape, x[1].shape, y.shape)
