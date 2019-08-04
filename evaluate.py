import pandas as pd
import nltk
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import classification_report

from model_architectures.bilstm import BiLSTM
from model_architectures.res_bilstm import ResBiLSTM
from model_architectures.sentence_pair import SentencePair

from ReadData import ReadData

from keras.models import Model
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', required=True, help='Name of model to train [bilstm]')
parser.add_argument('--weights', '-w', required=True, help='Trained weights for Model')
parser.add_argument('--dataset', '-d', default='Participants_Data_News_category/Data_Train.xlsx', help='Path to dataset')
parser.add_argument('--embedding_path', '-ep', default='fasttext-embedding/skipgram-256-news-classification.fasttext',
                    help='Path to Embedding Model | Default: fasttext-embedding/skipgram-256-news-classification.fasttext')
parser.add_argument('--embedding_type', '-et', default='fasttext', help='Embedding type [fasttext] | Default: fasttext')
parser.add_argument('--no_classes', '-c', default=4, help='Number of Classes | Default: 4', type=int)
parser.add_argument('--hidden_size', '-hs', default=256, help='Hidden Size of LSTM Cell | Default: 256', type=int)
parser.add_argument('--size', '-s', default=0.2, help='Fraction of Dataset to use to evaluate the model, Default: 0.2', type=float)
args = parser.parse_args()

hidden_size = args.hidden_size
if args.model == 'bilstm':
    inputs = (400, 256)
    model_instance = BiLSTM(hidden_size=hidden_size, no_classes=args.no_classes)
elif args.model == 'resbilstm':
    inputs = (400, 256)
    model_instance = ResBiLSTM(hidden_size=hidden_size, no_classes=args.no_classes)
elif args.model == 'sentence_pair':
    inputs = [(400, 256), (256,)]
    model_instance = SentencePair(hidden_size=hidden_size, no_classes=args.no_classes)

model = model_instance.build(inputs)

if args.no_classes > 1:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(args.weights)

print('Model Loaded from {}.'.format(args.weights))

model.summary()

embedding = {'type': args.embedding_type, 'path': args.embedding_path}
if args.model == 'sentence_pair':
    reader = ReadData(path_file=None, embedding_config=embedding, data_shape=inputs, sentence_pair=True)
else:
    reader = ReadData(path_file=None, embedding_config=embedding, data_shape=inputs, sentence_pair=False)

test_data = pd.read_excel(args.dataset, sheet_name=None)['Sheet1']
test_data = test_data.sample(frac=1.0).reset_index(drop=True)
test_data = test_data.head(int(len(test_data)*args.size))

print(test_data.columns)

assert len(test_data.columns) > 1, "Labels of Test set not available."

test_label = list(test_data.SECTION)
test_data = list(test_data.STORY)

predictions = []

if args.model != 'sentence_pair':
    for story in tqdm(test_data):
        words = nltk.word_tokenize(story.lower())
        embedding = np.array([reader.sent2vec(words)])

        prediction = model.predict(embedding)[0]
        predictions.append(np.argmax(prediction))
else:
    for story in tqdm(test_data):
        words = nltk.word_tokenize(story.lower())
        embedding = np.array([reader.sent2vec(words)])
        prediction = []
        for i in range(4):
            x2 = np.array([reader.embedding[reader.label_map[i]]])
            prediction.append(model.predict([embedding, x2])[0])

        predictions.append(np.argmax(prediction))

print(classification_report(test_label, predictions, labels=[0, 1, 2, 3]))
