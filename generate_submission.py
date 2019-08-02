import pandas as pd
import nltk
import numpy as np
import random
from tqdm import tqdm

from model_architectures.bilstm import BiLSTM
from model_architectures.res_bilstm import ResBiLSTM
from ReadData import ReadData

from keras.models import Model
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', required=True, help='Name of model to train [bilstm]')
parser.add_argument('--weights', '-w', required=True, help='Trained weights for Model')
parser.add_argument('--dataset', '-d', default='Participants_Data_News_category/Data_Test.xlsx', help='Path to dataset')
parser.add_argument('--embedding_path', '-ep', default='fasttext-embedding/skipgram-256-news-classification.fasttext',
                    help='Path to Embedding Model | Default: fasttext-embedding/skipgram-256-news-classification.fasttext')
parser.add_argument('--embedding_type', '-et', default='fasttext', help='Embedding type [fasttext] | Default: fasttext')
parser.add_argument('--no_classes', '-c', default=4, help='Number of Classes | Default: 4', type=int)
parser.add_argument('--hidden_size', '-hs', default=256, help='Hidden Size of LSTM Cell | Default: 256', type=int)

args = parser.parse_args()

hidden_size = args.hidden_size
if args.model == 'bilstm':
    inputs = (400, 256)
    model_instance = BiLSTM(hidden_size=hidden_size, no_classes=args.no_classes)
elif args.model == 'resbilstm':
    inputs = (400, 256)
    model_instance = ResBiLSTM(hidden_size=hidden_size, no_classes=args.no_classes)

model = model_instance.build(inputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(args.weights)

print('Model Loaded from {}.'.format(args.weights))

model.summary()

embedding = {'type': args.embedding_type, 'path': args.embedding_path}
reader = ReadData(path_file=None, embedding_config=embedding, data_shape=inputs, train_val_split=0)

test_data = pd.read_excel(args.dataset, sheet_name=None)['Sheet1']
test_data = list(test_data.STORY)

predictions = []

for story in tqdm(test_data):
    words = nltk.word_tokenize(story.lower())
    embedding = reader.sent2vec(words)

    prediction = model.predict(np.array([embedding]))[0]
    predictions.append(np.argmax(prediction))

df = pd.DataFrame([predictions])
df = df.transpose()
df.columns = ['SECTION']

df.to_excel('submission-file.xlsx')
