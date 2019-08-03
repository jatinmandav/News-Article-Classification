from model_architectures.bilstm import BiLSTM
from model_architectures.res_bilstm import ResBiLSTM
from model_architectures.sentence_pair import SentencePair

from ReadData import ReadData

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import os
import argparse

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', required=True, help='Name of model to train [bilstm, resbilstm, sentence_pair]')
parser.add_argument('--dataset', '-d', default='Participants_Data_News_category/Data_Train.xlsx', help='Path to dataset')
parser.add_argument('--embedding_path', '-ep', default='fasttext-embedding/skipgram-256-news-classification.fasttext',
                    help='Path to Embedding Model | Default: fasttext-embedding/skipgram-256-news-classification.fasttext')
parser.add_argument('--embedding_type', '-et', default='fasttext', help='Embedding type [fasttext] | Default: fasttext')
parser.add_argument('--batch_size', '-b', default=64, help='Batch Size | Default: 64', type=int)
parser.add_argument('--epochs', '-e', default=50, help='No of Epochs | Default: 50', type=int)
parser.add_argument('--logs', '-l', default='logs', help='Path to Logs (weights, tensorboard) | Default: logs_[model_name]', type=str)
parser.add_argument('--no_classes', '-c', default=4, help='Number of Classes | Default: 4', type=int)
parser.add_argument('--hidden_size', '-hs', default=256, help='Hidden Size of LSTM Cell | Default: 256', type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, help='Learning Rate | Default: 0.001', type=float)
parser.add_argument('--train_val_split', '-tvs', default=0.2, help='Train vs Validation Split | Default: 0.2', type=float)
parser.add_argument('--check_build', action='store_true', help='Check Model Build')

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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
else:
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
model.summary()

if args.check_build:
    exit()

embedding = {'type': args.embedding_type, 'path': args.embedding_path}
if args.model == 'sentence_pair':
    reader = ReadData(path_file=args.dataset, embedding_config=embedding, data_shape=inputs, train_val_split=args.train_val_split, sentence_pair=True)
else:
    reader = ReadData(path_file=args.dataset, embedding_config=embedding, data_shape=inputs, train_val_split=args.train_val_split, sentence_pair=False)

print('Reading Validation Data ..')
val_x, val_y = reader.read_val()

train_generator = reader.generator()

log_dir = args.logs + '_' + args.model
logging = TrainValTensorBoard(log_dir=log_dir)

checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'),
                            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

if not args.model == 'sentence_pair':
    model.fit_generator(generator=train_generator, steps_per_epoch=int(reader.train_size/args.batch_size),
                        validation_data=(val_x, val_y), epochs=args.epochs,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
else:
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    for epoch in range(args.epochs):
        num_batches = int(reader.train_size/args.batch_size)

        start_index = 0
        epoch_loss, epoch_acc = [], []
        for i in range(num_batches):
            start_index = i*args.batch_size
            epoch_x, epoch_y = reader.get_next_batch(start_index, args.batch_size)
            [loss, acc] = model.train_on_batch(epoch_x, epoch_y)

            epoch_loss.append(loss)
            epoch_acc.append(acc)


        num_batches = int(reader.val_size/args.batch_size)
        start_index = 0
        val_loss, val_acc = [], []
        for i in range(num_batches):
            start_index = i*args.batch_size
            epoch_x, epoch_y = reader.get_next_val_batch(start_index, args.batch_size)
            [loss, acc] = model.test_on_batch(epoch_x, epoch_y)

            val_loss.append(loss)
            val_acc.append(acc)

        print('Epoch {}/{}: loss: {}, acc: {}, val_loss: {}, val_acc: {}'.format(
                epoch+1, args.epochs, np.average(epoch_loss), np.average(epoch_acc),
                np.average(val_loss), np.average(val_acc)))

        model.save_weights(os.path.join(log_dir, 'ep{}-val_loss{}-val_acc{}.h5'.format(epoch+1, np.average(val_loss), np.average(val_acc))))
