from keras.layers import Bidirectional, LSTM, Input, RepeatVector, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras.layers.core import Activation

class BiLSTM:
    def __init__(self, hidden_size=512, no_classes=4):
        self.hidden_size = hidden_size
        self.no_classes = no_classes

    def build(self, input_shape=(400, 256)):
        inp = Input(shape=input_shape, name='Input')

        x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='Bidirectional-1')(inp)
        x = Bidirectional(LSTM(self.hidden_size, return_sequences=False, kernel_initializer='glorot_uniform'), name='Bidirectional-2')(x)

        x = Dense(self.no_classes, kernel_initializer='glorot_uniform', name='output')(x)
        x = Activation('softmax', name='softmax')(x)

        return Model(inputs=inp, outputs=x)

if __name__ == '__main__':
    model_instance = BiLSTM()
    model = model_instance.build()
    model.summary()
