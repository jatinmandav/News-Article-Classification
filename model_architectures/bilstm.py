from keras.layers import Bidirectional, LSTM, Input, RepeatVector, Dense
from keras.layers import GlobalAveragePooling1D, concatenate, GlobalMaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras.layers.core import Activation

from model_architectures.attention import AttentionLayer, AttentionWithContext

class BiLSTM:
    def __init__(self, hidden_size=512, no_classes=4, use_attention=False):
        self.hidden_size = hidden_size
        self.no_classes = no_classes
        self.use_attention = use_attention

    def build(self, input_shape=(400, 256)):
        inp = Input(shape=input_shape, name='Input')

        x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='Bidirectional-1')(inp)
        x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='Bidirectional-2')(x)

        if self.use_attention:
            #x1 = AttentionLayer(x1)
            x1 = AttentionWithContext()(x1)
        else:
            x2 = GlobalMaxPooling1D()(x1)
            x1 = GlobalAveragePooling1D()(x1)
            x1 = concatenate([x1, x2])

        x = Dense(self.no_classes, kernel_initializer='glorot_uniform', name='output')(x)
        x = Activation('softmax', name='softmax')(x)

        return Model(inputs=inp, outputs=x)

if __name__ == '__main__':
    model_instance = BiLSTM()
    model = model_instance.build()
    model.summary()
