from keras.layers import Bidirectional, LSTM, Input, RepeatVector, Dense
from keras.layers import GlobalAveragePooling1D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers.core import Activation

from model_architectures.attention import AttentionLayer

class SentencePair:
    def __init__(self, hidden_size=512, no_classes=1, use_attention=False):
        self.hidden_size = hidden_size
        self.no_classes = no_classes

        self.use_attention = use_attention

    def build(self, input_shape=[(400, 256), (256,)]):
        story_input = Input(shape=input_shape[0], name='story_input')
        x1 = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='Bidirectional-1')(story_input)
        x1 = Bidirectional(LSTM(self.hidden_size, return_sequences=True, kernel_initializer='glorot_uniform'), name='Bidirectional-2')(x1)

        if self.use_attention:
            x1 = AttentionLayer(x1)

        x1 = GlobalAveragePooling1D()(x1)

        section_input = Input(shape=input_shape[1], name='section_input')
        x2 = Dense(self.hidden_size*2, kernel_initializer='glorot_uniform', name='Dense-1')(section_input)
        x2 = LeakyReLU(0.2)(x2)

        x = concatenate([x1, x2])

        x = Dense(self.hidden_size*2, kernel_initializer='glorot_uniform', name='Dense-2')(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(self.hidden_size*2, kernel_initializer='glorot_uniform', name='Dense-3')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(self.no_classes, kernel_initializer='glorot_uniform', name='output')(x)
        x = Activation('sigmoid', name='sigmoid')(x)

        return Model(inputs=[story_input, section_input], outputs=x)

if __name__ == '__main__':
    model = SentencePair()
    model = model.build()
    model.summary()
