from keras.layers import merge, Dense, RepeatVector
from keras.layers.core import Permute, Reshape, Lambda

def AttentionLayer(inputs, timesteps=400):
    assert len(inputs.shape) == 3, 'Attention input should be of dim 3 but found {} dims'.format(len(inputs.shape))

    input_dim = inputs.shape[2]
    a = Permute((2, 1))(inputs)
    a = Reshape((int(input_dim), int(timesteps)))(a)
    a = Dense(timesteps, activation='softmax')(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output = merge([inputs, a_probs], name='attention_mul', mode='mul')

    return output
