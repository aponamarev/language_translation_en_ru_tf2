import os
from tqdm import tqdm, tqdm_notebook
from typing import Callable, Union, Iterable
from functools import partial
import numpy as np
import pickle as pk
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K


learning_rate = 1e-3
embeddings_units = 16
gru_units = 128
epochs = 5
validation_split = 0.1
batch_size = 16
loss = keras.losses.sparse_categorical_crossentropy
max_words: int = None


def focal_loss(gamma=2., alpha=.25):

    def call(y_true, y_pred):
        y_true = tf.squeeze(tf.cast(y_true, tf.int32))
        vocab_size = y_pred.shape[-1]

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        truth = tf.one_hot(y_true, depth=vocab_size, dtype=tf.float32)

        p_1 = -alpha * tf.reduce_sum(truth * tf.pow(1. - y_pred, gamma) * tf.math.log(y_pred), axis=-1)
        p_0 = - (1 - alpha) * tf.reduce_sum((1 - truth) * tf.pow(y_pred, gamma) * tf.math.log(1. - y_pred), axis=-1)

        cost = p_1 + p_0

        return cost

    return call


class Encoder(keras.Model):


    def __init__(self, vocab_size: int, gru_units: int, embeddings_units: int):

        super(Encoder, self).__init__()

        self.embedding = keras.layers.Embedding(vocab_size, embeddings_units)
        self.BN_embedding = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))
        self.gru = keras.layers.GRU(gru_units, return_sequences=False, return_state=True)

    def call(self, x):

        embedded = self.embedding(x)
        bn_embedding = self.BN_embedding(embedded)
        h_s = self.gru(bn_embedding)

        return h_s[0]


class Decoder(keras.Model):


    def __init__(self, vocab_size: int, gru_units: int, embeddings_units: int):
        super(Decoder, self).__init__()

        self.hs_BN = keras.layers.BatchNormalization(axis=-1)
        # the embedding layer needs to contain 1 additional token for the start of the sequence (dataset['y'].max()+1)
        self.embedding = keras.layers.Embedding(vocab_size+1, embeddings_units)
        self.embedding_BN = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))
        self.gru = keras.layers.GRU(gru_units, return_sequences=True, return_state=True)
        self.gru_BN = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))
        self.dense = keras.layers.Dense(vocab_size, activation='softmax')


    def call(self, x, hidden):

        embedded = self.embedding_BN(
                self.embedding(x)
        )

        seq, h_t = self.gru(embedded, initial_state = self.hs_BN(hidden))
        p = self.dense(self.gru_BN(seq))

        return p, h_t


def FocalLoss_model(input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    input_seq = keras.Input(input_shape[1:])
    if output_sequence_length > input_shape[1]:
        expanded_seq = keras.backend.squeeze(
                keras.layers.ZeroPadding1D((0, output_sequence_length - input_shape[1]))(
                        keras.layers.Reshape((input_shape[1], 1))(input_seq)
                ),
                axis=-1
        )
    else:
        expanded_seq = input_seq

    embedded_seq = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))(
            keras.layers.Embedding(source_vocab_size, embeddings_units, input_length=output_sequence_length)(
                expanded_seq)
    )
    rnn = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))(
            keras.layers.GRU(gru_units, return_sequences=True)(embedded_seq)
    )
    probabilities = keras.layers.TimeDistributed(keras.layers.Dense(target_vocab_size, activation='softmax'))(rnn)

    model = keras.Model(input_seq, probabilities)

    model.compile(loss=focal_loss(alpha=.25, gamma=2),
                  optimizer=keras.optimizers.Adam(learning_rate, clipnorm=3.0),
                  metrics=['accuracy'])
    return model


#@tf.function
def train_step(inputs, targets,
               encoder, decoder,
               loss_fn, optimizer,
               start_word_index):

    loss = 0.0
    BATCH_SIZE = int(targets.shape[0])
    N_STEPS = int(targets.shape[1])

    with tf.GradientTape() as tape:

        predictions = tf.TensorArray(dtype=tf.float32, size=N_STEPS)
        h_s = encoder(inputs=inputs)
        # Kick-off decoding with a start word
        dec_input = tf.expand_dims([start_word_index] * BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        h_t = h_s
        for t in range(N_STEPS):
            p, h_t = decoder(dec_input, hidden=h_t)
            predictions.write(t, tf.squeeze(p))
            # normalize loss across time dimension
            loss += loss_fn(targets[:, t], p) / tf.cast(N_STEPS, p.dtype)
            # using teacher forcing
            dec_input = tf.expand_dims(targets[:, t], 1)

    predictions = predictions.stack()
    predictions = tf.transpose(predictions, (1,0,2))
    # obtain all trainable variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # obtain gradient history across all steps with respect to trainable variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return tf.reduce_mean(loss), predictions


def fit(inputs: np.ndarray, targets: np.ndarray, batch_size: int, epochs: int,
        train_step: Callable[[np.ndarray, np.ndarray], float], metrics: Iterable[keras.metrics.Metric]):

    steps = inputs.shape[0] // batch_size
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):

        pbar = tqdm(range(steps), desc="Epoch {}".format(epoch))
        minibatch = enumerate(dataset.take(steps))

        for i in pbar:

            _, (x, y) = next(minibatch)
            loss, predictions = train_step(x, y)
            for m in metrics:
                m.update_state(y, predictions)

            reporting = {m.name: m.result().numpy()
                         for m in metrics}
            reporting["loss"] = loss.numpy()
            pbar.set_postfix(ordered_dict=reporting)


if __name__ == '__main__':
    # download the data
    with open("data/wlm_en_ru.pkl", 'rb') as file:
        dataset = pk.load(file)

    #v = focal_loss()(dataset['y'][:32, :, None], tf.zeros((32, dataset['y'].shape[1], 15000), dtype=tf.float32))
    # Create the neural network
    max_words = dataset['y'].max() + 1
    keras.backend.clear_session()
    # remember 1 token is reserved for unknown words
    encoder = Encoder(vocab_size=dataset['x'].max() + 1, gru_units=gru_units, embeddings_units=embeddings_units)
    # remember 1 token is reserved for unknown words
    decoder = Decoder(vocab_size=dataset['y'].max() + 1, gru_units=gru_units, embeddings_units=embeddings_units)


    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss_fn = keras.losses.sparse_categorical_crossentropy
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # the decoder contains embedding with target vocabulary size + 1. Additional (+1) token is reserved for the start
    # token.
    train_step_fn = partial(train_step, encoder=encoder, decoder=decoder, loss_fn=loss_fn, optimizer=optimizer,
                            start_word_index=dataset['y'].max()+1)

    fit(dataset['x'], dataset['y'], batch_size=batch_size, epochs=epochs, train_step=train_step_fn, metrics=metrics)