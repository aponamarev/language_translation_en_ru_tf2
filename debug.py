from tqdm import tqdm, tqdm_notebook
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_datasets.translate.ted_hrlr import TedHrlrTranslate


def data_etl(lang_pairs: str = 'ru_to_en', download_dir: str = ".") -> dict:
    print("Start data ETL")
    # Download a language data-set specified by :param language_pairs
    builder = TedHrlrTranslate(data_dir=download_dir, config=lang_pairs)
    builder.download_and_prepare()
    datasets = builder.as_dataset()
    print("Downloaded successfully")

    # extract data
    target, source = [], []
    for dataset_name in ['train', 'test', 'validation']:
        # extract dataset
        dataset = datasets[dataset_name]
        # convert into numpy
        dataset = tfds.as_numpy(dataset)
        # convert to string
        dataset = list(map(lambda features: (features['ru'].decode("utf-8"), features['en'].decode("utf-8")), dataset))
        source.extend([t[1] for t in dataset])
        target.extend([t[0] for t in dataset])

    print("Extracted successfully")

    # Tockenize
    x, x_tk = tokenize(source)
    y, y_tk = tokenize(target)

    x, x_length = pad(x)
    y, y_length = pad(y)

    print("Transformed successfully")

    return {'x': x, 'y': y, 'x_tk': x_tk, 'y_tk': y_tk, 'x_length': x_length, 'y_length': y_length}

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_tk = keras.preprocessing.text.Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None) -> tuple:
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in x])

    return keras.preprocessing.sequence.pad_sequences(x, maxlen=length, padding='post'), length

def logits_to_id(logits):
    """
    Turns logits into word ids
    :param logits: Logits from a neural network
    """
    return [prediction for prediction in np.argmax(logits, 1)]

def id_to_text(idx, tokenizer):
    """
    Turns id into text using the tokenizer
    :param idx: word id
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in idx])



if __name__ == '__main__':

    dataset = data_etl(lang_pairs='ru_to_en', download_dir=".")
    print("x shape: ", dataset['x'].shape)