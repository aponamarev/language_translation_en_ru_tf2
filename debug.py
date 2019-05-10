import os
import subprocess
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras


def data_etl(download_dir: str = ".", file_name: str = "en_ru.tgz", n_lines: int = 3000,
             lenght_lim_lower: int = 4, length_lim_upper: int = 500, num_words: int = 5000) -> dict:
    path = os.path.join(download_dir, file_name)

    print("Start data ETL")

    if os.path.isfile(path):
        print("Reuse pre-downloaded " + path)
    else:
        # download a dataset
        print("Start downloading")
        subprocess.run(
                ["curl", "--output",
                 "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz",
                 path]
        )
        # extract
        subprocess.run("tar --extract --file".split() + [path])
        print("Data downloaded and extracted")

    # read data
    en = read_lines("paracrawl-release1.en-ru.zipporah0-dedup-clean.en", n_lines)
    ru = read_lines("paracrawl-release1.en-ru.zipporah0-dedup-clean.ru", n_lines)

    print("Extracted successfully")

    # filter out small samples
    indices = list(filter(lambda idx: length_lim_upper > len(en[idx].split()) >=lenght_lim_lower, range(len(en))))
    en = [en[idx] for idx in indices]
    ru = [ru[idx] for idx in indices]

    # Tockenize
    en = [re.sub("[0-9]", " \g<0>", s) for s in en]
    ru = [re.sub("[0-9]", " \g<0>", s) for s in ru]

    x, x_tk = tokenize(en, num_words=num_words, filters_regex=None)
    y, y_tk = tokenize(ru, num_words=num_words * 3, filters_regex=None)

    x = pad(x)
    y = pad(y)

    print("Transformed successfully")

    return {'x': x, 'y': y, 'x_tk': x_tk, 'y_tk': y_tk}


def read_lines(from_file: str, n_lines: int) -> list:
    if not os.path.isfile(from_file):
        raise FileNotFoundError(from_file)

    lines: list = []
    with open(from_file, "r") as f:
        for idx, line in enumerate(f):
            if idx < n_lines:
                lines.append(line)
            else:
                break

    return lines


def tokenize(x: list, num_words: int = 5000, filters_regex: str = None):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :param num_words: maximum number of words in the vocabulary - rare words will be replaced with a [rare] token
    :param filters_regex: regular expression filter string such as [^a-zA-Z0-9/-]
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    if filters_regex is not None:
        x_tk = keras.preprocessing.text.Tokenizer(num_words=num_words, filters=filters_regex, oov_token="[rare]")
    else:
        x_tk = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="[rare]")

    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None) -> np.ndarray:
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in x])

    return keras.preprocessing.sequence.pad_sequences(x, maxlen=length, padding='post')


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
    dataset = data_etl(download_dir=".", n_lines=5000, num_words=3000)
    print(id_to_text(dataset['x'][0], dataset['x_tk']))
    print(id_to_text(dataset['y'][0], dataset['y_tk']))
    print("x shape: ", dataset['x'].shape)