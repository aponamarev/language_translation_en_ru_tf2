import os
import subprocess
import re
import numpy as np
from tensorflow import keras
import pickle as pk


root_path, file_name = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(root_path, "data")
preprocessed_ds_file = os.path.join(data_path, "wlm_en_ru.pkl")


def data_etl(download_dir: str = ".", file_name: str = "en_ru.tgz", n_lines: int = None,
             en_vocab_size: int = 3000, ru_vocab_size: int = 10000) -> dict:
    path = os.path.join(download_dir, file_name)
    en_path = os.path.join(download_dir, "paracrawl-release1.en-ru.zipporah0-dedup-clean.en")
    ru_path = os.path.join(download_dir, "paracrawl-release1.en-ru.zipporah0-dedup-clean.ru")

    print("Start data ETL")

    if os.path.isfile(path):
        print("Reuse pre-downloaded " + path)
    else:
        # download a data-set
        print("Start downloading")
        completed = subprocess.run(
                ["curl",
                 "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz",
                 "--output",
                 path]
        )
        if completed.check_returncode() is not None:
            print("Downloading returned error code:", completed.check_returncode())
    # extract
    if not (os.path.isfile(en_path) and os.path.isfile(ru_path)):
        subprocess.run("tar --extract --file".split() + [path])
    print("Data downloaded and extracted")

    # read data
    en = read_lines(en_path, n_lines)
    ru = read_lines(ru_path, n_lines)

    print("Extracted successfully")
    # Transformation
    # remove english to english sequences
    valid_idx = filter(lambda idx: len(re.sub("[^а-я]"," ", ru[idx]).split()) / len(ru[idx].split()) > 0.5,
                       range(len(ru)))
    for ds in [en, ru]:
        # remove short and long sequences based on sequences
        lengths = list(map(lambda s: len(s.split()), ds))
        # valid lengths only used for mean and std calculations while lengths used to filter out en and ru datasets
        valid_lengths = np.fromiter(filter(lambda l: l>3, lengths), dtype=np.int)

        lengths_mean: float = valid_lengths.mean()
        lengths_std: float = valid_lengths.std()

        # remove extra long sequence as they will skew the loss towards 0
        length_lim_upper = int(lengths_mean + lengths_std * 1)

        # remove very short sequences
        length_lim_lower = max(4, int(lengths_mean - lengths_std * 2))

        valid_idx = list(filter(lambda idx: length_lim_lower <= lengths[idx] < length_lim_upper, valid_idx))

    # slit numbers
    en = [re.sub("[0-9]", " \g<0>", en[idx]) for idx in valid_idx]
    ru = [re.sub("[0-9]", " \g<0>", ru[idx]) for idx in valid_idx]

    # Tokenize
    x, x_tk = tokenize(en, num_words=en_vocab_size, filters_regex=None)
    y, y_tk = tokenize(ru, num_words=ru_vocab_size, filters_regex=None)

    x = pad(x)
    y_length = max([len(sentence) for sentence in y])
    y = pad(y, length=x.shape[1] if x.shape[1] > y_length else y_length)

    print("Transformed successfully")

    return {'x': x, 'y': y, 'x_tk': x_tk, 'y_tk': y_tk}


def read_lines(from_file: str, n_lines: int = None) -> list:
    if not os.path.isfile(from_file):
        raise FileNotFoundError(from_file)

    with open(from_file, "r") as f:
        if n_lines is not None:
            lines: list = []
            for idx, line in enumerate(f):
                if idx < n_lines:
                    lines.append(line)
                else:
                    break
        else:
            lines = [s for s in f]

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


def main() -> int:

    dataset = data_etl(download_dir=".", n_lines=50000, en_vocab_size=3500, ru_vocab_size=15000)
    print("Source(en) first example:")
    print(id_to_text(dataset['x'][0], dataset['x_tk']))
    print("Target(ru) first example:")
    print(id_to_text(dataset['y'][0], dataset['y_tk']))
    print("Source(en) last example:")
    print(id_to_text(dataset['x'][-1], dataset['x_tk']))
    print("Target(ru) last example:")
    print(id_to_text(dataset['y'][-1], dataset['y_tk']))

    print("x shape: ", dataset['x'].shape)
    print("y shape: ", dataset['y'].shape)

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    with open(preprocessed_ds_file, "wb") as file:
        pk.dump(dataset, file)

    print("Preprocessed data stored at:", preprocessed_ds_file)

    return 0

if __name__ == '__main__':

    main()

