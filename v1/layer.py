# typed = Partial
import os
import pandas as pd
import numpy as np
import enum

from preprocess import preprocess

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Utils:
    @staticmethod
    def normalize(x, max_val):
        return x / max_val

    @staticmethod
    def stable_normalize(x, max_val):
        return x * 0.99 / max_val + 0.01

    @staticmethod
    def one_hot(y):
        ret = np.zeros((y.size, 2))
        ret[np.arange(y.size), y] = 1
        return ret.T

    @staticmethod
    def seed(val):
        np.random.seed(val)


class LayerType(enum.Enum):
    Starter = 1
    Dense = 2
    Ender = 3
    Test = 4


class DataTransformer:
    WORD_MIN_OCCURRENCE = 2
    max_val = 0
    dictionary = []

    def __init__(self, data_file_path, normalization, float_type, layer_type) -> None:
        data = pd.read_csv(data_file_path)
        tweets = data["text"]
        processed_tweets = []
        words = {}
        self._x = []
        self._y = data["target"].to_numpy() if layer_type == LayerType.Starter else None
        self._ids = data["id"] if layer_type == LayerType.Test else None

        def add(word: str):
            if word not in words:
                words[word] = 0
            words[word] += 1

        # preprocessing
        for tweet in tweets:
            processed = preprocess(tweet)
            processed_tweets.append(processed)
            if layer_type == LayerType.Starter:
                for word in processed:
                    add(word)

        # remove uncommon words
        if layer_type == LayerType.Starter:
            for key in list(words.keys()):
                if words[key] <= DataTransformer.WORD_MIN_OCCURRENCE:
                    del words[key]
            DataTransformer.dictionary = words.keys()

        # remove unseen words
        if layer_type == LayerType.Test:
            dict_set = set(DataTransformer.dictionary)
            for key in list(words.keys()):
                if key not in dict_set:
                    del words[key]

        # transformation
        for tweet in processed_tweets:
            cur_dict = {word: 0 for word in DataTransformer.dictionary}
            for word in tweet:
                if word in cur_dict:
                    cur_dict[word] += 1
            self._x.append(list(cur_dict.values()))
        self._x = np.array(self._x)

        # find maximum value for normalization
        if layer_type == LayerType.Starter:
            DataTransformer.max_val = self._x.max()

        # normalization
        self._x = normalization(self._x, DataTransformer.max_val).astype(float_type)

    @classmethod
    def train(cls, file_path, normalization, float_type):
        return cls(file_path, normalization, float_type, LayerType.Starter)

    @classmethod
    def test(cls, file_path, normalization, float_type):
        return cls(file_path, normalization, float_type, LayerType.Test)

    @classmethod
    def train_default(cls, file_path):
        return cls(file_path, Utils.normalize, "float32", LayerType.Starter)

    @classmethod
    def test_default(cls, file_path):
        return cls(file_path, Utils.normalize, "float32", LayerType.Test)

    @classmethod
    def train_stable(cls, file_path):
        return cls(file_path, Utils.stable_normalize, "float32", LayerType.Starter)

    @classmethod
    def test_stable(cls, file_path):
        return cls(file_path, Utils.stable_normalize, "float32", LayerType.Test)


class Starter:
    _type = LayerType.Starter

    def __init__(
        self, train_file_path, normalization, batch_size, float_type="float32"
    ) -> None:
        data = DataTransformer.train(train_file_path, normalization, float_type)
        self._x, self._y = data._x, data._y
        del data

        # other variables
        self.size = self._x.shape[1]
        self.batch_size = batch_size
        self.a, self.indices = None, None

    def init(self, ender, *args, **kwargs):
        ender._y = self._y.T
        del self._y

    def forward(self, *args, **kwargs):
        self.indices = np.random.choice(
            self._x.shape[0], self.batch_size, replace=False
        )
        self.a = self._x[self.indices].T

    def backward(self, *args, **kwargs):
        pass

    def update(self):
        pass


class Ender:
    _type = LayerType.Ender

    def __init__(self) -> None:
        self.a, self._y, self.expected, self._one_hot_y = (None for _ in range(4))

    def init(self, *args, **kwargs):
        self._one_hot_y = Utils.one_hot(self._y).T

    def forward(self, *args, **kwargs):
        pass

    def backward(self, prev, post):
        self.expected = self._y[post.indices]
        self.a = self._one_hot_y[post.indices].T

    def update(self):
        pass


class Dense:
    _type = LayerType.Dense

    def __init__(self, size, activation):
        self.size = size
        self.act = activation.forward
        self.deriv = activation.backward
        self.alpha = None
        (
            self.w,
            self.b,
            self.a,
            self.z,
            self.dw,
            self.db,
            self.dz,
            self.batch_size,
        ) = (None for i in range(8))

    def init(self, prev, **kwargs):
        self.alpha = kwargs.get("learning_rate")
        self.w = np.random.rand(self.size, prev.size) - 0.5
        self.b = np.random.rand(self.size, 1) - 0.5
        self.batch_size = prev.batch_size

    def forward(self, prev):
        self.z = self.w.dot(prev.a) + self.b
        self.a = self.act(self.z)

    def backward(self, prev, post):
        self.dz = self.deriv(self, post)
        self.dw = self.dz.dot(prev.a.T) / self.batch_size
        self.db = np.sum(self.dz) / self.batch_size

    def update(self):
        self.w = self.w - self.dw * self.alpha
        self.b = self.b - self.db * self.alpha
