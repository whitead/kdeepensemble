import unittest
import numpy as np
import tensorflow as tf
from kdens import *


class TestKDens(unittest.TestCase):
    def test_resample(self):
        y = np.random.randn(100)
        idx = resample(y, (y.shape[0], 10))
        assert idx.shape == (y.shape[0], 10)

        idx = resample(y, (3, 5))
        assert idx.shape == (3, 5)

        y = np.random.randn(20, 3)
        idx = resample(y, (3, 10))
        assert idx.shape == (3, 10)

    def test_resample_class(self):
        y = np.random.randint(0, 10, size=(100, 3))
        idx = resample(y, (y.shape[0], 10))
        assert idx.shape == (y.shape[0], 10)
        assert y[idx].shape == (y.shape[0], 10, 3)

    def test_deep_ensemble(self):
        x = np.random.randn(100, 10)
        y = np.random.randn(100)
        idx = resample(y)
        xb = x[idx]
        yb = y[idx]
        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_reshape(self):
        x = np.random.randn(100, 10)
        y = np.random.randn(100)
        idx = resample(y, output_shape=(None, 2))
        xb = x[idx]
        yb = y[idx]
        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            ),
            nmodels=2,
            adv_epsilon=0.1,
        )
        d.compile(metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_tfdata(self):
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y)).map(map_reshape()).batch(8)
        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(metrics=["mae"])
        d.fit(data, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_tfdata_resample(self):
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y)).map(map_reshape()).batch(8)

        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(metrics=["mae"])
        d.fit(data, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_partial(self):
        x = np.random.randint(0, 10, size=(100, 25))
        y = np.random.randn(100)
        idx = resample(y)
        xb = x[idx]
        yb = y[idx]

        def build_model():
            inputs = tf.keras.Input(shape=(None,))

            # make embedding and indicate that 0 should be treated as padding mask
            e = tf.keras.layers.Embedding(input_dim=10, output_dim=3)(inputs)
            x = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(e)
            out = tf.keras.layers.Dense(2)(x)
            model = tf.keras.Model(inputs=inputs, outputs=out, name="sol-rnn")
            partial_in = tf.keras.Model(inputs=inputs, outputs=e)
            partial_out = tf.keras.Model(inputs=e, outputs=out)
            return model, partial_in, partial_out

        d = DeepEnsemble(build_model, partial=True)
        d.compile(metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)
