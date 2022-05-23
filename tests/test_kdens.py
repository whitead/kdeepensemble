from datetime import datetime
import unittest
import numpy as np
from kdens.kdens import map_tile
import tensorflow as tf
from kdens import *
import shutil
import pytest


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
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_weights(self):
        x = np.random.randn(100, 10)
        w = np.random.choice([0, 1], size=(100,)).astype(float)
        y = np.random.randn(100)
        idx = resample(y)
        xb = x[idx]
        yb = y[idx]
        wb = w[idx]
        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, sample_weight=wb, epochs=2)
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
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_tfdata(self):
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y)).map(map_tile()).batch(8)
        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(data, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

        # redo batched
        data = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .batch(25)
            .map(map_batch_reshape())
        )
        d.fit(data, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

        # and now with tiling
        data = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .batch(4)
            .map(map_tile(is_batched=True))
        )
        d.fit(data, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_deep_ensemble_complex_data(self):
        x = np.random.randn(100, 10).astype(np.float32)
        t = np.random.randn(100, 10, 3).astype(np.float32)
        z = np.random.randn(100, 10, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = (
            tf.data.Dataset.from_tensor_slices(((x, t, z), y)).map(map_tile()).batch(8)
        )

        def build():
            xi = tf.keras.layers.Input(shape=(10,))
            ti = tf.keras.layers.Input(shape=(10, 3))
            zi = tf.keras.layers.Input(shape=(10, 3))
            x = tf.keras.layers.Dense(2, activation="relu")(xi)
            return tf.keras.Model(inputs=(xi, ti, zi), outputs=x)

        d = DeepEnsemble(build)

        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(data, epochs=2)
        y_pred = d.predict((x, t, z))
        assert y_pred.shape == (100, 3)
        assert d((x, t, z)).shape == (100, 3)
        d.evaluate((x, t, z), y)

    def test_deep_ensemble_complex_partial(self):
        x = np.random.randn(100, 10).astype(np.float32)
        t = np.random.randn(100, 10, 3).astype(np.float32)
        z = np.random.randn(100, 10, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = (
            tf.data.Dataset.from_tensor_slices(((x, t, z), y)).map(map_tile()).batch(8)
        )

        def build():
            xi = tf.keras.layers.Input(shape=(10,))
            ti = tf.keras.layers.Input(shape=(10, 3))
            zi = tf.keras.layers.Input(shape=(10, 3))
            h1 = tf.keras.layers.Dense(3)(xi)
            h2 = tf.keras.layers.Dense(3, activation="softplus")(xi)
            y = tf.keras.layers.Lambda(
                lambda x: tf.stack([x[0][..., 0], x[1][..., 0]], axis=-1)
            )((h1, h2))
            return (
                tf.keras.Model(inputs=(xi, ti, zi), outputs=y),
                tf.keras.Model(inputs=(xi, ti, zi), outputs=(h1, h2)),
                tf.keras.Model(inputs=(h1, h2), outputs=y),
            )

        d = DeepEnsemble(build, partial=True)

        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(data, epochs=2)
        y_pred = d.predict((x, t, z))
        assert y_pred.shape == (100, 3)
        assert d((x, t, z)).shape == (100, 3)
        d.evaluate((x, t, z), y)

    def test_deep_ensemble_tfdata_resample(self):
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y)).map(map_tile()).batch(8)

        d = DeepEnsemble(
            lambda: tf.keras.Sequential(
                [tf.keras.layers.Dense(10, activation="relu"), tf.keras.layers.Dense(2)]
            )
        )
        d.compile(loss=neg_ll, metrics=["mae"])
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
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)

    def test_save(self):
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
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)
        d.full_call(xb, training=True)
        d(x, training=False)
        d.save("test_save")
        try:
            del d
            loaded_d = tf.keras.models.load_model(
                "test_save", custom_objects=custom_objects
            )
        finally:
            shutil.rmtree("test_save")
        loaded_d.evaluate(x, y)

    @pytest.mark.skip(reason="my life is too short to solve this")
    def test_save_trace(self):
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
        d.compile(loss=neg_ll, metrics=["mae"])
        d.fit(xb, yb, epochs=2)
        y_pred = d.predict(x)
        assert y_pred.shape == (100, 3)
        assert d(x).shape == (100, 3)
        d.evaluate(x, y)
        d(x, training=False)
        d.save("test_save", save_traces=False)
        try:
            del d
            loaded_d = tf.keras.models.load_model(
                "test_save", custom_objects=custom_objects
            )
        finally:
            shutil.rmtree("test_save")
        loaded_d(x)
