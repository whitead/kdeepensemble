# Keras Deep Ensemble Implementation

This is an implementation of Lakshminarayanan et al. deep ensembles paper in Keras. It creates an ensemble
of models that can predict uncertainty. You provide a model which outputs two values (mean, variance) and the
library will ensemble and resample your data for ensemble training. We have made some modifications, which will be described more fully in an upcoming paper. Please no scoops.

## Install

```sh
pip install kdeepensemble
```

## Quickstart

This example makes a Keras model inside a function and then reshapes data for ensemble training. Notice a ``DeepEnsemble`` model acts just like a Keras model.

```python
import kdens
import tensorflow as tf

# this is where you define your model
def make_model():
    i = tf.keras.Input((None,))
    x = tf.keras.layers.Dense(10, activation="relu")
    mean = tf.keras.layers.Dense(1)(x)
    # this activation makes our variance strictly positive
    var = tf.keras.layers.Dense(1, activation='softplus')(x)
    out = tf.squeeze(tf.stack([muhat, stdhat], axis=-1))
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model

# prepare data for ensemble training
resampled_idx = kdens.resample(y)
x_train = x[idx]
y_train = y[idx]

deep_ens = kdens.DeepEnsemble(make_model)

deep_ens.compile(loss=kdens.neg_ll)
deep_ens.fit(x_train, y_train)

deep_ens(x)
```

## Model Output

The output is shape `(N, 3)`, where the last axis is mean, variance, and epistemic variance. Epistemic variance is from disagreements from models and reflects model uncertainty. The variance includes both epistemic and aleatoric variance. It represents the models best estimate of uncertainty.

## Saving/Loading

You can serialize the model with `model.save`, but note that training will not be abel to continue. To continue training, use the `load_weights` and `save_weights` methods.

## Tensorflow Dataset

You can use ``map_reshape`` when working with a Tensorflow dataset. It will  If your data is already batched, add the `is_batched=True` argument.

```python

# data is a tf.data.Dataset

data = data.map(kdens.map_reshape()).batch(8)
deep_ens = kdens.DeepEnsemble(make_model)
deep_ens.compile(loss=kdens.neg_ll)
deep_ens.fit(data)
```

Note that ``map_reshape`` will not resample, just reshape. If you would like to balance your labels, consider [`rejection_sample`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

## Working with multiple inputs

This library does support Keras models that have multiple inputs with the following restrictions:

1. The inputs must be tuples (lists and dicts are not supported).
2. The adversarial step will be done on only the first element of the input tuple.

Here's an example

```python

# make a model that takes three inputs as a tuple
x = np.random.randn(100, 10).astype(np.float32)
t = np.random.randn(100, 10, 3).astype(np.float32)
z = np.random.randn(100, 10, 3).astype(np.float32)
y = np.random.randn(100).astype(np.float32)

# can still use map_reshape with tuples
data = tf.data.Dataset.from_tensor_slices(
    ((x, t, z), y)).map(map_reshape()).batch(8)

def build():
    xi = tf.keras.layers.Input(shape=(10,))
    ti = tf.keras.layers.Input(shape=(10, 3))
    zi = tf.keras.layers.Input(shape=(10, 3))
    x = tf.keras.layers.Dense(2)(xi)
    return tf.keras.Model(inputs=(xi, ti, zi), outputs=x)

deep_ens = kdens.DeepEnsemble(make_model)
deep_ens.compile(loss=kdens.neg_ll)
deep_ens.fit(data, epochs=2)
deep_ens.evaluate((x, t, z), y)
```

## API

[See API](https://whitead.github.io/kdeepensemble/api.html)

## Citation

Deep ensemble paper:
```bibtex
@article{lakshminarayanan2017simple,
  title={Simple and scalable predictive uncertainty estimation using deep ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
