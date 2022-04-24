# Keras Deep Ensemble Implementation

This is an implementation of Lakshminarayanan et al. deep ensembles paper in Keras. It creates an ensemble
of models that can predict uncertainty. You provide a model which outputs two values (mean, variance) and the
library will ensemble and resample your data for ensemble training.

This package is meant to be really simple. It has one function and one class: ``resample(y)``, which reshapes data for ensemble training and ``DeepEnsemble``, which ensembles a Keras model.

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

# loss is always log-likelihood, so not specified
deep_ens.compile()
deep_ens.fit(x_train, y_train)

deep_ens(x)
```

## API

[See API](https://whitead.github.io/kdeepensemble/api.html)
