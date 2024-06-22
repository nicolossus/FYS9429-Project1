#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import islice

import jax.numpy as jnp
import numpy as np
from keras.datasets import mnist


def load_mnist(
    batch_size=None,
    drop_remainder=False,
    as_supervised=True,
    select_digits=None,
    binarized=False,
    shuffle=False,
    shuffle_seed=None,
):
    """
    Load JAX compatible MNIST train and test sets.

    Parameters
    ----------
    batch_size : int, optional
        The number of training samples in a mini-batch. Defualt: Entire train set
    drop_remainder : bool, optional
        Whether to drop the remainder of samples if batch_size is not integer
        divisble. Default: False
    as_supervised : bool, optional
        Whether to process and return dataset labels. Default: True
    select_digits : list, optional
        Create a dataset with a subset of the digits. Default: All digits in
        the dataset
    binarized : bool, optional
        Whether to binarize the pixel values. Default: False
    shuffle : bool, optional
        Whether to shuffle the files in the train set. Default: False
    shuffle_seed : int, optional
        Seed for random shuffling of files. Default: None

    Returns
    -------
    tuple
        If ``as_supervised=True``, the loader returns a 4-tuple structure
        ``(train_imgs, train_labels, test_imgs, test_labels)``.
        If ``as_supervised=False``, the loader returns a 2-tuple structure
        ``(train_imgs, test_imgs)``.
    """

    (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()

    train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], -1)

    if select_digits is not None:
        train_mask = np.isin(train_labels, select_digits)
        test_mask = np.isin(test_labels, select_digits)

        train_imgs = train_imgs[train_mask]
        test_imgs = test_imgs[test_mask]
        if as_supervised:
            train_labels = train_labels[train_mask]
            test_labels = test_labels[test_mask]

    # normalize pixel values
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0

    if binarized:
        train_imgs = _binarize_data(train_imgs)
        test_imgs = _binarize_data(test_imgs)

    # cast to jax
    train_imgs = jnp.asarray(train_imgs, dtype=jnp.float32)
    test_imgs = jnp.asarray(test_imgs, dtype=jnp.float32)
    if as_supervised:
        train_labels = jnp.asarray(train_labels, dtype=jnp.uint8)
        test_labels = jnp.asarray(test_labels, dtype=jnp.uint8)

    if shuffle:
        rng = np.random.default_rng(seed=shuffle_seed)
        permutation = rng.permutation(train_imgs.shape[0])
        train_imgs = train_imgs[permutation]
        if as_supervised:
            train_labels = train_labels[permutation]

    if batch_size is None:
        # no mini-batches, return early
        if as_supervised:
            return train_imgs, train_labels, test_imgs, test_labels
        return train_imgs, test_imgs

    # create mini-batches
    train_imgs_batched = _create_batches(train_imgs, batch_size, drop_remainder)

    if as_supervised:
        train_labels_batched = _create_batches(train_labels, batch_size, drop_remainder)
        return train_imgs_batched, train_labels_batched, test_imgs, test_labels

    return train_imgs_batched, test_imgs


def _create_batches(data, batch_size, drop_remainder):

    data_size = data.shape[0]
    remainder = data_size % batch_size

    if drop_remainder and remainder != 0:
        it = iter(data[: data_size - remainder, :])
    else:
        it = iter(data)

    batches = []

    while batch := tuple(islice(it, batch_size)):
        batches.append(jnp.asarray(batch))

    return batches


def _binarize_data(data):
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    return data
