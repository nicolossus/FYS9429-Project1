#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy.testing as nptest
import pytest

from neurovae import load_mnist


@pytest.fixture(scope="module")
def full_ds():
    train_imgs, train_labels, test_imgs, test_labels = load_mnist()

    # Declaring namedtuple()
    full_ds = namedtuple("full_ds", ["train_imgs", "train_labels", "test_imgs", "test_labels"])

    return full_ds(train_imgs, train_labels, test_imgs, test_labels)


@pytest.mark.parametrize(
    "item, correct_shape",
    [
        ["train_imgs", (60000, 784)],
        ["test_imgs", (10000, 784)],
        ["train_labels", (60000,)],
        ["test_labels", (10000,)],
    ],
)
def test_correct_shape_full_ds(full_ds, item, correct_shape):
    """Check that the loaded full dataset has correct shapes"""
    nptest.assert_array_equal(getattr(full_ds, item).shape, correct_shape)


def test_correct_batch_size():
    pass


"""
import numpy.testing as nptest

# TODO: move these to a proper test file

train_imgs_full, train_labels_full, test_imgs_full, test_labels_full = load_mnist()

unique_full, counts_full = np.unique(train_labels_full, return_counts=True)
counts_dict = dict(zip(unique_full, counts_full))

# check correct shapes for full dataset
nptest.assert_array_equal(train_imgs_full.shape, (60000, 784))
nptest.assert_array_equal(test_imgs_full.shape, (10000, 784))
nptest.assert_array_equal(train_labels_full.shape, (60000,))
nptest.assert_array_equal(test_labels_full.shape, (10000,))

# check correct batch size
ds_train, _ = load_mnist(batch_size=32, drop_remainder=True, as_supervised=False)
assert len(ds_train) == int(60000 / 32)  # TODO: rhs computation can be written more robust

ds_train, _ = load_mnist(batch_size=128, drop_remainder=False, as_supervised=False)
assert len(ds_train) == int(60000 / 128) + 1  # TODO: rhs computation can be written more robust

# check correct batch size for supervised
_, train_labels_batched, _, _ = load_mnist(batch_size=32, drop_remainder=True, as_supervised=True)
assert len(train_labels_batched) == int(60000 / 32)  # TODO: rhs computation can be written more robust

# check select digits
# TODO: extend to test_labels
_, train_labels, _, test_labels = load_mnist(select_digits=[0, 1])

unique, counts = np.unique(train_labels, return_counts=True)
for digit, count in zip(unique, counts):
    assert count == counts_dict[digit]

# check binarize
# TODO: extend to ds_test
ds_train, ds_test = load_mnist(binarized=True, as_supervised=False)
unique_train = np.unique(ds_train)
nptest.assert_array_equal(unique_train, [0.0, 1.0])

# TODO: check shuffling
"""
