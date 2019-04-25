from __future__ import absolute_import, division, print_function, unicode_literals

import os
print(os)
import numpy as np
print(np)
import matplotlib
import matplotlib.pyplot as plt
print(plt)
import tensorflow as tf
print(tf)
import tensorflow_datasets as tfds

keras = tf.keras

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)


get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
