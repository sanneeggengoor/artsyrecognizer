from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import pathlib

def path_to_label(image_path):
    im_name = image_path.split("/")[-1]
    art_name_list = im_name.split("_")[:-1]
    if art_name_list[0] == "Albrecht":
        art_name = "Albrecht Durer"
    else:
        art_name = ""
        for n in art_name_list:
            art_name = art_name + n + " "
        art_name = art_name[:-1]
    return art_name

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

print(sys.version)

tf.compat.v1.enable_eager_execution()
tf.version.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE
keras = tf.keras

data_root = "/home/sanne/Documents/RUG/DeepLearning/artsyrecognizer/resized/resizebyme"
data_root = pathlib.Path(data_root)
print(data_root)


import random
all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
label_names = []
for im_path in all_image_paths:
    label_names += [path_to_label(im_path)]
# print(label_names[:5])
label_names = list(set(label_names))
# print(label_names)

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)

all_image_labels = []
for path in all_image_paths:
    for name in label_names:
        if name.split(" ")[0] in path:
            all_image_labels += [label_to_index[name]]
            break



print("First 10 labels indices: ", all_image_labels[:10])
print(all_image_paths[0])



img_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.title(label_names[label].title())
print()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
