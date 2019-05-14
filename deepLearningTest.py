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
import datasettf as dstf


keras = tf.keras


image_label_ds, image_count, label_names = dstf.make_dataset()

DATASET_SIZE  = image_count

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

#full_dataset = tf.data.TFRecordDataset(image_label_ds)
full_dataset =  image_label_ds.shuffle(buffer_size = image_count)
train_dataset = image_label_ds.take(train_size)
test_dataset =  image_label_ds.skip(train_size)
val_dataset =   image_label_ds.skip(val_size)
test_dataset =  image_label_ds.take(test_size)


ds = train_dataset
ds_size = int(0.7 * image_count)

BATCH_SIZE = 5

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=ds_size)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds_full = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_full



mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer= 'adam' ,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
len(model.trainable_variables)

model.summary()

steps_per_epoch=tf.math.ceil(ds_size/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)

from tensorflow.keras.preprocessing import image

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(192, 192))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

#img_path = 'C:/Users/Ferhat/Python Code/Workshop/Tensoorflow transfer learning/blue_tit.jpg'
img_path = "/home/sanne/Documents/RUG/DeepLearning/artsyrecognizer/resized/resizebyme/Amedeo_Modigliani_32.jpg"
new_image = load_image(img_path)

pred = model.predict(new_image)
print(pred)

# The dataset may take a few seconds to start, as it fills its shuffle buffer.



# SPLIT_WEIGHTS = (8, 1, 1)
# splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
#
# (raw_train, raw_validation, raw_test), metadata = tfds.load(
#     'cats_vs_dogs', split=list(splits),
#     with_info=True, as_supervised=True)
#
# print(raw_train)
# print(raw_validation)
# print(raw_test)
#
#
# get_label_name = metadata.features['label'].int2str
#
# for image, label in raw_train.take(2):
#   plt.figure()
#   plt.imshow(image)
#   plt.title(get_label_name(label))
