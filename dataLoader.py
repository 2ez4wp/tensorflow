import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def image2tfrecord(image_list,label_list,tfrecord_path = None):
    if tfrecord_path == None:
        tfrecord_path = './tfrecord'
    length = len(image_list)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for i in range(length):
        image = Image.open(image_list[i])
        image_bytes = image.tobytes()
        features = {}
        features["images"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        features["labels"] = tf.train.Feature(int64_list=tf.train.BytesList(value=[int(label_list[i])]))
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
    writer.close()


