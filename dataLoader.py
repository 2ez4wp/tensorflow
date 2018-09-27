import tensorflow as tf
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
slim_example_decoder = tf.contrib.slim.tfexample_decoder

#parameter


def image2tfrecord(image_list,label_list,tfrecord_path = None):
    if tfrecord_path == None:
        tfrecord_path = './tfrecord/train01.tfrecords'
    length = len(image_list)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for i in range(length):
        with tf.gfile.GFile(image_list[i], 'rb') as fid:
            encoded_image = fid.read()

        features = {}
        features["image"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image]))
        features["format"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')]))
        features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_list[i])]))
        #features["width"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.width]))
        #features["height"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.height]))
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
    writer.close()

def pares_tfrecord(serialized_example):
    dict = {
        'image': tf.FixedLenFeature((), tf.string),
        'format':tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)}

    parsed_example = tf.parse_single_example(serialized=serialized_example, features=dict)
    image = tf.image.decode_jpeg(parsed_example['image'])

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [640, 480],method=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    label = parsed_example['label']
    label = tf.cast(label, tf.int64)
    label = tf.one_hot(label, depth=10, on_value=1)

    return image,label

def get_list(path):
    dir_list = []
    if not os.path.exists(path):
        print("path not exist")
    else:
        name_list = os.listdir(path)
        for filename in name_list:
            filepath = os.path.join(path, filename)
            dir_list.append(filepath)
        return dir_list

if __name__ == '__main__':
    images_path = "./images"
    train_image_label_list = list()
    train_image_list = get_list(images_path)
    for i in range(len(train_image_list)):
        if i%2 ==0:
            train_image_label_list.append(0)
        else:
            train_image_label_list.append(1)
    #image2tfrecord(train_image_list, train_image_label_list)
    dataset = tf.data.TFRecordDataset(filenames=['./tfrecord/train01.tfrecords'])
    dataset = dataset.map(pares_tfrecord)
    dataset = dataset.batch(1).repeat(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print(next_element)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print("start")
        sess.run(fetches=init)
        try:
            while True:
                image, label = sess.run(fetches=next_element)
                plt.figure(2)
                plt.imshow(image[0])
                plt.show()
        except tf.errors.OutOfRangeError:
                print("end")
