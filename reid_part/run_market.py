import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow import keras
import numpy as np
import cv2
import market1501_dataset
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", dest="data_dir", type=str, metavar='<str>', default='datasets', help='path to dataset')
parser.add_argument('-e','--epochs', dest="epochs", type=int, metavar='<int>', default=500, help='max epochs for training')
parser.add_argument('-b','--batch_size', dest="batch_size", type=int, metavar='<int>', default=32, help='batch_size for training')
parser.add_argument('-l','--logs', dest="logs_dir", type=str, metavar='<str>',default='logs/', help='path to logs')
parser.add_argument('-m', '--mode', dest='mode', type=str, metavar='<str>', default='train', help='mode - train, val, or test')

FLAGS = parser.parse_args()
    
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160


def process_train(images):
    split = tf.split(images, FLAGS.batch_size, axis=0)
    for i in range(FLAGS.batch_size):
        split[i] = tf.reshape(split[i], shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        split[i] = process(split[i])
    split = tf.stack(split)
    return split
    
def process(img):
    img = tf.image.resize(img, [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
    img = tf.image.random_crop(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.per_image_standardization(img)
    return img

def process_test_single(img):
    return tf.image.per_image_standardization(img)

def process_test(images):
    split = tf.split(images, FLAGS.batch_size, axis=0)
    for i in range(FLAGS.batch_size):
        split[i] = process_test_single(split[i])
    split = tf.stack(split)
    return split

def network(weight_decay):
    images1 = Input(name='input_1', shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
    images2 = Input(name='input_2', shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
    
        # Tied Convolution
    conv1_1 = keras.layers.Conv2D( 20, [5, 5], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='conv1_1') (images1)
    pool1_1 = keras.layers.MaxPooling2D( [2, 2], [2, 2], name='pool1_1') (conv1_1)
    conv1_2 = keras.layers.Conv2D( 25, [5, 5], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='conv1_2') (pool1_1)
    pool1_2 = keras.layers.MaxPooling2D([2, 2], [2, 2], name='pool1_2') (conv1_2)
    conv2_1 = keras.layers.Conv2D( 20, [5, 5], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='conv2_1') (images2)
    pool2_1 = keras.layers.MaxPooling2D( [2, 2], [2, 2], name='pool2_1') (conv2_1)
    conv2_2 = keras.layers.Conv2D( 25, [5, 5], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='conv2_2') (pool2_1)
    pool2_2 = keras.layers.MaxPooling2D( [2, 2], [2, 2], name='pool2_2') (conv2_2)

    # Cross-Input Neighborhood Differences
    trans = tf.transpose(pool1_2, [0, 3, 1, 2])
    shape = trans.get_shape().as_list()
    m1s = tf.ones([FLAGS.batch_size, shape[1], shape[2], shape[3], 5, 5])
    reshape = tf.reshape(trans, [FLAGS.batch_size, shape[1], shape[2], shape[3], 1, 1])
    f = tf.multiply(reshape, m1s)

    trans = tf.transpose(pool2_2, [0, 3, 1, 2])
    reshape = tf.reshape(trans, [1, FLAGS.batch_size, shape[1], shape[2], shape[3]])
    g = []
    pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
    for i in range(shape[2]):
        for j in range(shape[3]):
            g.append(pad[:,:,:,i:i+5,j:j+5])

    concat = tf.concat(g, axis=0)
    reshape = tf.reshape(concat, [shape[2], shape[3], FLAGS.batch_size, shape[1], 5, 5])
    g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
    reshape1 = tf.reshape(tf.subtract(f, g), [FLAGS.batch_size, shape[1], shape[2] * 5, shape[3] * 5])
    reshape2 = tf.reshape(tf.subtract(g, f), [FLAGS.batch_size, shape[1], shape[2] * 5, shape[3] * 5])
    k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
    k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

    # Patch Summary Features
    l1 = keras.layers.Conv2D(25, [5, 5], (5, 5), activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='l1')(k1)
    l2 = keras.layers.Conv2D(25, [5, 5], (5, 5), activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='l2')(k2)

    # Across-Patch Features
    m1 = keras.layers.Conv2D( 25, [3, 3], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='m1')(l1)
    pool_m1 = keras.layers.MaxPooling2D([2, 2], [2, 2], padding='same', name='pool_m1') (m1)
    m2 = keras.layers.Conv2D( 25, [3, 3], activation=tf.nn.relu,
        kernel_regularizer=keras.regularizers.l2(weight_decay), name='m2')(l2)
    pool_m2 = keras.layers.MaxPooling2D([2, 2], [2, 2], padding='same', name='pool_m2') (m2)

    # Higher-Order Relationships
    concat = keras.layers.Concatenate(axis=3) ([pool_m1, pool_m2])
    reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
    fc1 = keras.layers.Dense( 500, tf.nn.relu, name='fc1') (reshape)
    fc2 = keras.layers.Dense( 2, name='fc2') (fc1)
    fc2 = keras.layers.Activation('softmax')(fc2)
    
    print(fc2.get_shape())
    
    return Model(inputs=[images1,images2], outputs=fc2)

def load_model(model_path):
    weight_decay = 0.0005
    FLAGS.batch_size = 1
    model = network(weight_decay)
    model.load_weights(model_path)
    return model

def check_images(model, image1, image2):

    image1 = cv2.imread(image1)
    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
    image2 = cv2.imread(image2)
    image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
    
    image1 = process_test_single(image1)
    image2 = process_test_single(image2)
#     print(image1.get_shape())
#     test_images = tf.stack([image1, image2])
    pred = model.predict({'input_1':image1,'input_2':image2})
    return bool(not np.argmax(pred[0]))

if __name__ == "__main__":
    
    print('getting ids')
    train_ids = market1501_dataset.get_id(FLAGS.data_dir, 'bounding_box_train')
    gen = market1501_dataset.read_data(FLAGS.data_dir, 'bounding_box_train', train_ids,
            IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
        
    ds = tf.data.Dataset.from_generator(
        lambda: gen ,
        output_types=({"input_1":tf.float32, "input_2":tf.float32}, tf.float32),
        output_shapes=({"input_1":[FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], 
                        "input_2":[FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH,3]},
                       [FLAGS.batch_size, 2])
    )
    print('tf.data loaded')
    
    ds = ds.map(lambda x, y : ( 
        {"input_1" : process_train(x['input_1']), 
         "input_2" : process_train(x['input_2'])},
        y))
                
    print('data mapping working')
    
    test_ids = market1501_dataset.get_id(FLAGS.data_dir, 'bounding_box_test')
    test_gen = market1501_dataset.read_data(FLAGS.data_dir, 'bounding_box_test', test_ids,
                                            IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
    test_ds = tf.data.Dataset.from_generator(
        lambda: test_gen ,
            output_types=({"input_1":tf.float32, "input_2":tf.float32}, tf.float32),
            output_shapes=({"input_1":[FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], 
                            "input_2":[FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH,3]},
                           [FLAGS.batch_size, 2])
        )
    print('test tf.data loaded')

    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1

    weight_decay = 0.0005

    print('Build network')
    model = network(weight_decay)
    # print(model.summary())

    lr = 0.001

    optim = optimizers.Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['acc'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='reid_model.h5', mode='max', monitor='val_acc', verbose=2, save_best_only=True)

    history = model.fit(
        x=ds, epochs=100, verbose=1, callbacks=[checkpoint],
        validation_data=test_ds, steps_per_epoch=500,
        validation_steps=100
    )