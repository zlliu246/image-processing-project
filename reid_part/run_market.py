import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--data_dir", dest="data_dir", type=str, metavar='<str>', default='datasets', help='path to dataset')
# parser.add_argument('-e','--epochs', dest="epochs", type=int, metavar='<int>', default=100, help='max epochs for training')
# parser.add_argument('-b','--batch_size', dest="batch_size", type=int, metavar='<int>', default=32, help='batch_size for training')
# parser.add_argument('-l','--logs', dest="logs_dir", type=str, metavar='<str>',default='logs/', help='path to logs')
# parser.add_argument('-m', '--mode', dest='mode', type=str, metavar='<str>', default='train', help='mode - train, val, or test')
# parser.add_argument('-s', '--save', dest='save', type=str, metavar='<str>', help='mode - train, val, or test')
# parser.add_argument('-g', '--gpu',dest='gpu',type=int,metavar='<int>')

# FLAGS = parser.parse_args()

import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow import keras
import numpy as np
import cv2
import reid_part.market1501_dataset
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

tf.get_logger().setLevel('INFO')

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160

def process_img_train(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
    img = tf.image.random_crop(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.per_image_standardization(img)
#     print(img.get_shape())
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return img

def process_img_test(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.image.per_image_standardization(img)
#     print(img.get_shape())
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return img

def real_process_img_test(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.central_crop(img, 0.8)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.image.per_image_standardization(img)
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return img
    
def process_label(y):
    y.set_shape([2])
    return y

def network(weight_decay, batch_size):
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
    m1s = tf.ones([batch_size, shape[1], shape[2], shape[3], 5, 5])
    reshape = tf.reshape(trans, [batch_size, shape[1], shape[2], shape[3], 1, 1])
    f = tf.multiply(reshape, m1s)

    trans = tf.transpose(pool2_2, [0, 3, 1, 2])
    reshape = tf.reshape(trans, [1, batch_size, shape[1], shape[2], shape[3]])
    g = []
    pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
    for i in range(shape[2]):
        for j in range(shape[3]):
            g.append(pad[:,:,:,i:i+5,j:j+5])

    concat = tf.concat(g, axis=0)
    reshape = tf.reshape(concat, [shape[2], shape[3], batch_size, shape[1], 5, 5])
    g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
    reshape1 = tf.reshape(tf.subtract(f, g), [batch_size, shape[1], shape[2] * 5, shape[3] * 5])
    reshape2 = tf.reshape(tf.subtract(g, f), [batch_size, shape[1], shape[2] * 5, shape[3] * 5])
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
    reshape = tf.reshape(concat, [batch_size, -1])
    fc1 = keras.layers.Dense( 500, tf.nn.relu, name='fc1') (reshape)
    fc2 = keras.layers.Dense( 2, name='fc2') (fc1)
    fc2 = keras.layers.Activation('softmax')(fc2)
    
    print(fc2.get_shape())
    
    return Model(inputs=[images1,images2], outputs=fc2)

def load_model(model_path
              ):
    weight_decay = 0.0005
    model = network(weight_decay, 1)
    
    model.load_weights(model_path)
    
    lr = 0.001

    optim = optimizers.Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['acc'])
    
    return model

@tf.autograph.experimental.do_not_convert
def check_images(model, image, db):
    
    def x():
        output = []
        for i in range(len(db)):
            output.append({'input_1':image, 'input_2':db[i]})
        return output
    
    test_ds = tf.data.Dataset.from_generator(
        lambda: x() ,
            output_types=({"input_1":tf.string, "input_2":tf.string})
    )
        
    test_ds = test_ds.map(lambda x : ( 
        {"input_1" : process_img_test(x['input_1']), 
         "input_2" : process_img_test(x['input_2'])})).batch(1).prefetch(1)
    
    pred = model.predict(x=test_ds, batch_size=1)
    return (1-np.argmax(pred,axis=1)).astype(bool), pred[:,0]

if __name__ == "__main__":
    
    print('getting ids')
    train_ids = market1501_dataset.get_id(FLAGS.data_dir, 'bounding_box_train')
    gen = market1501_dataset.read_data(FLAGS.data_dir, 'bounding_box_train', train_ids)
        
    ds = tf.data.Dataset.from_generator(
        lambda: gen ,
            output_types=({"input_1":tf.string, "input_2":tf.string}, tf.float32)
    )
    print('tf.data loaded')
    
    ds = ds.map(lambda x, y : ( 
        {"input_1" : process_img_train(x['input_1']), 
         "input_2" : real_process_img_train(x['input_2'])},
        process_label(y))).batch(FLAGS.batch_size).prefetch(FLAGS.batch_size)
                
    print('data mapping working')
    
    test_ids = market1501_dataset.get_id(FLAGS.data_dir, 'bounding_box_test')
    test_gen = market1501_dataset.read_data(FLAGS.data_dir, 'bounding_box_test', test_ids)
    test_ds = tf.data.Dataset.from_generator(
        lambda: test_gen ,
            output_types=({"input_1":tf.string, "input_2":tf.string}, tf.float32)
    )
        
    test_ds = test_ds.map(lambda x, y : ( 
        {"input_1" : real_process_img_test(x['input_1']), 
         "input_2" : real_process_img_test(x['input_2'])},
        process_label(y))).batch(FLAGS.batch_size).prefetch(FLAGS.batch_size)
    print('test tf.data loaded')

    weight_decay = 0.0005

    print('Build network')
    model = network(weight_decay, FLAGS.batch_size)
    # print(model.summary())

    lr = 0.001

    optim = optimizers.Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['acc'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.save, mode='max', monitor='val_acc', verbose=2, save_best_only=True)

    history = model.fit(
        x=ds, epochs=100, verbose=1, callbacks=[checkpoint],
        validation_data=test_ds, steps_per_epoch=500,
        validation_steps=100
    )