    ''' Code with Tensorflow 2.0 and Keras for classifiying cifar data in grayscale range.
        Tensorboard implementation for visualising training real-time.
        Compute Graph is also added to tensorboard log_files.
        Only for Tensorflow 2.0 nd tensorboard usage testing,
        not optimized for high oerdiction accuracy.
        Naman D. Singh 14-01-2020
    '''

import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


def train_augment(x: tf.Tensor, y:tf.Tensor):

    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def rand_crop(img):
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    switch = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(switch < 0.5, lambda: tf.image.random_flip_left_right(x), lambda: rand_crop(x)), y

def get_model(c_out, input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,kernel_regularizer=tf.keras.regularizers.l2(0.1), padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.2), padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(rate = 0.2),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3), padding = 'same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2), padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(rate = 0.2),

        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2), padding = 'same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(rate = 0.3),
        layers.Conv2D(128, (3, 3), activation='relu',padding = 'same'),
        layers.BatchNormalization(),
	layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2), padding = 'same'),
        layers.BatchNormalization(),
        layers.Dropout(rate = 0.3),
        layers.Flatten(),
	layers.Dense(960), #### remove this for previous version
        layers.Dense(128),  ### 128
        layers.Dense(c_out),
    ])
    return model


def main():
    data_dir = ""
    save_dir = ""
    batch_size = 64
    epochs = 25
    eval_data_size = 5000
    (x_train, y_train), (x_test) = np.load(data_dir + "", allow_pickle=True)

    x_train = np.expand_dims(x_train, 4).astype('float32') / 255
    x_eval = x_train[0:eval_data_size, ...]
    x_train = x_train[eval_data_size:, ...]
    y_eval = y_train[0:eval_data_size, ...]
    y_train = y_train[eval_data_size:, ...]
    x_test = np.expand_dims(x_test, 4).astype('float32') / 255
    num_classes = np.max(y_train) + 1

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(train_augment).batch(batch_size).prefetch(2)
    eval_set = tf.data.Dataset.from_tensor_slices((x_eval, y_eval)).batch(batch_size).prefetch(2)
    test_set = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size).prefetch(2)

    model = get_model(num_classes, [32, 32, 1])
    model.summary()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer2 = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)  ## previous version 0.005, 1e-6, 0.9



    # tensorboard writer
    logdir = save_dir + "/tb/%d/" % time.time()
    writer = tf.summary.create_file_writer(logdir)  # Needed for Tensorboard logging

    @tf.function
    def graph_trace_function(x, y):
        with tf.GradientTape():
            logits = model(x, training=True)
            loss_value = loss(y, logits)
            # when we add gradients here the graph gets quite uninterpretable
        return loss_value

    # TODO use a tf file writer in combination with tf.summary.trace_on() tf.summary.trace_export()
    #  graph_trace_function() and zero tensor inputs to save the graph

    # Sample data for your function.
    x = tf.zeros((32, 32, 32,1))
    y = tf.zeros((32,1))

    tf.summary.trace_on(graph=True, profiler=True)
    z = graph_trace_function(x, y)
    with writer.as_default():
        tf.summary.trace_export(
          name="graph_trace",
          step=0,
          profiler_outdir=logdir)

    #     For visualising training and eval accuracy and loss graphs

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'log/gradient_tape/' + current_time + '/train'
    test_log_dir = 'log/gradient_tape/' + current_time + '/eval'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for e in range(epochs):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_loss = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(train_set):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss(y, logits)

            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer2.apply_gradients(zip(gradients, model.trainable_weights))
            train_accuracy.update_state(y, logits)
            train_loss.update_state(loss_value)
        #### Writing training  parameters per epoch
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=e)
        tf.print("-" * 50, output_stream=sys.stdout)
        eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        eval_loss = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(eval_set):
            logits = model(x, training=False)
            loss_value = loss(y, logits)
            eval_accuracy.update_state(y, logits)
            eval_loss.update_state(loss_value)
        ### Writing eval parameter per epoch
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', eval_loss.result(), step=e)
            tf.summary.scalar('accuracy', eval_accuracy.result(), step=e)
        tf.print("epoch {0:d} \ntrain_loss: {1:2.5f} \ntrain_accuracy: {2:2.5f}".format(e, train_loss.result(), train_accuracy.result()), output_stream=sys.stdout)
        tf.print("eval_loss: {0:2.5f} \neval_accuracy: {1:2.5f}".format(eval_loss.result(),
                                                                         eval_accuracy.result()),
                 output_stream=sys.stdout)

if __name__ == '__main__':
    main()
