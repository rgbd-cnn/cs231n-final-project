import tensorflow as tf
from models.inception_resnet import inception_res_features
import tensorflow.contrib.slim as slim

def two_branch_cnn(X, A, B, C, num_classes, is_training):
    inception_res_features_3d = inception_res_features(X, A, B, C, is_training)

    inception_res_features_2d = inception_res_features(X[:, :, :, :3], A, B, C, is_training)

    stacked = tf.stack([inception_res_features_2d, inception_res_features_3d], axis=1)

    output = slim.fully_connected(stacked, num_classes, activation_fn=None)

    return output

def setup_two_branch_cnn_model(image_size, num_classes, A, B, C, learning_rate=1e-3):
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    size = [None] + image_size
    X = tf.placeholder(tf.float32, size)
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Define Output and Calculate Loss
    y_out = two_branch_cnn(X, A, B, C, num_classes, is_training)

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(y, num_classes),
        logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08)

    # Required for Batch Normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)

    # Store Model in Dictionary
    model = {}
    model['X'] = X
    model['y'] = y
    model['is_training'] = is_training
    model['y_out'] = y_out
    model['loss_val'] = mean_loss
    model['train_step'] = train_step

    return model
