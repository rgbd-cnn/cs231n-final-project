import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.inception_resnet import inception_res_features


def two_branch_cnn(X, A, B, C, num_classes, is_training, branch1=None,
                   branch2=None, keep_prob=None):

    with tf.variable_scope(branch1):
        if branch1 == "IR2d":
            feature1 = inception_res_features(X[:, :, :, :3], A, B, C, is_training, keep_prob=keep_prob)
        elif branch1 == "IR3d":
            feature1 = inception_res_features(X, A, B, C, is_training, keep_prob=keep_prob)
        elif branch1 == "IRd":
            feature1 = inception_res_features(X[:, :, :, 3:], A, B, C, is_training, keep_prob=keep_prob)
        else:
            raise Exception()

    with tf.variable_scope(branch2):
        if branch2 == "IR2d":
            feature2 = inception_res_features(X[:, :, :, :3], A, B, C, is_training, keep_prob=keep_prob)
        elif branch2 == "IR3d":
            feature2 = inception_res_features(X, A, B, C, is_training, keep_prob=keep_prob)
        elif branch2 == "IRd":
            feature2 = inception_res_features(X[:, :, :, 3:], A, B, C, is_training, keep_prob=keep_prob)
        else:
            raise Exception()

    stacked = tf.concat([feature1, feature2], 1)
    # print(stacked.get_shape().as_list())

    output = slim.fully_connected(stacked, num_classes, activation_fn=None)

    # print(output.get_shape().as_list())
    return output


def setup_two_branch_cnn_model(image_size, num_classes, A, B, C,
                               learning_rate=1e-3, branch1=None, branch2=None,
                               reg=0.0, keep_prob=None):
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    size = [None] + image_size
    X = tf.placeholder(tf.float32, size)
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Define Output and Calculate Loss
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(reg)):
        y_out = two_branch_cnn(X, A, B, C, num_classes, is_training,
                               branch1=branch1, branch2=branch2,
                               keep_prob=keep_prob)

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(y, num_classes),
        logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    loss = mean_loss # + tf.add_n(slim.losses.get_regularization_losses())

    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08)

    # Required for Batch Normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(loss)

    # Store Model in Dictionary
    model = {}
    model['X'] = X
    model['y'] = y
    model['is_training'] = is_training
    model['y_out'] = y_out
    model['loss_val'] = loss
    model['train_step'] = train_step

    return model
