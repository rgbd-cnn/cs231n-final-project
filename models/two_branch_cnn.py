import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.inception_resnet import inception_res_features


def two_branch_cnn(X, A, B, C, num_classes, is_training, branch1=None,
                   branch2=None):
    branches = {
        "IR2d": inception_res_features(X[:, :, :, :3], A, B, C, is_training),
        "IR3d": inception_res_features(X, A, B, C, is_training),
        "IRd": inception_res_features(X[:, :, :, 3:], A, B, C, is_training)
    }

    with tf.variable_scope(branch1):
        feature1 = branches[branch1]

    with tf.variable_scope(branch2):
        feature2 = branches[branch2]

    stacked = tf.concat([feature1, feature2], 1)
    print(stacked.get_shape().as_list())

    output = slim.fully_connected(stacked, num_classes, activation_fn=None)

    print(output.get_shape().as_list())
    return output


def setup_two_branch_cnn_model(image_size, num_classes, A, B, C,
                               learning_rate=1e-3, branch1=None, branch2=None):
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    size = [None] + image_size
    X = tf.placeholder(tf.float32, size)
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Define Output and Calculate Loss
    y_out = two_branch_cnn(X, A, B, C, num_classes, is_training,
                           branch1=branch1, branch2=branch2)

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
